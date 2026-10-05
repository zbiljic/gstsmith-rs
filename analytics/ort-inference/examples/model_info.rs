//! Inspect local ONNX metadata and optionally run ortinference's CPU startup checks.

use std::error::Error;
use std::path::Path;
use std::process::ExitCode;

use gst::prelude::*;
use ort::session::Session;
use ort::value::{TensorElementType, ValueType};

const USAGE: &str = "Usage: model_info MODEL.onnx [MODEL.modelinfo video|tensor-meta]";

fn describe_value(value: &ValueType) -> String {
    let ValueType::Tensor {
        ty,
        shape,
        dimension_symbols,
    } = value
    else {
        return format!("{value:?}; unsupported plugin value (requires a tensor)");
    };
    let dims = shape
        .iter()
        .enumerate()
        .map(|(index, dim)| {
            if *dim >= 0 {
                dim.to_string()
            } else {
                dimension_symbols
                    .get(index)
                    .filter(|symbol| !symbol.is_empty())
                    .map_or_else(|| "?".to_owned(), String::clone)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mut summary = format!("{} [{dims}]", format!("{ty:?}").to_ascii_lowercase());
    if !matches!(
        ty,
        TensorElementType::Float16
            | TensorElementType::Float32
            | TensorElementType::Float64
            | TensorElementType::Int8
            | TensorElementType::Int16
            | TensorElementType::Int32
            | TensorElementType::Int64
            | TensorElementType::Uint8
            | TensorElementType::Uint16
            | TensorElementType::Uint32
            | TensorElementType::Uint64
    ) {
        summary.push_str("; unsupported plugin scalar type");
    }
    if shape.is_empty()
        || shape
            .iter()
            .any(|dim| *dim == 0 || *dim > i64::from(i32::MAX))
    {
        summary.push_str("; unsupported plugin shape (requires nonempty positive axes within i32)");
    }
    summary
}

fn check_contract(model: &Path, info: &Path, mode: &str) -> Result<(), Box<dyn Error>> {
    if !matches!(mode, "video" | "tensor-meta") {
        return Err(format!("invalid input mode {mode:?}; {USAGE}").into());
    }
    // Starting the actual element reuses model-info parsing, image validation,
    // session options, name/type/shape checks, and output selection. No buffers
    // or source are attached, so this never executes the model.
    let element = gst::ElementFactory::make("ortinference")
        .property(
            "model-file",
            model.to_str().ok_or("model path is not UTF-8")?,
        )
        .property(
            "model-info-file",
            info.to_str().ok_or("model-info path is not UTF-8")?,
        )
        .property_from_str("input-mode", mode)
        .build()?;
    let pipeline = gst::Pipeline::new();
    pipeline.add(&element)?;
    let bus = pipeline.bus().ok_or("pipeline has no bus")?;
    let started = pipeline.set_state(gst::State::Paused);
    let error = bus
        .pop_filtered(&[gst::MessageType::Error])
        .map(|message| match message.view() {
            gst::MessageView::Error(error) => {
                format!("{}: {}", error.error(), error.debug().unwrap_or_default())
            }
            _ => format!("{message:?}"),
        });
    pipeline.set_state(gst::State::Null)?;
    if let Some(error) = error {
        return Err(format!("Startup check: FAIL ({mode}, CPU): {error}").into());
    }
    started?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.as_slice() == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let (model, contract) = match args.as_slice() {
        [model] => (model, None),
        [model, info, mode] if matches!(mode.as_str(), "video" | "tensor-meta") => {
            (model, Some((info, mode)))
        }
        _ => return Err(USAGE.into()),
    };
    let session = Session::builder()?
        .with_execution_providers([ort::ep::CPU::default().build().error_on_failure()])?
        .commit_from_file(model)
        .map_err(|error| format!("CPU session load failed: {error}"))?;
    println!("Model: {model}");
    {
        let metadata = session.metadata()?;
        for (label, value) in [
            ("Name", metadata.name()),
            ("Producer", metadata.producer()),
            ("Description", metadata.description()),
        ] {
            if let Some(value) = value.filter(|value| !value.is_empty()) {
                println!("{label}: {value}");
            }
        }
        let mut keys = metadata.custom_keys()?;
        keys.sort();
        if !keys.is_empty() {
            println!("\nMetadata:");
            for key in keys {
                if let Some(value) = metadata.custom(&key) {
                    println!("  {key}: {value}");
                }
            }
        }
    }
    let mut dynamic = false;
    for (label, values) in [("Inputs", session.inputs()), ("Outputs", session.outputs())] {
        println!("\n{label}:");
        let width = values
            .iter()
            .map(|value| value.name().chars().count())
            .max()
            .unwrap_or(0);
        for value in values {
            println!(
                "  {:<width$}  {}",
                value.name(),
                describe_value(value.dtype())
            );
            if let ValueType::Tensor { shape, .. } = value.dtype() {
                dynamic |= shape.iter().any(|dim| *dim < 0);
            }
        }
    }
    if dynamic {
        println!("\nDimensions: names are symbolic; ? is unknown.");
        println!("Bind them in .modelinfo, or use -1 in tensor-meta mode.");
    }
    drop(session);
    if let Some((info, mode)) = contract {
        println!("\nContract: {info}\nMode: {mode}");
        gst::init()?;
        gstortinference::plugin_register_static()?;
        check_contract(Path::new(model), Path::new(info), mode)?;
        println!("Startup check: PASS (CPU; inference not executed)");
    } else {
        println!("\nStartup check: NOT CHECKED (no .modelinfo supplied; inference not executed)");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_symbolic_and_unsupported_values() {
        let tensor = ValueType::Tensor {
            ty: TensorElementType::Bool,
            shape: [-1_i64, -1, 3].into(),
            dimension_symbols: ort::value::SymbolicDimensions::new([
                "seq".to_owned(),
                String::new(),
                String::new(),
            ]),
        };
        let description = describe_value(&tensor);
        assert!(description.contains("bool [seq, ?, 3]"), "{description}");
        assert!(
            description.contains("unsupported plugin scalar type"),
            "{description}"
        );
        assert!(
            describe_value(&ValueType::Sequence(Box::new(tensor)))
                .contains("unsupported plugin value")
        );
        for shape in [vec![], vec![0], vec![i64::from(i32::MAX) + 1]] {
            let tensor = ValueType::Tensor {
                ty: TensorElementType::Float32,
                dimension_symbols: ort::value::SymbolicDimensions::empty(shape.len()),
                shape: shape.into(),
            };
            assert!(describe_value(&tensor).contains("unsupported plugin shape"));
        }
    }

    #[test]
    fn checks_real_startup_and_reports_contract_failures() -> Result<(), Box<dyn Error>> {
        gst::init()?;
        gstortinference::plugin_register_static()?;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inference-common/tests/fixtures");
        for (name, mode) in [
            ("identity", "video"),
            ("masked-frames", "video"),
            ("masked-sequence", "tensor-meta"),
            ("tensor-axes", "tensor-meta"),
            ("tensor-nonzero", "tensor-meta"),
            ("image-reshape", "video"),
        ] {
            check_contract(
                &root.join(format!("{name}.onnx")),
                &root.join(format!("{name}.onnx.modelinfo")),
                mode,
            )?;
        }
        let model = root.join("identity.onnx");
        let original = std::fs::read_to_string(root.join("identity.onnx.modelinfo"))?;
        let file = tempfile::NamedTempFile::new()?;
        std::fs::write(
            file.path(),
            original.replace("dir=output", "dir=output\ndims-order=col-major"),
        )?;
        check_contract(&model, file.path(), "video")?;
        for (contents, expected) in [
            (original.replace("[x]", "[missing]"), "not declared"),
            (original.replace("[y]", "[missing]"), "not a model output"),
            (
                original.replace("type=float32", "type=uint8"),
                "scalar type mismatch",
            ),
            (
                original.replace("dims=1,1,2,3", "dims=1,1,4,3"),
                "dimensions mismatch",
            ),
            (
                original.replace("dims=1,1,2,3", "dims=2,1,2,3"),
                "batch dimension of one",
            ),
            (
                original.replace("dims=1,1,2,3", "dims=1,3,2,3"),
                "unambiguous",
            ),
            (
                original.replace("dims=1,1,2,3", "dims=1,1,?,3"),
                "invalid dimension",
            ),
            (
                original.replace("dir=output", "dir=output\ndims-order=unknown"),
                "unsupported dims-order",
            ),
            (
                original.replace("type=float32", "type=bool"),
                "unsupported scalar type",
            ),
        ] {
            std::fs::write(file.path(), contents)?;
            let error = check_contract(&model, file.path(), "video")
                .err()
                .ok_or("invalid contract passed")?;
            if !error.to_string().contains(expected) {
                return Err(format!("expected {expected:?}, got {error}").into());
            }
        }
        Ok(())
    }
}
