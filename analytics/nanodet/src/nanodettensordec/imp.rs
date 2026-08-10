use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use byte_slice_cast::AsSliceOf;
use gst::{glib, prelude::*, subclass::prelude::*};
use gst_analytics::prelude::*;
use gst_base::subclass::prelude::*;

use super::decode::{self, CONTRACTS, Contract, NUM_CLASSES};

const DEFAULT_TENSOR_ID: &str = "nanodet-output";
const TENSOR_GROUP_ID: &str = "gstsmith-nanodet";
const DEFAULT_SCORE_THRESHOLD: f32 = 0.3;
const DEFAULT_IOU_THRESHOLD: f32 = 0.6;
const DEFAULT_MAX_DETECTIONS: u32 = 100;
const MAX_DETECTIONS_LIMIT: u32 = 1_000;

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "nanodettensordec",
        gst::DebugColorFlags::empty(),
        Some("NanoDet tensor decoder"),
    )
});

#[derive(Clone)]
struct Settings {
    tensor_id: String,
    label_file: Option<PathBuf>,
    score_threshold: f32,
    iou_threshold: f32,
    max_detections: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            tensor_id: DEFAULT_TENSOR_ID.to_owned(),
            label_file: None,
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            iou_threshold: DEFAULT_IOU_THRESHOLD,
            max_detections: DEFAULT_MAX_DETECTIONS,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorKind {
    Float32,
    Float16,
}

impl TensorKind {
    fn from_caps_name(name: &str) -> Option<Self> {
        match name {
            "float32" => Some(Self::Float32),
            "float16" => Some(Self::Float16),
            _ => None,
        }
    }

    const fn caps_name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
        }
    }

    const fn data_type(self) -> gst_analytics::TensorDataType {
        match self {
            Self::Float32 => gst_analytics::TensorDataType::Float32,
            Self::Float16 => gst_analytics::TensorDataType::Float16,
        }
    }

    const fn element_size(self) -> usize {
        match self {
            Self::Float32 => std::mem::size_of::<f32>(),
            Self::Float16 => std::mem::size_of::<u16>(),
        }
    }
}

const TENSOR_KINDS: [TensorKind; 2] = [TensorKind::Float32, TensorKind::Float16];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NegotiatedContract {
    model: Contract,
    tensor_kind: TensorKind,
}

struct State {
    labels: Arc<[glib::Quark; NUM_CLASSES]>,
    tensor_id: glib::Quark,
    score_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    negotiated: Option<NegotiatedContract>,
}

#[derive(Clone, Copy)]
struct DecodeParameters {
    tensor_id: glib::Quark,
    score_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    negotiated: NegotiatedContract,
}

#[derive(Default)]
pub struct NanoDetTensorDec {
    settings: Mutex<Settings>,
    state: Mutex<Option<State>>,
    scratch: Mutex<Vec<decode::Detection>>,
}

#[glib::object_subclass]
impl ObjectSubclass for NanoDetTensorDec {
    const NAME: &'static str = "GstSmithNanoDetTensorDec";
    type Type = super::NanoDetTensorDec;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for NanoDetTensorDec {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("tensor-id")
                    .nick("Tensor ID")
                    .blurb("ID of the NanoDet output tensor")
                    .default_value(Some(DEFAULT_TENSOR_ID))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("label-file")
                    .nick("Label File")
                    .blurb("Optional UTF-8 file containing exactly 80 labels, one per line")
                    .mutable_ready()
                    .build(),
                glib::ParamSpecFloat::builder("score-threshold")
                    .nick("Score Threshold")
                    .blurb("Minimum post-sigmoid class score")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(DEFAULT_SCORE_THRESHOLD)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecFloat::builder("iou-threshold")
                    .nick("IoU Threshold")
                    .blurb("Maximum same-class intersection-over-union before suppression")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(DEFAULT_IOU_THRESHOLD)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("max-detections")
                    .nick("Maximum Detections")
                    .blurb("Maximum analytics detections attached to each frame")
                    .minimum(1)
                    .maximum(MAX_DETECTIONS_LIMIT)
                    .default_value(DEFAULT_MAX_DETECTIONS)
                    .mutable_ready()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let Ok(mut settings) = self.settings.lock() else {
            gst::error!(CAT, imp = self, "settings lock is poisoned");
            return;
        };
        match pspec.name() {
            "tensor-id" => {
                if let Ok(Some(tensor_id)) = value.get::<Option<String>>() {
                    settings.tensor_id = tensor_id;
                }
            }
            "label-file" => {
                if let Ok(path) = value.get::<Option<String>>() {
                    settings.label_file = path.map(PathBuf::from);
                }
            }
            "score-threshold" => {
                if let Ok(threshold) = value.get::<f32>() {
                    settings.score_threshold = threshold;
                }
            }
            "iou-threshold" => {
                if let Ok(threshold) = value.get::<f32>() {
                    settings.iou_threshold = threshold;
                }
            }
            "max-detections" => {
                if let Ok(maximum) = value.get::<u32>() {
                    settings.max_detections = maximum;
                }
            }
            _ => gst::warning!(CAT, imp = self, "unexpected property {}", pspec.name()),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let Ok(settings) = self.settings.lock() else {
            return pspec.default_value().clone();
        };
        match pspec.name() {
            "tensor-id" => settings.tensor_id.to_value(),
            "label-file" => settings
                .label_file
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .to_value(),
            "score-threshold" => settings.score_threshold.to_value(),
            "iou-threshold" => settings.iou_threshold.to_value(),
            "max-detections" => settings.max_detections.to_value(),
            _ => pspec.default_value().clone(),
        }
    }
}

impl GstObjectImpl for NanoDetTensorDec {}

impl ElementImpl for NanoDetTensorDec {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "NanoDet Tensor Decoder",
                "Filter/Analysis/Video",
                "Decodes supported NanoDet-m and NanoDet-Plus tensors into analytics metadata",
                "Nemanja Zbiljic <nemanja.zbiljic@gmail.com>",
            )
        });
        Some(&METADATA)
    }

    #[expect(
        clippy::expect_used,
        reason = "static pad-template construction uses fixed valid names and caps"
    )]
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let sink_caps = supported_sink_caps(None);
            let src_caps = supported_src_caps();
            let sink = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &sink_caps,
            )
            .expect("construct NanoDet decoder sink template");
            let src = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &src_caps,
            )
            .expect("construct NanoDet decoder source template");
            vec![sink, src]
        });
        TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for NanoDetTensorDec {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings.lock().map_err(|_error| {
            gst::error_msg!(gst::LibraryError::Settings, ["settings lock is poisoned"])
        })?;
        if settings.tensor_id.trim().is_empty() {
            return Err(gst::error_msg!(
                gst::LibraryError::Settings,
                ["tensor-id must not be empty"]
            ));
        }
        let labels = if let Some(label_file) = &settings.label_file {
            let contents = std::fs::read_to_string(label_file).map_err(|error| {
                gst::error_msg!(
                    gst::ResourceError::OpenRead,
                    [
                        "failed to read label file {}: {error}",
                        label_file.display()
                    ]
                )
            })?;
            let labels = contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(glib::Quark::from_str)
                .collect::<Vec<_>>();
            if labels.len() != NUM_CLASSES {
                return Err(gst::error_msg!(
                    gst::LibraryError::Settings,
                    [
                        "label file {} contains {} non-empty labels; expected {NUM_CLASSES}",
                        label_file.display(),
                        labels.len()
                    ]
                ));
            }
            labels
        } else {
            (0..NUM_CLASSES)
                .map(|class| glib::Quark::from_str(format!("class-{class}")))
                .collect::<Vec<_>>()
        };
        let labels = labels.try_into().map_err(|labels: Vec<_>| {
            gst::error_msg!(
                gst::LibraryError::Settings,
                ["resolved {} labels; expected {NUM_CLASSES}", labels.len()]
            )
        })?;
        let tensor_id = glib::Quark::from_str(&settings.tensor_id);
        let score_threshold = settings.score_threshold;
        let iou_threshold = settings.iou_threshold;
        let max_detections = settings.max_detections as usize;
        drop(settings);
        *self.state.lock().map_err(|_error| {
            gst::error_msg!(gst::LibraryError::Settings, ["state lock is poisoned"])
        })? = Some(State {
            labels: Arc::new(labels),
            tensor_id,
            score_threshold,
            iou_threshold,
            max_detections,
            negotiated: None,
        });
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        *self.state.lock().map_err(|_error| {
            gst::error_msg!(gst::LibraryError::Settings, ["state lock is poisoned"])
        })? = None;
        Ok(())
    }

    fn transform_caps(
        &self,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        let tensor_id = self.settings.lock().map_or_else(
            |_| DEFAULT_TENSOR_ID.to_owned(),
            |settings| settings.tensor_id.clone(),
        );
        let result = transform_tensor_caps(caps, direction, &tensor_id);
        Some(filter.map_or(result.clone(), |filter| {
            filter.intersect_with_mode(&result, gst::CapsIntersectMode::First)
        }))
    }

    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let tensor_id = self
            .settings
            .lock()
            .map_err(|_error| gst::loggable_error!(CAT, "settings lock is poisoned"))?
            .tensor_id
            .clone();
        let negotiated = negotiated_contract(incaps, &tensor_id).ok_or_else(|| {
            gst::loggable_error!(
                CAT,
                "caps do not contain a supported NanoDet tensor contract: {incaps:?}"
            )
        })?;
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_error| gst::loggable_error!(CAT, "state lock is poisoned"))?;
            let Some(state) = state.as_mut() else {
                return Err(gst::loggable_error!(CAT, "decoder is not started"));
            };
            state.negotiated = Some(negotiated);
        }
        gst::debug!(
            CAT,
            imp = self,
            "negotiated {} {} output",
            negotiated.model.name,
            negotiated.tensor_kind.caps_name()
        );
        self.parent_set_caps(incaps, outcaps)
    }

    fn transform_ip(
        &self,
        buffer: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let (labels, parameters) = {
            let state = self.state.lock().map_err(|_error| gst::FlowError::Error)?;
            let state = state.as_ref().ok_or(gst::FlowError::Flushing)?;
            let negotiated = state.negotiated.ok_or_else(|| {
                gst::element_error!(
                    self.obj(),
                    gst::CoreError::Negotiation,
                    ("NanoDet tensor contract is not negotiated")
                );
                gst::FlowError::NotNegotiated
            })?;
            (
                Arc::clone(&state.labels),
                DecodeParameters {
                    tensor_id: state.tensor_id,
                    score_threshold: state.score_threshold,
                    iou_threshold: state.iou_threshold,
                    max_detections: state.max_detections,
                    negotiated,
                },
            )
        };

        let mut detections = self
            .scratch
            .lock()
            .map_err(|_error| gst::FlowError::Error)?;
        if !decode_tensor(self, buffer, parameters, &mut detections)? {
            return Ok(gst::FlowSuccess::Ok);
        }

        attach_metadata(self, buffer, labels.as_ref(), &detections)?;
        Ok(gst::FlowSuccess::Ok)
    }
}

fn decode_tensor(
    imp: &NanoDetTensorDec,
    buffer: &gst::BufferRef,
    parameters: DecodeParameters,
    detections: &mut Vec<decode::Detection>,
) -> Result<bool, gst::FlowError> {
    let Some(tensor_meta) = find_tensor_meta(buffer, parameters.tensor_id) else {
        gst::trace!(
            CAT,
            imp = imp,
            "no tensor metadata with ID {:?}; passing buffer through",
            parameters.tensor_id.as_str()
        );
        return Ok(false);
    };
    let Some(tensor) = tensor_meta
        .as_slice()
        .iter()
        .find(|tensor| tensor.id() == parameters.tensor_id)
    else {
        return Ok(false);
    };
    validate_tensor(tensor, parameters.negotiated).map_err(|error| {
        gst::element_error!(
            imp.obj(),
            gst::StreamError::Format,
            ("invalid NanoDet tensor"),
            ["{error}"]
        );
        gst::FlowError::Error
    })?;
    let map = tensor.data().map_readable().map_err(|error| {
        gst::element_error!(
            imp.obj(),
            gst::ResourceError::Read,
            ("failed to map NanoDet tensor"),
            ["{error}"]
        );
        gst::FlowError::Error
    })?;
    let decoded = match parameters.negotiated.tensor_kind {
        TensorKind::Float32 => map.as_slice_of::<f32>().map_or_else(
            |error| {
                Err(format!(
                    "NanoDet tensor bytes are not an aligned float32 slice: {error}"
                ))
            },
            |values| {
                decode::decode(
                    values,
                    parameters.negotiated.model,
                    parameters.score_threshold,
                    parameters.iou_threshold,
                    parameters.max_detections,
                    detections,
                )
            },
        ),
        TensorKind::Float16 => map.as_slice_of::<u16>().map_or_else(
            |error| {
                Err(format!(
                    "NanoDet tensor bytes are not an aligned float16 slice: {error}"
                ))
            },
            |values| {
                decode::decode_float16(
                    values,
                    parameters.negotiated.model,
                    parameters.score_threshold,
                    parameters.iou_threshold,
                    parameters.max_detections,
                    detections,
                )
            },
        ),
    };
    decoded.map_err(|error| {
        gst::element_error!(
            imp.obj(),
            gst::StreamError::Format,
            ("failed to decode NanoDet tensor"),
            ["{error}"]
        );
        gst::FlowError::Error
    })?;
    Ok(true)
}

fn attach_metadata(
    imp: &NanoDetTensorDec,
    buffer: &mut gst::BufferRef,
    labels: &[glib::Quark],
    detections: &[decode::Detection],
) -> Result<(), gst::FlowError> {
    if detections.is_empty() {
        return Ok(());
    }
    let mut relation = gst_analytics::AnalyticsRelationMeta::add(buffer);
    for detection in detections {
        let Some(label) = labels.get(detection.class).copied() else {
            gst::error!(
                CAT,
                imp = imp,
                "decoded class {} has no configured label",
                detection.class
            );
            return Err(gst::FlowError::Error);
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coordinates are finite and clamped to the supported input frame"
        )]
        let (x, y, width, height) = (
            detection.x1 as i32,
            detection.y1 as i32,
            (detection.x2 - detection.x1) as i32,
            (detection.y2 - detection.y1) as i32,
        );
        let object_id = relation
            .add_od_mtd(label, x, y, width.max(1), height.max(1), detection.score)
            .map_err(|error| {
                gst::error!(CAT, imp = imp, "failed to add object metadata: {error}");
                gst::FlowError::Error
            })?
            .id();
        let classification_id = relation
            .add_one_cls_mtd(detection.score, label)
            .map_err(|error| {
                gst::error!(
                    CAT,
                    imp = imp,
                    "failed to add classification metadata: {error}"
                );
                gst::FlowError::Error
            })?
            .id();
        relation
            .set_relation(
                gst_analytics::RelTypes::RELATE_TO,
                object_id,
                classification_id,
            )
            .map_err(|error| {
                gst::error!(
                    CAT,
                    imp = imp,
                    "failed to relate analytics metadata: {error}"
                );
                gst::FlowError::Error
            })?;
    }
    Ok(())
}

fn find_tensor_meta(
    buffer: &gst::BufferRef,
    tensor_id: glib::Quark,
) -> Option<gst::MetaRef<'_, gst_analytics::TensorMeta>> {
    buffer
        .iter_meta::<gst_analytics::TensorMeta>()
        .find(|meta| {
            meta.as_slice()
                .iter()
                .any(|tensor| tensor.id() == tensor_id)
        })
}

fn negotiated_contract(caps: &gst::CapsRef, tensor_id: &str) -> Option<NegotiatedContract> {
    caps.iter().find_map(|structure| {
        let width = usize::try_from(structure.get::<i32>("width").ok()?).ok()?;
        let height = usize::try_from(structure.get::<i32>("height").ok()?).ok()?;
        let groups = structure.get::<gst::Structure>("tensors").ok()?;
        let descriptors = groups.get::<gst::UniqueList>(TENSOR_GROUP_ID).ok()?;

        descriptors.as_slice().iter().find_map(|value| {
            let descriptor_caps = value.get::<gst::Caps>().ok()?;
            descriptor_caps.iter().find_map(|descriptor| {
                if descriptor.get::<String>("tensor-id").ok()?.as_str() != tensor_id
                    || descriptor.get::<String>("dims-order").ok()?.as_str() != "row-major"
                {
                    return None;
                }
                let tensor_kind =
                    TensorKind::from_caps_name(descriptor.get::<String>("type").ok()?.as_str())?;
                let dimensions = descriptor.get::<gst::Array>("dims").ok()?;
                let dimensions = dimensions
                    .as_slice()
                    .iter()
                    .map(|value| usize::try_from(value.get::<i32>().ok()?).ok())
                    .collect::<Option<Vec<_>>>()?;
                let model = decode::contract_for_dims(&dimensions)?;
                (width == model.input_size && height == model.input_size)
                    .then_some(NegotiatedContract { model, tensor_kind })
            })
        })
    })
}

fn validate_tensor(
    tensor: &gst_analytics::Tensor,
    negotiated: NegotiatedContract,
) -> Result<(), String> {
    if tensor.data_type() != negotiated.tensor_kind.data_type() {
        return Err(format!(
            "tensor type is {:?}; negotiated {}",
            tensor.data_type(),
            negotiated.tensor_kind.caps_name()
        ));
    }
    if tensor.dims_order() != gst_analytics::TensorDimOrder::RowMajor {
        return Err(format!(
            "tensor dimension order is {:?}; expected RowMajor",
            tensor.dims_order()
        ));
    }
    let expected_dims = negotiated.model.dims();
    if tensor.dims() != expected_dims {
        return Err(format!(
            "tensor dimensions are {:?}; expected {:?}",
            tensor.dims(),
            expected_dims
        ));
    }
    let expected_bytes = negotiated
        .model
        .elements()
        .checked_mul(negotiated.tensor_kind.element_size())
        .ok_or_else(|| "expected tensor byte length overflowed".to_owned())?;
    if tensor.data().size() != expected_bytes {
        return Err(format!(
            "tensor contains {} bytes; expected {expected_bytes}",
            tensor.data().size()
        ));
    }
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "all supported NanoDet dimensions fit in i32"
)]
fn tensor_caps(contract: Contract, tensor_kind: TensorKind, tensor_id: Option<&str>) -> gst::Caps {
    let mut structure = gst::Structure::builder("tensor/strided")
        .field(
            "dims",
            gst::Array::from_values(
                contract
                    .dims()
                    .map(|dimension| (dimension as i32).to_send_value()),
            ),
        )
        .field("dims-order", "row-major")
        .field("type", tensor_kind.caps_name())
        .build();
    if let Some(tensor_id) = tensor_id {
        structure.set("tensor-id", tensor_id);
    }
    gst::Caps::builder_full().structure(structure).build()
}

fn supported_sink_caps(tensor_id: Option<&str>) -> gst::Caps {
    let mut result = gst::Caps::new_empty();
    for contract in CONTRACTS {
        for tensor_kind in TENSOR_KINDS {
            let mut groups = gst::Structure::new_empty("tensorgroups");
            groups.set(
                TENSOR_GROUP_ID,
                gst::UniqueList::new([tensor_caps(contract, tensor_kind, tensor_id)]),
            );
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                reason = "supported input sizes fit in i32"
            )]
            let caps = gst_video::VideoCapsBuilder::new()
                .width(contract.input_size as i32)
                .height(contract.input_size as i32)
                .field("tensors", groups)
                .build();
            result.make_mut().append(caps);
        }
    }
    result
}

fn supported_src_caps() -> gst::Caps {
    let mut result = gst::Caps::new_empty();
    for input_size in [320, 416] {
        let caps = gst_video::VideoCapsBuilder::new()
            .width(input_size)
            .height(input_size)
            .build();
        result.make_mut().append(caps);
    }
    result
}

fn transform_tensor_caps(
    caps: &gst::Caps,
    direction: gst::PadDirection,
    tensor_id: &str,
) -> gst::Caps {
    if direction == gst::PadDirection::Src {
        let mut result = gst::Caps::new_empty();
        for (structure, features) in caps.iter_with_features() {
            for contract in CONTRACTS {
                if !structure_supports_size(structure, contract.input_size) {
                    continue;
                }
                for tensor_kind in TENSOR_KINDS {
                    let mut candidate = structure.to_owned();
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_possible_wrap,
                        reason = "supported input sizes fit in i32"
                    )]
                    {
                        candidate.set("width", contract.input_size as i32);
                        candidate.set("height", contract.input_size as i32);
                    }
                    add_required_tensor(&mut candidate, tensor_id, contract, tensor_kind);
                    result
                        .make_mut()
                        .append_structure_full(candidate, Some(features.to_owned()));
                }
            }
        }
        result
    } else {
        let mut result = caps.copy();
        for structure in result.make_mut().iter_mut() {
            remove_required_tensor(structure, tensor_id);
        }
        result.simplify();
        result
    }
}

fn structure_supports_size(structure: &gst::StructureRef, input_size: usize) -> bool {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "supported input sizes fit in i32"
    )]
    let input_size = input_size as i32;
    structure
        .get::<i32>("width")
        .map_or(true, |width| width == input_size)
        && structure
            .get::<i32>("height")
            .map_or(true, |height| height == input_size)
}

fn add_required_tensor(
    structure: &mut gst::StructureRef,
    tensor_id: &str,
    contract: Contract,
    tensor_kind: TensorKind,
) {
    let mut groups = structure
        .get::<gst::Structure>("tensors")
        .unwrap_or_else(|_| gst::Structure::new_empty("tensorgroups"));
    let descriptors = groups
        .get::<gst::UniqueList>(TENSOR_GROUP_ID)
        .unwrap_or_default();
    let mut retained = descriptors
        .as_slice()
        .iter()
        .filter(|value| {
            value
                .get::<gst::Caps>()
                .map_or(true, |caps| !caps_has_tensor_id(&caps, tensor_id))
        })
        .cloned()
        .collect::<Vec<_>>();
    retained.push(tensor_caps(contract, tensor_kind, Some(tensor_id)).to_send_value());
    groups.set(TENSOR_GROUP_ID, gst::UniqueList::from_values(retained));
    structure.set("tensors", groups);
}

fn remove_required_tensor(structure: &mut gst::StructureRef, tensor_id: &str) {
    let Ok(mut groups) = structure.get::<gst::Structure>("tensors") else {
        return;
    };
    let Ok(descriptors) = groups.get::<gst::UniqueList>(TENSOR_GROUP_ID) else {
        return;
    };
    let retained = descriptors
        .as_slice()
        .iter()
        .filter(|value| {
            value
                .get::<gst::Caps>()
                .map_or(true, |caps| !caps_has_tensor_id(&caps, tensor_id))
        })
        .cloned()
        .collect::<Vec<_>>();
    if retained.is_empty() {
        groups.remove_field(TENSOR_GROUP_ID);
    } else {
        groups.set(TENSOR_GROUP_ID, gst::UniqueList::from_values(retained));
    }
    if groups.n_fields() == 0 {
        structure.remove_field("tensors");
    } else {
        structure.set("tensors", groups);
    }
}

fn caps_has_tensor_id(caps: &gst::CapsRef, tensor_id: &str) -> bool {
    caps.iter().any(|structure| {
        structure
            .get::<String>("tensor-id")
            .is_ok_and(|candidate| candidate == tensor_id)
    })
}

#[cfg(test)]
mod tests {
    use super::super::decode::CHANNELS;
    use super::*;

    fn video_caps_with_groups(groups: gst::Structure, input_size: i32) -> gst::Caps {
        gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("width", input_size)
            .field("height", input_size)
            .field("tensors", groups)
            .build()
    }

    #[test]
    fn supported_caps_cover_all_contracts_and_tensor_types() {
        gst::init().expect("initialize GStreamer");
        let caps = supported_sink_caps(Some("custom-output"));
        assert_eq!(caps.size(), CONTRACTS.len() * TENSOR_KINDS.len());

        for contract in CONTRACTS {
            for tensor_kind in TENSOR_KINDS {
                assert!(caps.iter().any(|structure| {
                    let candidate = gst::Caps::builder_full()
                        .structure(structure.to_owned())
                        .build();
                    negotiated_contract(&candidate, "custom-output")
                        == Some(NegotiatedContract {
                            model: contract,
                            tensor_kind,
                        })
                }));
            }
        }
    }

    #[test]
    fn caps_add_exact_contracts_and_remove_only_configured_tensor() {
        gst::init().expect("initialize GStreamer");
        let mut groups = gst::Structure::new_empty("tensorgroups");
        groups.set(
            "other-model",
            gst::UniqueList::new([tensor_caps(
                CONTRACTS[0],
                TensorKind::Float32,
                Some("custom-output"),
            )]),
        );
        let plain = video_caps_with_groups(groups, 320);
        let sink = transform_tensor_caps(&plain, gst::PadDirection::Src, "custom-output");
        assert_eq!(sink.size(), 4);
        assert!(sink.iter().all(|structure| {
            structure
                .get::<gst::Structure>("tensors")
                .is_ok_and(|sink_groups| {
                    sink_groups.has_field("other-model") && sink_groups.has_field(TENSOR_GROUP_ID)
                })
        }));

        let source = transform_tensor_caps(&sink, gst::PadDirection::Sink, "custom-output");
        assert!(source.iter().all(|structure| {
            structure
                .get::<gst::Structure>("tensors")
                .is_ok_and(|source_groups| {
                    source_groups.has_field("other-model")
                        && !source_groups.has_field(TENSOR_GROUP_ID)
                })
        }));
    }

    #[test]
    fn validates_tensor_type_order_shape_and_byte_length() {
        gst::init().expect("initialize GStreamer");
        let make = |data_type, order, dims: &[usize], byte_len| {
            gst_analytics::Tensor::new_simple(
                glib::Quark::from_str(DEFAULT_TENSOR_ID),
                data_type,
                gst::Buffer::with_size(byte_len).expect("allocate tensor"),
                order,
                dims,
            )
        };
        for model in CONTRACTS {
            for tensor_kind in TENSOR_KINDS {
                let negotiated = NegotiatedContract { model, tensor_kind };
                let valid = make(
                    tensor_kind.data_type(),
                    gst_analytics::TensorDimOrder::RowMajor,
                    &model.dims(),
                    model.elements() * tensor_kind.element_size(),
                );
                assert_eq!(validate_tensor(&valid, negotiated), Ok(()));
            }
        }

        let negotiated = NegotiatedContract {
            model: CONTRACTS[0],
            tensor_kind: TensorKind::Float32,
        };
        assert_ne!(
            validate_tensor(
                &make(
                    gst_analytics::TensorDataType::Uint8,
                    gst_analytics::TensorDimOrder::RowMajor,
                    &negotiated.model.dims(),
                    negotiated.model.elements(),
                ),
                negotiated
            ),
            Ok(())
        );
        assert_ne!(
            validate_tensor(
                &make(
                    gst_analytics::TensorDataType::Float32,
                    gst_analytics::TensorDimOrder::ColMajor,
                    &negotiated.model.dims(),
                    negotiated.model.elements() * 4,
                ),
                negotiated
            ),
            Ok(())
        );
        assert_ne!(
            validate_tensor(
                &make(
                    gst_analytics::TensorDataType::Float32,
                    gst_analytics::TensorDimOrder::RowMajor,
                    &[1, negotiated.model.points, CHANNELS - 1],
                    negotiated.model.points * (CHANNELS - 1) * 4,
                ),
                negotiated
            ),
            Ok(())
        );
        let mut wrong_length = make(
            gst_analytics::TensorDataType::Float32,
            gst_analytics::TensorDimOrder::RowMajor,
            &negotiated.model.dims(),
            negotiated.model.elements() * 4,
        );
        wrong_length
            .data_mut()
            .set_size(negotiated.model.elements() * 4 - 1);
        assert_ne!(validate_tensor(&wrong_length, negotiated), Ok(()));
    }
}
