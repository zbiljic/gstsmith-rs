use crate::model_info::{ScalarType, TensorDescription};

#[derive(Debug)]
pub enum InputTensor {
    Float32(Vec<f32>),
    Uint8(Vec<u8>),
}

#[derive(Debug)]
pub struct OwnedTensor {
    pub description: TensorDescription,
    /// Actual dimensions of this tensor, never wildcard declarations.
    pub dims: Vec<usize>,
    pub bytes: Vec<u8>,
}

pub trait Engine: Send {
    fn run(&self, input: InputTensor) -> Result<Vec<OwnedTensor>, String>;
}

/// Runs a tensor-input model: `inputs` are the model-info's
/// inputs, in model-info order, already validated against it.
pub trait TensorEngine: Send {
    fn run(&self, inputs: &[OwnedTensor]) -> Result<Vec<OwnedTensor>, String>;
}

/// Tensor bytes (little-endian, row-major) decoded into typed values.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorValues {
    /// IEEE 754 half-precision bit patterns.
    Float16(Vec<u16>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Uint8(Vec<u8>),
    Uint16(Vec<u16>),
    Uint32(Vec<u32>),
    Uint64(Vec<u64>),
}

impl TensorValues {
    /// Decode `tensor`'s bytes according to its description.
    pub fn decode(tensor: &OwnedTensor) -> Result<Self, String> {
        let size = tensor.description.validate_dims(&tensor.dims)?;
        if tensor.bytes.len() != size {
            return Err(format!(
                "tensor {} holds {} bytes; its shape requires {size}",
                tensor.description.id,
                tensor.bytes.len()
            ));
        }
        macro_rules! values {
            ($type:ty, $variant:ident) => {{
                let (chunks, rest) = tensor.bytes.as_chunks::<{ size_of::<$type>() }>();
                if !rest.is_empty() {
                    return Err(format!(
                        "tensor {} is not a whole number of elements",
                        tensor.description.id
                    ));
                }
                Self::$variant(
                    chunks
                        .iter()
                        .map(|chunk| <$type>::from_le_bytes(*chunk))
                        .collect(),
                )
            }};
        }
        Ok(match tensor.description.data_type {
            ScalarType::Float16 => values!(u16, Float16),
            ScalarType::Float32 => values!(f32, Float32),
            ScalarType::Float64 => values!(f64, Float64),
            ScalarType::Int8 => values!(i8, Int8),
            ScalarType::Int16 => values!(i16, Int16),
            ScalarType::Int32 => values!(i32, Int32),
            ScalarType::Int64 => values!(i64, Int64),
            ScalarType::Uint8 => Self::Uint8(tensor.bytes.clone()),
            ScalarType::Uint16 => values!(u16, Uint16),
            ScalarType::Uint32 => values!(u32, Uint32),
            ScalarType::Uint64 => values!(u64, Uint64),
        })
    }
}
