use crate::model_info::TensorDescription;

#[derive(Debug)]
pub enum InputTensor {
    Float32(Vec<f32>),
    Uint8(Vec<u8>),
}

#[derive(Debug)]
pub struct OwnedTensor {
    pub description: TensorDescription,
    pub bytes: Vec<u8>,
}

pub trait Engine: Send {
    fn run(&self, input: InputTensor) -> Result<Vec<OwnedTensor>, String>;
}
