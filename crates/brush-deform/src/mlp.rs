//! Skip-connected MLP used by the deformation network (mirrors the Python
//! `MLP` in `hashgrid_deform.py`): each block maps the (widening) input to a
//! hidden width `W` through a small inner network, then re-concatenates the
//! original input, and a final linear reads the `W + input_ch` result.

use burn::module::Module;
use burn::nn::activation::Relu;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;

/// A single `D`-layer inner block mapping `input_ch → W` with ReLU activations.
#[derive(Module, Debug)]
pub struct InnerBlock {
    layers: Vec<Linear>,
    activation: Relu,
}

impl InnerBlock {
    pub fn new(input_ch: usize, width: usize, depth: usize, device: &burn::tensor::Device) -> Self {
        let mut layers = Vec::with_capacity(depth);
        let mut ch_in = input_ch;
        for _ in 0..depth {
            layers.push(LinearConfig::new(ch_in, width).init(device));
            ch_in = width;
        }
        Self {
            layers,
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mut h = x;
        for layer in &self.layers {
            h = self.activation.forward(layer.forward(h));
        }
        h
    }
}

/// Skip-connected MLP: `combine_layers` inner blocks, each re-concatenating
/// the original input, then a final linear `W + input_ch → output_ch`.
#[derive(Module, Debug)]
pub struct SkipMlp {
    blocks: Vec<InnerBlock>,
    out: Linear,
    input_ch: usize,
    width: usize,
}

impl SkipMlp {
    pub fn new(
        input_ch: usize,
        width: usize,
        blocks: usize,
        block_depth: usize,
        output_ch: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        let mut inner = Vec::with_capacity(blocks);
        let mut ch_in = input_ch;
        for _ in 0..blocks {
            inner.push(InnerBlock::new(ch_in, width, block_depth, device));
            ch_in = input_ch + width;
        }
        let out = LinearConfig::new(input_ch + width, output_ch).init(device);
        Self {
            blocks: inner,
            out,
            input_ch,
            width,
        }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(h);
            h = Tensor::cat(vec![x.clone(), h], 1);
        }
        self.out.forward(h)
    }

    /// Hidden width (used by the model heads).
    pub fn width(&self) -> usize {
        self.width
    }
}

/// Light feed-forward MLP for the HexPlane decoder: `n_layers` Linear layers
/// (`input → hidden`, then `hidden → hidden`), ReLU between them, no final
/// activation. Output width is `hidden`.
#[derive(Module, Debug)]
pub struct Mlp {
    layers: Vec<Linear>,
    activation: Relu,
}

impl Mlp {
    pub fn new(
        input_ch: usize,
        hidden: usize,
        n_layers: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        let mut layers = Vec::with_capacity(n_layers.max(1));
        let mut ch_in = input_ch;
        for _ in 0..n_layers.max(1) {
            layers.push(LinearConfig::new(ch_in, hidden).init(device));
            ch_in = hidden;
        }
        Self {
            layers,
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mut h = x;
        let mut it = self.layers.iter().peekable();
        while let Some(layer) = it.next() {
            h = layer.forward(h);
            if it.peek().is_some() {
                h = self.activation.forward(h);
            }
        }
        h
    }
}
