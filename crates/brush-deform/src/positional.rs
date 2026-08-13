//! Sin/cos positional encoding for scalar (phase/time) conditioning inputs.

use burn::tensor::Tensor;

/// Positional encoding: `[x, sin(2^0 π x), cos(2^0 π x), ..., sin(2^(f-1) π x),
/// cos(2^(f-1) π x)]` per input channel.
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    input_channels: u32,
    n_frequencies: u32,
}

impl PositionalEncoding {
    pub fn new(input_channels: u32, n_frequencies: u32) -> Self {
        Self {
            input_channels,
            n_frequencies,
        }
    }

    /// Output dimension per sample.
    pub fn output_channels(&self) -> u32 {
        self.input_channels * (1 + 2 * self.n_frequencies)
    }

    /// Encode `x` (`[N, input_channels]`) → `[N, output_channels]`.
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mut out = vec![x.clone()];
        for i in 0..self.n_frequencies {
            let freq = (1u32 << i) as f32 * std::f32::consts::PI;
            let scaled = x.clone() * freq;
            out.push(scaled.clone().sin());
            out.push(scaled.cos());
        }
        Tensor::cat(out, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_dimension_is_correct() {
        let enc = PositionalEncoding::new(1, 6);
        assert_eq!(enc.output_channels(), 1 + 2 * 6);
    }
}
