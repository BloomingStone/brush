"""Generate R2-Gaussian voxelization reference data.

Run from the R2Gaussian pixi env:

    cd /media/F/sj/Code/R2Gaussian-pixi-env
    pixi run python <brush>/crates/brush-voxel/test_cases/generate_reference.py

Produces `r2_voxel_reference.safetensors` next to this script, containing
the deterministic splat inputs (in brush's parameterisation: means,
log_scales, raw_opac logits, quats) plus R2's cone-beam voxelization
`fields` [nVoxel_x, nVoxel_y, nVoxel_z] and the voxel grid settings. The
Rust test reads the inputs from the same file and compares brush-voxel's
output volume.
"""

import math

import numpy as np
import torch
from safetensors.torch import save_file

from xray_gaussian_rasterization_voxelization import (
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

DEVICE = torch.device("cuda:0")
SEED = 1234

# ---------------------------------------------------------------------------
# Voxel grid (volume spans center +/- sVoxel/2)
# ---------------------------------------------------------------------------
NV_X, NV_Y, NV_Z = 16, 16, 16
SV_X, SV_Y, SV_Z = 2.0, 2.0, 2.0
CENTER = (0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------
# Deterministic scene (brush parameterisation, same RNG as the rasterizer)
# ---------------------------------------------------------------------------
g = torch.Generator(device="cpu").manual_seed(SEED)
N = 32
means = torch.rand(N, 3, generator=g, dtype=torch.float32) * 2.0 - 1.0
# Anisotropic random scales — with equal scales Σ = s²·I is rotation-
# independent, so it would NOT exercise the Rᵀ·S²·R covariance convention.
gs = torch.Generator(device="cpu").manual_seed(9876)
log_scales = (torch.rand(N, 3, generator=gs, dtype=torch.float32) - 0.5) * 2.0
# Non-identity random rotations (w,x,y,z).
gq = torch.Generator(device="cpu").manual_seed(20240811)
quats = torch.rand(N, 4, generator=gq, dtype=torch.float32) * 2.0 - 1.0
raw_opac = torch.full((N,), 2.0, dtype=torch.float32)

# ---------------------------------------------------------------------------
# R2 inputs: scales = exp(log_scales), opacities = PRE-ACTIVATED
# μ = MU_WATER·silu(raw_opac) [N,1] (brush's in-kernel activation, mirrored
# on the host here), rotations normalized (w,x,y,z). The CUDA kernels are
# activation-neutral — the caller activates.
# ---------------------------------------------------------------------------
means3D = means.to(DEVICE)
scales = log_scales.exp().to(DEVICE)
rotations = torch.nn.functional.normalize(quats, dim=1).to(DEVICE)
opacities = torch.sigmoid(raw_opac).unsqueeze(-1).to(DEVICE)  # [N,1] pre-activated μ

voxel_settings = GaussianVoxelizationSettings(
    scale_modifier=1.0,
    nVoxel_x=NV_X,
    nVoxel_y=NV_Y,
    nVoxel_z=NV_Z,
    sVoxel_x=SV_X,
    sVoxel_y=SV_Y,
    sVoxel_z=SV_Z,
    center_x=CENTER[0],
    center_y=CENTER[1],
    center_z=CENTER[2],
    prefiltered=False,
    debug=False,
)
voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

means3D = means.to(DEVICE).requires_grad_(True)
log_scales_g = log_scales.to(DEVICE).requires_grad_(True)
quats_g = quats.to(DEVICE).requires_grad_(True)
raw_opac_g = raw_opac.to(DEVICE).requires_grad_(True)
scales = log_scales_g.exp()
rotations = torch.nn.functional.normalize(quats_g, dim=1)
# R2's CUDA kernel is activation-neutral. brush's voxelizer activates
# in-kernel: μ = MU_WATER·silu(raw). Reproduce that here by chaining the
# activation through autograd so the stored grads are dL/draw (raw logits),
# which the Rust golden test feeds directly.
MU_WATER = 0.002
sig = torch.sigmoid(raw_opac_g)
opacities = (MU_WATER * raw_opac_g * sig).unsqueeze(-1)  # μ = MU_WATER·silu(raw)

fields, radii = voxelizer(means3D, opacities, scales, rotations, None)
out_vol = fields.detach().cpu()  # [NV_X, NV_Y, NV_Z]

loss = fields.sum()
loss.backward()

grad_means = means3D.grad.cpu()
grad_log_scales = log_scales_g.grad.cpu()
grad_raw_opac = raw_opac_g.grad.cpu().reshape(-1)
grad_quats = quats_g.grad.cpu()

print("R2 voxel max density:", out_vol.max().item())
print("R2 voxel sum:", out_vol.sum().item())
print("R2 grads: means absmax", grad_means.abs().max().item(),
      "| dlog_scales", grad_log_scales.abs().max().item(),
      "| draw_opac(dL/draw)", grad_raw_opac.abs().max().item(),
      "| dquats", grad_quats.abs().max().item())

save_file(
    {
        "means": means,
        "log_scales": log_scales,
        "quats": quats,
        "raw_opac": raw_opac,
        "out_vol": out_vol,
        "grad_means": grad_means,
        "grad_log_scales": grad_log_scales,
        "grad_raw_opac": grad_raw_opac,
        "grad_quats": grad_quats,
        "nVoxel": torch.tensor([NV_X, NV_Y, NV_Z], dtype=torch.int32),
        "sVoxel": torch.tensor([SV_X, SV_Y, SV_Z], dtype=torch.float32),
        "center": torch.tensor(list(CENTER), dtype=torch.float32),
    },
    "./r2_voxel_reference.safetensors",
)
print("Saved ./r2_voxel_reference.safetensors")
