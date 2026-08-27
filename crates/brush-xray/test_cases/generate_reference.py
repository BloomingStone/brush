"""Generate R2-Gaussian cone-beam X-ray rasterization reference data.

Run from the R2Gaussian pixi env:

    cd /media/F/sj/Code/R2Gaussian-pixi-env
    pixi run python <brush>/crates/brush-xray/test_cases/generate_reference.py

Produces `r2_xray_reference.safetensors` next to this script, containing
the deterministic splat inputs (in brush's parameterisation: means,
log_scales, raw_opac logits, quats) plus R2's cone-beam forward projection
`out_img` [H, W] and the camera parameters. The Rust test reads the inputs
from the same file and compares brush-xray's output.

Camera convention: R2's cone beam looks down +Z (getProjectionMatrix uses
z_sign=+1), which matches brush's pinhole `Camera`. viewmatrix / full_proj
are built exactly like R2's `r2_gaussian/dataset/cameras.py`.
"""

import math

import numpy as np
import torch
from safetensors.torch import save_file

import xray_gaussian_rasterization_voxelization
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

DEVICE = torch.device("cuda:0")
SEED = 1234

# ---------------------------------------------------------------------------
# Camera (brush convention: pos, identity rot, fov, img)
# ---------------------------------------------------------------------------
CAM_POS = torch.tensor([0.0, 0.0, -5.0], dtype=torch.float32, device=DEVICE)
FOV_X = 0.6
FOV_Y = 0.6
W, H = 64, 64


def getWorld2View2(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(fovX, fovY):
    znear = 0.01
    zfar = 100.0
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(fov / 2))


R = np.eye(3)
t = -R @ CAM_POS.cpu().numpy()  # -R^T·cam

# Transposed like R2's Camera (world_view_transform / projection_matrix).
world_view_transform = (
    torch.tensor(getWorld2View2(R, t), dtype=torch.float32).transpose(0, 1).to(DEVICE)
)
projection_matrix = getProjectionMatrix(FOV_X, FOV_Y).transpose(0, 1).to(DEVICE)
full_proj_transform = world_view_transform.unsqueeze(0).bmm(
    projection_matrix.unsqueeze(0)
).squeeze(0)

tanfovx = math.tan(FOV_X / 2)
tanfovy = math.tan(FOV_Y / 2)
focal_x = fov2focal(FOV_X, W)
focal_y = fov2focal(FOV_Y, H)

# ---------------------------------------------------------------------------
# Deterministic scene (brush parameterisation)
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
# Build R2 inputs (scales = exp(log_scales), opacities = MU_WATER·silu(raw)).
# brush-xray's unsigned mode activates `μ = MU_WATER·silu(raw)` (exp6); the R2
# CUDA kernel is activation-neutral, so the pre-activated density is passed
# directly (the R2 python wrapper's `torch.sigmoid` is NOT applied).
# ---------------------------------------------------------------------------
MU_WATER = 0.002


def activated_density(raw_opac: torch.Tensor) -> torch.Tensor:
    # silu(x) = x·sigmoid(x); μ = MU_WATER·silu(raw).
    sig = torch.sigmoid(raw_opac)
    return (MU_WATER * raw_opac * sig).unsqueeze(-1)


means3D = means.to(DEVICE)
scales = log_scales.exp().to(DEVICE)
rotations = torch.nn.functional.normalize(quats, dim=1).to(DEVICE)
opacities = activated_density(raw_opac).to(DEVICE)  # [N,1] pre-activated μ

# means2D — projected pixel coords, detached (only used by R2 backward).
means_view = torch.nn.functional.pad(means3D, (0, 1), value=1.0) @ world_view_transform.T
means_view = means_view[:, :3]
means2D = torch.stack(
    [focal_x * means_view[:, 0] / means_view[:, 2] + W / 2.0,
     focal_y * means_view[:, 1] / means_view[:, 2] + H / 2.0],
    dim=1,
).detach()

# ---------------------------------------------------------------------------
# R2 forward + backward (cone beam, mode=1). One grad-enabled call gives
# both the projection and the autograd gradients.
# ---------------------------------------------------------------------------
raster_settings = GaussianRasterizationSettings(
    image_height=H,
    image_width=W,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    scale_modifier=1.0,
    viewmatrix=world_view_transform,
    projmatrix=full_proj_transform,
    campos=CAM_POS,
    prefiltered=False,
    mode=1,
    debug=False,
)
rasterizer = GaussianRasterizer(raster_settings=raster_settings)

means3D = means.to(DEVICE).requires_grad_(True)
log_scales_g = log_scales.to(DEVICE).requires_grad_(True)
quats_g = quats.to(DEVICE).requires_grad_(True)
raw_opac_g = raw_opac.to(DEVICE).requires_grad_(True)
scales = log_scales_g.exp()
rotations = torch.nn.functional.normalize(quats_g, dim=1)
opacities = activated_density(raw_opac_g)

color, radii = rasterizer(means3D, means2D, opacities, scales, rotations, None)
out_img = color[0].detach().cpu()  # [H, W]

loss = color.sum()
loss.backward()

grad_means = means3D.grad.cpu()
grad_log_scales = log_scales_g.grad.cpu()
grad_raw_opac = raw_opac_g.grad.cpu().reshape(-1)
grad_quats = quats_g.grad.cpu()

print("R2 forward max density:", out_img.max().item())
print("R2 forward sum:", out_img.sum().item())
print("R2 grads: means absmax", grad_means.abs().max().item(),
      "| dlog_scales", grad_log_scales.abs().max().item(),
      "| draw_opac", grad_raw_opac.abs().max().item(),
      "| dquats", grad_quats.abs().max().item())

save_file(
    {
        "means": means,
        "log_scales": log_scales,
        "quats": quats,
        "raw_opac": raw_opac,
        "out_img": out_img,
        "grad_means": grad_means,
        "grad_log_scales": grad_log_scales,
        "grad_raw_opac": grad_raw_opac,
        "grad_quats": grad_quats,
        "cam_pos": CAM_POS.cpu(),
        "fov_x": torch.tensor([FOV_X]),
        "fov_y": torch.tensor([FOV_Y]),
    },
    "./r2_xray_reference.safetensors",
)
print("Saved ./r2_xray_reference.safetensors")
