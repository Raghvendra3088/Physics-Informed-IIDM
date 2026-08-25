import torch
from src.models.vae import CarbonVAE
from src.models.kd_vgg import LightweightStudentEncoder
from src.models.kd_unet import KDUNet
from src.models.diffusion import DiffusionScheduler
from src.models.pi_ldm import PILDM, compute_physics_loss

device = torch.device('cpu')
vae = CarbonVAE(in_channels=1, latent_channels=4, base_channels=64)
encoder = LightweightStudentEncoder(in_channels=8)
unet = KDUNet(in_channels=4, out_channels=4, context_dim=256)
scheduler = DiffusionScheduler(T=1000, device=device)

model = PILDM(vae, encoder, unet, scheduler)
x_cond = torch.randn(2, 8, 256, 256, requires_grad=True)
loss = compute_physics_loss(model, x_cond)
print("Physics loss computed:", loss.item())
