import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherCondenser(nn.Module):
    """
    Teacher features → each projected to 64ch → adaptive pooled to match latent shape → concat = 256ch.
    """
    TEACHER_CHS = [64, 128, 256, 512]
    OUT_CH_EACH = 64

    def __init__(self):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(t_ch, self.OUT_CH_EACH, 1, bias=False),
                nn.GroupNorm(8, self.OUT_CH_EACH),
                nn.SiLU(),
            )
            for t_ch in self.TEACHER_CHS
        ])

    def forward(self, teacher_feats, target_hw):
        projected = []
        for feat, proj in zip(teacher_feats, self.projs):
            p = proj(feat)
            if p.shape[-2:] != target_hw:
                p = F.adaptive_avg_pool2d(p, target_hw)
            projected.append(p)
        return torch.cat(projected, dim=1)   # (B, 256, H', W')

class PILDM(nn.Module):
    def __init__(self, vae, encoder, unet, diffusion_scheduler):
        super().__init__()
        self.vae = vae
        self.encoder = encoder
        self.unet = unet
        self.scheduler = diffusion_scheduler
        self.teacher_cond = TeacherCondenser()
        
        # Freeze VAE and Encoder (if KD is pre-trained)
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x_cond, y_target, t, noise=None):
        """
        x_cond:   (B, 6, H, W)   Input condition (Satellite + DEM + Canopy)
        y_target: (B, 1, H, W)   Target carbon map
        t:        (B,)           Timesteps
        """
        # Encode Condition
        with torch.no_grad():
            f_enc, f_proj, _ = self.encoder(x_cond)
            # f_proj has scales for the U-Net cross-attention or concat
            
            # Encode Target into Latent Space
            z_0, _ = self.vae.encode(y_target)
            
        lat_hw = z_0.shape[-2:]
        cond = self.teacher_cond(f_proj, lat_hw)
            
        # Forward diffusion
        z_t, noise_true = self.scheduler.q_sample(z_0, t, noise)
        
        # Denoising
        noise_pred = self.unet(z_t, t, cond)
        
        loss_diff = F.l1_loss(noise_pred, noise_true)
        return loss_diff
        
    def predict(self, x_cond, num_steps=50):
        """DDIM sampling for inference"""
        B = x_cond.shape[0]
        device = x_cond.device
        
        with torch.no_grad():
            f_enc, f_proj, _ = self.encoder(x_cond)
            z_T = torch.randn(B, 4, 32, 32, device=device)
            lat_hw = z_T.shape[-2:]
            cond = self.teacher_cond(f_proj, lat_hw)
            
            z_0_pred = self.scheduler.ddim_sample(self.unet, z_T.shape, cond, steps=num_steps)
            y_pred = self.vae.decode(z_0_pred)
        return y_pred
    
    def decode(self, x_cond, num_steps=50):
        """For physics loss: Requires grad enabled through decoder"""
        B = x_cond.shape[0]
        device = x_cond.device
        
        # Get conditions
        f_enc, f_proj, _ = self.encoder(x_cond)
        z_T = torch.randn(B, 4, 32, 32, device=device)
        
        lat_hw = z_T.shape[-2:]
        cond = self.teacher_cond(f_proj, lat_hw)
        
        # We need to trace gradients from x_cond (specifically CHM channel) through UNet and VAE
        # Standard DDIM sampling isn't fully differentiable if we detach, but we must NOT detach here
        z_0_pred = self.scheduler.ddim_sample(self.unet, z_T.shape, cond, steps=num_steps)
        y_pred = self.vae.decode(z_0_pred)
        return y_pred

def compute_physics_loss(model, x_cond):
    """
    Computes L_mono = mean(ReLU(-∂Ŷ/∂H_canopy))
    x_cond must have requires_grad=True
    """
    # Channel 7 is Canopy Height (CHM)
    y_pred = model.decode(x_cond, num_steps=5) # Reduced steps to 5 for gradient tracking to avoid OOM
    
    # Compute gradients of sum(y_pred) w.r.t x_cond
    dY_dX = torch.autograd.grad(
        outputs=y_pred.sum(),
        inputs=x_cond,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Extract derivative w.r.t Canopy Height (index 7 out of 8)
    dY_dH = dY_dX[:, 7, :, :]
    
    # L_mono penalizes negative derivatives
    l_mono = F.relu(-dY_dH).mean()
    return l_mono
