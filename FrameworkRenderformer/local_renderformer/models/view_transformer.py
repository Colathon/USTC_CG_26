import torch
import torch.nn as nn

from .config import RenderFormerConfig
from ..encodings.nerf_encoding import NeRFEncoding
from ..layers.attention import TransformerDecoder
from ..layers.dpt import DPTHead
from ..compat_einops import rearrange


class ViewTransformer(nn.Module):
    def __init__(self, config: RenderFormerConfig):
        super().__init__()
        self.config = config
        if config.pe_type == 'nerf':
            self.pos_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=config.vertex_pe_num_freqs,
                include_input=True
            )
            self.pe_token_proj = nn.Linear(
                self.pos_pe.get_out_dim(),
                config.view_transformer_latent_dim
            )
            if config.norm_type == 'layer_norm':
                self.token_pos_pe_norm = nn.LayerNorm(config.view_transformer_latent_dim)
            elif config.norm_type == 'rms_norm':
                self.token_pos_pe_norm = nn.RMSNorm(config.view_transformer_latent_dim)
            else:
                raise ValueError(f"Unsupported normalization type: {config.norm_type}")
            self.rope_dim = None
        elif config.pe_type == 'rope':
            self.rope_dim = min(config.vertex_pe_num_freqs, config.view_transformer_latent_dim // config.view_transformer_n_heads // 18 * 2)
        else:
            raise ValueError(f"Unsupported positional encoding type: {config.pe_type}")

        self.ray_map_patch_token = nn.Parameter(torch.randn(1, 1, config.view_transformer_latent_dim))
        if config.vdir_pe_type == 'nerf':
            self.vdir_pe = NeRFEncoding(
                in_dim=3,
                num_frequencies=config.vdir_num_freqs,
                include_input=True
            )
            self.ray_map_encoder = nn.Linear(
                self.vdir_pe.get_out_dim() * config.patch_size * config.patch_size,
                config.view_transformer_latent_dim
            )
            if config.norm_type == 'layer_norm':
                self.ray_map_encoder_norm = nn.LayerNorm(config.view_transformer_latent_dim)
            elif config.norm_type == 'rms_norm':
                self.ray_map_encoder_norm = nn.RMSNorm(config.view_transformer_latent_dim)
            else:
                raise ValueError(f"Unsupported normalization type: {config.norm_type}")
        else:
            raise ValueError(f"Unsupported view direction positional encoding type: {config.vdir_pe_type}")

        self.transformer = TransformerDecoder(
            num_layers=self.config.view_transformer_n_layers,
            num_heads=self.config.view_transformer_n_heads,
            hidden_dim=self.config.view_transformer_latent_dim,
            ctx_dim=self.config.latent_dim,
            ffn_hidden_dim=self.config.view_transformer_ffn_hidden_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            norm_first=self.config.norm_first,
            rope_dim=self.rope_dim,
            rope_type=self.config.rope_type,
            rope_double_max_freq=self.config.rope_double_max_freq,
            qk_norm=self.config.qk_norm,
            bias=self.config.bias,
            include_self_attn=self.config.view_transformer_include_self_attn,
            use_swin_attn=self.config.view_transformer_use_swin_attn,
            ffn_opt=self.config.ffn_opt
        )
        if not config.use_dpt_decoder:
            self.out_proj = nn.Linear(self.config.view_transformer_latent_dim, self.config.patch_size * self.config.patch_size * (4 if config.include_alpha else 3))
        else:
            self.out_dpt = DPTHead(
                in_channels=self.config.view_transformer_latent_dim,
                features=self.config.dpt_features,
                out_channels=self.config.dpt_out_channels,
                out_dim=4 if config.include_alpha else 3
            )
            self.out_layers = list(range(self.config.view_transformer_n_layers - 4, self.config.view_transformer_n_layers)) if self.config.dpt_out_layers is None else self.config.dpt_out_layers
        self.out_proj_act = nn.ELU(alpha=1e-3)

    def forward(self, camera_o, ray_map, tri_tokens, tri_pos, valid_mask, tf32_mode=False):
        """
        Cross attention between ray map and triangle tokens.

        Args:
            camera_o (torch.Tensor): (B, 3)
            ray_map (torch.Tensor): (B, H, W, 3)
            tri_tokens (torch.Tensor): (B, N_TRIS, D)
            tri_pos (torch.Tensor): (B, N_TRIS, 9)
            valid_mask (torch.Tensor): (B, N_TRIS)
            tf32_mode (bool): whether to use tf32 mode
        Returns:
            decoded_img: (B, 3, H, W)
        """

        # ====== HW8_TODO: Implement Ray Bundle Embedding ======
        p = self.config.patch_size
        patch_h = ray_map.size(1) // p
        patch_w = ray_map.size(2) // p

        # 1. Apply NeRF PE to ray direction map
        encoded_ray = self.vdir_pe(ray_map)  # [B, H, W, C_pe]

        # 2. Reshape into non-overlapping patches
        ray_patches = rearrange(encoded_ray, 'b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)', p1=p, p2=p)

        # 3. Linear projection + norm + learnable patch token
        ray_tokens = self.ray_map_encoder_norm(self.ray_map_encoder(ray_patches)) + self.ray_map_patch_token

        # 4. Per-patch position: repeat camera origin to 9D and expand over patches
        B = camera_o.size(0)
        N_patches = patch_h * patch_w
        camera_pos_9d = camera_o.repeat(1, 3)  # [B, 9]
        ray_token_pos = camera_pos_9d.unsqueeze(1).expand(-1, N_patches, -1).contiguous()  # [B, N_patches, 9]

        # 5. For nerf PE, add absolute position encoding to both ray and triangle tokens
        if self.config.pe_type == 'nerf':
            cam_pe = self.token_pos_pe_norm(self.pe_token_proj(self.pos_pe(camera_pos_9d)))  # [B, D]
            ray_tokens = ray_tokens + cam_pe.unsqueeze(1)
            tri_pe = self.token_pos_pe_norm(self.pe_token_proj(self.pos_pe(tri_pos)))  # [B, N_tri, D]
            tri_tokens = tri_tokens + tri_pe
        # =====================================================

        # do per-ray attention
        if self.config.use_dpt_decoder:
            with torch.autocast(device_type="cuda", dtype=torch.float32 if tf32_mode else torch.bfloat16):
                out_features = self.transformer(ray_tokens, tri_tokens, src_key_padding_mask=valid_mask, triangle_pos=tri_pos, ray_pos=ray_token_pos, out_layers=self.out_layers, tf32_mode=tf32_mode, patch_h=patch_h, patch_w=patch_w)
            decoded_img = self.out_dpt(out_features, patch_h, patch_w, patch_size=self.config.patch_size)
            return self.out_proj_act(decoded_img)
        else:
            seq = self.transformer(ray_tokens, tri_tokens, src_key_padding_mask=valid_mask, triangle_pos=tri_pos, ray_pos=ray_token_pos, tf32_mode=tf32_mode, patch_h=patch_h, patch_w=patch_w)  # [B, N_PATCHES, D]
            decoded_patches = self.out_proj_act(self.out_proj(seq))  # [B, N_PATCHES, P*P*3]
            decoded_img = rearrange(decoded_patches, 'b (h1 w1) (c p1 p2) -> b c (h1 p1) (w1 p2)', p1=self.config.patch_size, p2=self.config.patch_size, h1=patch_h, w1=patch_w)
            return decoded_img
