from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


os.environ.setdefault("ATTN_IMPL", "sdpa")

from local_renderformer.models.config import RenderFormerConfig
from local_renderformer.models.renderformer import RenderFormer


def build_baseline_config(
    latent_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    view_layers: int = 4,
    view_num_heads: int = 4,
    use_dpt_decoder: bool = False,
    num_register_tokens: int = 4,
    patch_size: int = 8,
    texture_patch_size: int = 8,
    vertex_pe_num_freqs: int = 6,
    vn_pe_num_freqs: int = 6,
    use_vn_encoder: bool = True,
    ffn_opt: str = "checkpoint",
    use_object_emb: bool = False,
    object_emb_max_objects: int = 64,
) -> RenderFormerConfig:
    if vertex_pe_num_freqs < 4:
        raise ValueError("vertex_pe_num_freqs must be at least 4 for a safe RoPE setup.")
    if latent_dim % num_heads != 0:
        raise ValueError("latent_dim must be divisible by num_heads.")
    if latent_dim % view_num_heads != 0:
        raise ValueError("latent_dim must be divisible by view_num_heads.")
    if view_layers < 1 or num_layers < 1:
        raise ValueError("Transformer depth must be at least 1.")
    if use_dpt_decoder and view_layers < 4:
        raise ValueError("use_dpt_decoder=True requires view_layers >= 4.")

    encoder_head_dim = latent_dim // num_heads
    max_encoder_rope_dim = encoder_head_dim // 9
    if max_encoder_rope_dim % 2 != 0:
        max_encoder_rope_dim -= 1
    if max_encoder_rope_dim < 4:
        raise ValueError(
            "latent_dim / num_heads is too small for a safe triangle RoPE setup. "
            "Increase latent_dim or reduce num_heads."
        )

    view_head_dim = latent_dim // view_num_heads
    max_view_rope_dim = (view_head_dim // 18) * 2
    if max_view_rope_dim < 4:
        raise ValueError(
            "latent_dim / view_num_heads is too small for a safe view-transformer RoPE setup. "
            "Increase latent_dim or reduce view_num_heads."
        )

    vertex_pe_num_freqs = min(vertex_pe_num_freqs, max_encoder_rope_dim)

    return RenderFormerConfig(
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=latent_dim * 4,
        num_register_tokens=num_register_tokens,
        pe_type="rope",
        vertex_pe_num_freqs=vertex_pe_num_freqs,
        use_vn_encoder=use_vn_encoder,
        vn_pe_num_freqs=vn_pe_num_freqs,
        texture_encode_patch_size=texture_patch_size,
        texture_channels=10,
        view_transformer_latent_dim=latent_dim,
        view_transformer_ffn_hidden_dim=latent_dim * 4,
        view_transformer_n_heads=view_num_heads,
        view_transformer_n_layers=view_layers,
        patch_size=patch_size,
        use_dpt_decoder=use_dpt_decoder,
        turn_to_cam_coord=True,
        use_ldr=False,
        ffn_opt=ffn_opt,
        use_object_emb=use_object_emb,
        object_emb_max_objects=object_emb_max_objects,
    )


class CourseRenderFormerWrapper(nn.Module):
    """
    Minimal wrapper that adapts the repo's RenderFormer model to a plain batch dict.
    """

    def __init__(self, config: RenderFormerConfig, encode_emission_log: bool = True):
        super().__init__()
        self.config = config
        self.encode_emission_log = encode_emission_log
        self.model = RenderFormer(config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_texture_patches(
        self,
        tri_patches: torch.Tensor | None,
        batch_size: int,
        object_count: int,
        triangle_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        target_patch_size = self.config.texture_encode_patch_size
        target_channels = self.config.texture_channels

        if tri_patches is None:
            return torch.zeros(
                batch_size,
                object_count * triangle_count,
                target_channels,
                target_patch_size,
                target_patch_size,
                dtype=dtype,
                device=device,
            )

        tri_patches = tri_patches.to(device=device, dtype=dtype)

        if tri_patches.shape[3] >= 13 and self.encode_emission_log:
            tri_patches = tri_patches.clone()
            emission = tri_patches[..., 10:13, :, :]
            tri_patches[..., 10:13, :, :] = torch.log10(torch.clamp(emission, min=0.0) + 1.0)

        current_channels = tri_patches.shape[3]
        if target_channels == 10 and current_channels >= 13:
            tri_patches = torch.cat(
                [
                    tri_patches[:, :, :, 0:7, :, :],
                    tri_patches[:, :, :, 10:13, :, :],
                ],
                dim=3,
            )
            current_channels = tri_patches.shape[3]

        if current_channels > target_channels:
            tri_patches = tri_patches[:, :, :, :target_channels, :, :]
            current_channels = target_channels
        elif current_channels < target_channels:
            pad_channels = target_channels - current_channels
            tri_patches = torch.cat(
                [
                    tri_patches,
                    torch.zeros(
                        *tri_patches.shape[:3],
                        pad_channels,
                        tri_patches.shape[-2],
                        tri_patches.shape[-1],
                        dtype=tri_patches.dtype,
                        device=tri_patches.device,
                    ),
                ],
                dim=3,
            )
            current_channels = target_channels

        patch_h = tri_patches.shape[-2]
        patch_w = tri_patches.shape[-1]

        if target_patch_size == 1 and (patch_h > 1 or patch_w > 1):
            center_h = patch_h // 2
            center_w = patch_w // 2
            tri_patches = tri_patches[:, :, :, :, center_h:center_h + 1, center_w:center_w + 1]
        elif (patch_h, patch_w) != (target_patch_size, target_patch_size):
            flat_patches = tri_patches.reshape(-1, current_channels, patch_h, patch_w)
            flat_patches = F.interpolate(
                flat_patches,
                size=(target_patch_size, target_patch_size),
                mode="bilinear",
                align_corners=False,
            )
            tri_patches = flat_patches.reshape(
                batch_size,
                object_count,
                triangle_count,
                current_channels,
                target_patch_size,
                target_patch_size,
            )

        return tri_patches.reshape(
            batch_size,
            object_count * triangle_count,
            target_channels,
            target_patch_size,
            target_patch_size,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tri_pos = batch["tri_pos"].to(device=self.device, dtype=torch.float32)
        tri_mask = batch["tri_mask"].to(device=self.device, dtype=torch.bool)
        batch_size, object_count, triangle_count, _ = tri_pos.shape

        tri_vpos_list = tri_pos.reshape(batch_size, object_count * triangle_count, 9)
        valid_mask = tri_mask.reshape(batch_size, object_count * triangle_count)

        tri_normals = batch.get("tri_normals")
        if self.config.use_vn_encoder:
            if tri_normals is None:
                tri_normals = torch.zeros_like(tri_pos)
            tri_normals = tri_normals.to(device=self.device, dtype=tri_pos.dtype)
            tri_normals = tri_normals.reshape(batch_size, object_count * triangle_count, 9)
        else:
            tri_normals = None

        texture_patch_list = self._prepare_texture_patches(
            batch.get("tri_patches"),
            batch_size=batch_size,
            object_count=object_count,
            triangle_count=triangle_count,
            dtype=tri_pos.dtype,
            device=self.device,
        )

        camera_o = batch["camera_o"].to(device=self.device, dtype=tri_pos.dtype)
        ray_map = batch["ray_map"].to(device=self.device, dtype=tri_pos.dtype)
        c2w = batch["c2w"].to(device=self.device, dtype=tri_pos.dtype)

        rays_o = camera_o.unsqueeze(1)
        rays_d = ray_map.unsqueeze(1)

        if self.config.turn_to_cam_coord:
            w2c = torch.linalg.inv(c2w.float()).to(dtype=tri_pos.dtype)

            flat_vertices = tri_vpos_list.reshape(batch_size, object_count * triangle_count * 3, 3)
            ones = torch.ones(
                batch_size,
                flat_vertices.shape[1],
                1,
                dtype=flat_vertices.dtype,
                device=flat_vertices.device,
            )
            vertices_h = torch.cat([flat_vertices, ones], dim=-1)
            vertices_cam = torch.bmm(vertices_h, w2c.transpose(1, 2))[..., :3]
            tri_vpos_view_tf = vertices_cam.reshape(batch_size, 1, object_count * triangle_count, 9)

            w2c_rot = w2c[:, :3, :3]
            height, width = ray_map.shape[1], ray_map.shape[2]
            rays_d_flat = ray_map.reshape(batch_size, -1, 3)
            rays_d_cam = torch.bmm(rays_d_flat, w2c_rot.transpose(1, 2))
            rays_d = rays_d_cam.reshape(batch_size, 1, height, width, 3)
            rays_o = torch.zeros_like(rays_o)
        else:
            tri_vpos_view_tf = tri_vpos_list.unsqueeze(1)

        # per-object segment IDs: triangle at slot [o, t] -> object index o
        if self.config.use_object_emb:
            obj_ids = torch.arange(object_count, device=tri_pos.device).repeat_interleave(triangle_count)
            obj_ids = obj_ids.unsqueeze(0).expand(batch_size, -1)  # [B, N_obj*N_tri]
        else:
            obj_ids = None

        rendered = self.model(
            tri_vpos_list,
            texture_patch_list,
            valid_mask,
            tri_normals,
            rays_o,
            rays_d,
            tri_vpos_view_tf,
            tf32_view_tf=False,
            obj_ids=obj_ids,
        )
        return rendered.squeeze(1).contiguous()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
