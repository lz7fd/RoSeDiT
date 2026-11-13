# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# REPA: https://github.com/sihyun-yu/REPA (this file extends its models/sit.py)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def build_struct_allow_mask(grid: int, num_ctrl: int) -> torch.Tensor:
    """Attention mask M of the structural pass: bool [S, S], True = may attend,
    token order [controller tokens, image tokens]. Controller tokens read all
    tokens; an image token reads the controller tokens and its 3x3 neighborhood."""
    P, L = num_ctrl, grid * grid
    S = P + L
    m = torch.zeros(S, S, dtype=torch.bool)
    m[:P, :] = True   # controller tokens -> all
    m[:, :P] = True   # all -> controller tokens
    for site in range(L):
        r, c = divmod(site, grid)
        qi = P + site
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid and 0 <= cc < grid:
                    m[qi, P + rr * grid + cc] = True
    return m


class RoseOldMlp(nn.Module):
    """MLP of the dual-weight block (GELU-tanh)."""
    def __init__(self, dim, ratio):
        super().__init__()
        h = int(dim * ratio)
        self.fc1 = nn.Linear(dim, h)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(h, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class RoseOldAttention(nn.Module):
    """Multi-head attention with an optional attention mask (True = may attend)
    and optional RMS normalization of the queries and keys."""
    def __init__(self, dim, heads, qk_norm=False):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.hd) if qk_norm else nn.Identity()

    def forward(self, x, allow_mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allow_mask)
        return self.proj(o.transpose(1, 2).reshape(B, S, D))


class RoSeBlockOld(nn.Module):
    """Dual-weight block (RoSeDiT-Dualweight and its ablations in the paper).
    The dense attention is an unconditioned pass with its own weights; the
    structural pass has a second set of attention weights. A fixed adaLN,
    computed from the condition alone, modulates the structural pass and the MLP."""
    def __init__(self, dim, heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        qk_norm = block_kwargs.get("qk_norm", False)
        self.n_dense = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.attn_dense = RoseOldAttention(dim, heads, qk_norm=qk_norm)
        self.n_struct = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_struct = RoseOldAttention(dim, heads, qk_norm=qk_norm)
        self.n_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = RoseOldMlp(dim, mlp_ratio)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, mask):
        x = x + self.attn_dense(self.n_dense(x))   # unconditioned dense pass
        shs, scs, gs, shm, scm, gm = self.mod(c).chunk(6, dim=1)
        x = x + gs.unsqueeze(1) * self.attn_struct(
            modulate(self.n_struct(x), shs, scs), allow_mask=mask)
        x = x + gm.unsqueeze(1) * self.mlp(modulate(self.n_mlp(x), shm, scm))
        return x


class RoseSP6Block(nn.Module):
    """RoSeViT-style two-pass block with a fixed adaLN (the row "RoSeViT (fixed
    adaLN)" in the paper). One set of attention and MLP weights. Pass 1: dense
    attention with adaLN entry modulation, written to the residual stream
    through a zero-initialized projection. Pass 2 (structural pass): queries
    and keys recomputed from the same weights, values recomputed for the
    controller tokens and reused for the image tokens, masked by M and gated
    by adaLN. The MLP has its own adaLN triple. adaLN reads the condition only."""

    def __init__(self, hidden_size, num_heads, num_ctrl, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.h, self.dh, self.d = num_heads, hidden_size // num_heads, hidden_size
        self.P = num_ctrl
        self.n1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ns = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu, drop=0
            )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        qk_norm = block_kwargs.get("qk_norm", False)
        self.q_norm = nn.RMSNorm(self.dh) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.dh) if qk_norm else nn.Identity()

    def _qkv(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(x).reshape(B, S, 3, self.h, self.dh) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        return self.q_norm(q), self.k_norm(k), v

    def _o(self, o):
        B = o.shape[0]
        return self.proj(o.transpose(1, 2).reshape(B, -1, self.d))

    def _pass2(self, h2, v1, mask):
        B, S, D = h2.shape
        w, b = self.qkv.weight, self.qkv.bias
        q2 = self.q_norm(F.linear(h2, w[:D], b[:D])
                         .reshape(B, S, self.h, self.dh).transpose(1, 2))
        k2 = self.k_norm(F.linear(h2, w[D:2 * D], b[D:2 * D])
                         .reshape(B, S, self.h, self.dh).transpose(1, 2))
        v2c = F.linear(h2[:, :self.P], w[2 * D:], b[2 * D:]) \
            .reshape(B, self.P, self.h, self.dh).transpose(1, 2)
        v = torch.cat([v2c, v1[:, :, self.P:]], 2)
        return self._o(F.scaled_dot_product_attention(q2, k2, v,
                                                      attn_mask=mask))

    def forward(self, x, c, mask):
        m = self.ada(c).reshape(x.shape[0], 6, self.d)
        h1 = self.n1(x) * (1 + m[:, 1].unsqueeze(1)) + m[:, 0].unsqueeze(1)
        q1, k1, v1 = self._qkv(h1)
        x = x + self._o(F.scaled_dot_product_attention(q1, k1, v1))
        hs = self.ns(x)
        x = x + m[:, 2].unsqueeze(1) * self._pass2(hs, v1, mask)
        h2 = self.n2(x) * (1 + m[:, 4].unsqueeze(1)) + m[:, 3].unsqueeze(1)
        return x + m[:, 5].unsqueeze(1) * self.mlp(h2)


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

class BenchRoseBlock(nn.Module):
    """The RoSeDiT block: SiT's adaLN-zero block with the adaLN listening
    mechanism and a parameter-shared structural pass.

    Listening: adaLN computes its six modulation terms (shift, scale and gate
    for attention and for the MLP) from u = c + read(mean of the controller
    tokens) instead of from the condition c alone. They modulate the image
    tokens only.
      readout "diag": read(m) = g * LayerNorm(m), the per-channel listening gate g (RoSeDiT)
      readout "rank": read(m) = B(A(m)), a rank-64 linear readout (ablation)
    Structural pass: the dense attention weights are applied a second time under
    the attention mask M, so the pass adds no attention parameters.
      order "B": listen -> dense pass -> structural pass -> listen -> MLP (RoSeDiT)
      order "C": structural pass -> listen -> dense pass -> MLP (rank-64 ablation)
      nostruct:  listen -> dense pass -> MLP (ablation without the structural pass)
    The listening gate (or the second readout layer) and the gates g_* of the
    structural pass and of the controller tokens are zero-initialized, so at
    initialization the modulation reads the condition alone and the structural
    pass contributes nothing."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, order="B",
                 readout_rank=64, nostruct=False, readout="diag", **block_kwargs):
        super().__init__()
        assert order in ("B", "C")
        assert readout in ("rank", "diag")
        self.readout_mode = readout
        self.order, self.nostruct = order, nostruct
        qk_norm = block_kwargs.get("qk_norm", False)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_s = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = RoseOldAttention(hidden_size, num_heads, qk_norm=qk_norm)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, readout_rank, bias=False),
            nn.Linear(readout_rank, hidden_size, bias=False),  # [1] zero-init
        ) if readout == "rank" else None
        if readout == "diag":   # per-channel listening gate; the norm has no affine (g is the affine)
            self.read_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.g_read = nn.Parameter(torch.zeros(hidden_size))
        self.g_ctrl_attn = nn.Parameter(torch.zeros(hidden_size))
        self.g_ctrl_mlp = nn.Parameter(torch.zeros(hidden_size))
        self.g_struct_canvas = nn.Parameter(torch.zeros(hidden_size))
        self.g_struct_ctrl = nn.Parameter(torch.zeros(hidden_size))

    def _listen(self, x, c, P):
        m = x[:, :P].mean(dim=1)                        # mean of the controller tokens
        u = c
        if self.readout is not None: u = u + self.readout(m)
        if self.readout_mode == "diag": u = u + self.g_read * self.read_norm(m)
        return self.adaLN_modulation(u).chunk(6, dim=-1)

    def _dense(self, x, P, sh, sc, g):
        h = self.norm1(x)
        h = torch.cat([h[:, :P], modulate(h[:, P:], sh, sc)], dim=1)
        o = self.attn(h)
        return torch.cat([x[:, :P] + self.g_ctrl_attn * o[:, :P],
                          x[:, P:] + g.unsqueeze(1) * o[:, P:]], dim=1)

    def _struct(self, x, P, mask):
        o = self.attn(self.norm_s(x), allow_mask=mask)   # same weights as the dense pass
        return torch.cat([x[:, :P] + self.g_struct_ctrl * o[:, :P],
                          x[:, P:] + self.g_struct_canvas * o[:, P:]], dim=1)

    def _mlpstep(self, x, P, sh, sc, g):
        h = self.norm2(x)
        h = torch.cat([h[:, :P], modulate(h[:, P:], sh, sc)], dim=1)
        o = self.mlp(h)
        return torch.cat([x[:, :P] + self.g_ctrl_mlp * o[:, :P],
                          x[:, P:] + g.unsqueeze(1) * o[:, P:]], dim=1)

    def forward(self, x, c, mask, num_ctrl=12):
        P = num_ctrl
        if self.nostruct:
            sa, ca, ga, sm, cm, gm = self._listen(x, c, P)
            x = self._dense(x, P, sa, ca, ga)
            return self._mlpstep(x, P, sm, cm, gm)
        if self.order == "C":
            x = self._struct(x, P, mask)
            sa, ca, ga, sm, cm, gm = self._listen(x, c, P)
            x = self._dense(x, P, sa, ca, ga)
            return self._mlpstep(x, P, sm, cm, gm)
        # order "B"
        sa, ca, ga, _, _, _ = self._listen(x, c, P)
        x = self._dense(x, P, sa, ca, ga)
        x = self._struct(x, P, mask)
        _, _, _, sm, cm, gm = self._listen(x, c, P)     # the MLP listens again after the structural pass
        return self._mlpstep(x, P, sm, cm, gm)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.

    Three families share this class (selected by the configs at the end of the file):
      plain SiT (default);
      bench_rose: RoSeDiT, with twelve controller tokens built from the condition
                  (eight from the class embedding, four from the timestep embedding)
                  and BenchRoseBlock;
      sp6:        the ablation models with RoSeViT-style controller tokens
                  (four pooled from the image tokens, one from the condition).
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.bench_rose = bool(block_kwargs.pop("bench_rose", False))       # RoSeDiT
        self.bench_order = block_kwargs.pop("bench_order", "B")
        self.bench_nostruct = bool(block_kwargs.pop("bench_nostruct", False))
        self.bench_readout = block_kwargs.pop("bench_readout", "diag")
        self.sp6 = bool(block_kwargs.pop("sp6", False))                     # RoSeViT-style controller tokens (ablations)
        self.sp6_onepass = bool(block_kwargs.pop("sp6_onepass", False))     # plain SiT blocks with the controller tokens in context
        self.rose_old = bool(block_kwargs.pop("rose_old", False))           # dual-weight block
        self.sp6_dense_p2 = bool(block_kwargs.pop("sp6_dense_p2", False))   # structural pass without the mask
        self.rose_noctrl = bool(block_kwargs.pop("rose_noctrl", False))     # no controller tokens
        if self.bench_rose:
            grid = int(num_patches ** 0.5)
            self.num_ctrl = 12   # 8 class-conditioned + 4 timestep-conditioned controller tokens
            self.class_tokens = nn.Parameter(torch.zeros(1, 8, hidden_size))
            self.time_tokens = nn.Parameter(torch.zeros(1, 4, hidden_size))
            self.register_buffer(
                "allow_mask", build_struct_allow_mask(grid, self.num_ctrl), persistent=False
                )
            self.final_readout = nn.Sequential(
                nn.Linear(hidden_size, 64, bias=False),
                nn.Linear(64, hidden_size, bias=False),  # [1] zero-init
            ) if self.bench_readout == "rank" else None
            if self.bench_readout == "diag":
                self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.g_final = nn.Parameter(torch.zeros(hidden_size))
            self.blocks = nn.ModuleList([
                BenchRoseBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                               order=self.bench_order, nostruct=self.bench_nostruct,
                               readout=self.bench_readout,
                               **block_kwargs) for _ in range(depth)
            ])
        elif self.sp6:
            grid = int(num_patches ** 0.5)
            self.num_ctrl = 0 if self.rose_noctrl else 5   # 4 pooled from the image tokens + 1 from the condition
            self.ctrl_pos = nn.Parameter(torch.zeros(1, max(self.num_ctrl, 1), hidden_size))   # (unused row when num_ctrl == 0)
            self.register_buffer(
                "allow_mask", build_struct_allow_mask(grid, self.num_ctrl), persistent=False
                )
            if self.sp6_onepass:
                self.blocks = nn.ModuleList([SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)])
            elif self.rose_old:
                if not self.rose_noctrl:
                    self.instr_mlp = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                        nn.Linear(hidden_size, hidden_size))
                self.blocks = nn.ModuleList([
                    RoSeBlockOld(hidden_size, num_heads,
                                 mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
                ])
            else:
                self.blocks = nn.ModuleList([
                    RoseSP6Block(hidden_size, num_heads, self.num_ctrl,
                                 mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
                ])
        else:
            self.num_ctrl = 0
            self.blocks = nn.ModuleList([
                SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
            ])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            if isinstance(block, RoSeBlockOld):
                nn.init.constant_(block.mod[-1].weight, 0)
                nn.init.constant_(block.mod[-1].bias, 0)
            elif isinstance(block, RoseSP6Block):
                nn.init.constant_(block.ada[-1].weight, 0)
                nn.init.constant_(block.ada[-1].bias, 0)
                nn.init.constant_(block.proj.weight, 0)  # pass 1 writes through a zero-init projection
                nn.init.constant_(block.proj.bias, 0)
            else:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        if self.bench_rose:
            nn.init.normal_(self.class_tokens, std=0.02)
            nn.init.normal_(self.time_tokens, std=0.02)
            for block in self.blocks:
                if block.readout is not None: nn.init.constant_(block.readout[1].weight, 0)
            if self.final_readout is not None: nn.init.constant_(self.final_readout[1].weight, 0)
        if self.sp6:
            nn.init.normal_(self.ctrl_pos, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y, return_logvar=False):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if self.bench_rose:
            # analytic first-order preconditioning: z_t = (1-t)x + t*eps has
            # std sqrt((1-t)^2 + t^2) for unit-variance latents; normalize it out.
            cin = torch.rsqrt((1.0 - t) ** 2 + t ** 2).to(x.dtype)
            x = x * cin.view(-1, 1, 1, 1)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        if self.bench_rose:
            ctrl = torch.cat([
                self.class_tokens.expand(N, -1, -1) + y.unsqueeze(1),
                self.time_tokens.expand(N, -1, -1) + t_embed.unsqueeze(1),
            ], dim=1)                                    # (N, 12, D)
            x = torch.cat([ctrl, x], dim=1)
            for i, block in enumerate(self.blocks):
                x = block(x, c, self.allow_mask, num_ctrl=self.num_ctrl)
                if (i + 1) == self.encoder_depth:
                    xp = x[:, self.num_ctrl:]            # image tokens only feed REPA
                    zs = [projector(xp.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
            m_final = x[:, :self.num_ctrl].mean(dim=1)   # the final layer listens as well
            u_final = c
            if self.final_readout is not None: u_final = u_final + self.final_readout(m_final)
            if self.bench_readout == "diag": u_final = u_final + self.g_final * self.final_norm(m_final)
            x = x[:, self.num_ctrl:]
            x = self.final_layer(x, u_final)
            x = self.unpatchify(x)
            return x, zs

        if self.sp6:
            # RoSeViT-style controller tokens: [4 pooled + 1 from the condition | image tokens]
            grid = int(T ** 0.5)
            fmap = x.transpose(1, 2).reshape(N, D, grid, grid)
            pooled = F.adaptive_avg_pool2d(fmap, (2, 2)).flatten(2).transpose(1, 2)  # [N,4,D]
            instr = self.instr_mlp(c) if (self.rose_old and not self.rose_noctrl) else c
            ctrl = torch.cat([pooled, instr.unsqueeze(1)], dim=1)
            if self.rose_noctrl: ctrl = ctrl[:, :0]             # no controller tokens at all
            ctrl = ctrl + self.ctrl_pos[:, :ctrl.shape[1]]      # [N,P,D]
            x = torch.cat([ctrl, x], dim=1)
            p2_mask = None if self.sp6_dense_p2 else self.allow_mask
            for i, block in enumerate(self.blocks):
                x = block(x, c) if self.sp6_onepass else block(x, c, p2_mask)
                if (i + 1) == self.encoder_depth:
                    xp = x[:, self.num_ctrl:]    # image tokens only feed REPA
                    zs = [projector(xp.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
            x = x[:, self.num_ctrl:]
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, c)                      # (N, T, D)
                if (i + 1) == self.encoder_depth:
                    zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x, zs


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################
# The names are the ones the models were trained under; README.md maps each of
# them to its row in the paper.

# --- SiT baselines -------------------------------------------------------------

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_B_d15_2(**kwargs):       # SiT-B/2 Depth (15): 15 blocks
    return SiT(depth=15, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_w864_2(**kwargs):      # SiT-B/2 Width (864): hidden size 864
    return SiT(depth=12, hidden_size=864, decoder_hidden_size=864, patch_size=2, num_heads=12, **kwargs)

# --- RoSeDiT ---------------------------------------------------------------------

def SiT_BR_B1W_DG_S_2(**kwargs):   # RoSeDiT-S/2
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=2, num_heads=6,
               bench_rose=True, bench_order="B", bench_readout="diag", **kwargs)

def SiT_BR_B1W_DG_B_2(**kwargs):   # RoSeDiT-B/2
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12,
               bench_rose=True, bench_order="B", bench_readout="diag", **kwargs)

def SiT_BR_B1W_DG_L_2(**kwargs):   # RoSeDiT-L/2
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16,
               bench_rose=True, bench_order="B", bench_readout="diag", **kwargs)

def SiT_BR_B1W_DG_XL_2(**kwargs):  # RoSeDiT-XL/2
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16,
               bench_rose=True, bench_order="B", bench_readout="diag", **kwargs)

# --- Ablations at B/2 ------------------------------------------------------------

def SiT_SP6_B_2(**kwargs):         # RoSeViT (fixed adaLN)
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, sp6=True, **kwargs)

def SiT_BR_C1W_B_2(**kwargs):      # RoSeViT (fixed adaLN) + Listening (rank-64)
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12,
               bench_rose=True, bench_order="C", bench_readout="rank", **kwargs)

def SiT_CTRL_B_2(**kwargs):        # SiT-B/2 + In-context controller
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, sp6=True, sp6_onepass=True, **kwargs)

def SiT_BR_NS_B_2(**kwargs):       # SiT-B/2 + In-context controller + Listening (rank-64)
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12,
               bench_rose=True, bench_order="C", bench_readout="rank", bench_nostruct=True, **kwargs)

def SiT_ROSE_B_2(**kwargs):        # RoSeDiT-Dualweight-B/2
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, sp6=True, rose_old=True, **kwargs)

def SiT_ROSE_DP2_B_2(**kwargs):    # RoSeDiT-Dualweight-B/2 - Mask
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, sp6=True, rose_old=True, sp6_dense_p2=True, **kwargs)

def SiT_ROSE_NC_B_2(**kwargs):     # RoSeDiT-Dualweight-B/2 - Controller
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, sp6=True, rose_old=True, rose_noctrl=True, **kwargs)

def SiT_ROSE_DP2_NC_B_2(**kwargs): # RoSeDiT-Dualweight-B/2 - Both
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, sp6=True, rose_old=True, sp6_dense_p2=True, rose_noctrl=True, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-L/2': SiT_L_2,  'SiT-B/2': SiT_B_2,  'SiT-S/2': SiT_S_2,
    'SiT-B-d15/2': SiT_B_d15_2,  'SiT-B-w864/2': SiT_B_w864_2,
    'SiT-BR-B1W-DG-S/2': SiT_BR_B1W_DG_S_2,  'SiT-BR-B1W-DG-B/2': SiT_BR_B1W_DG_B_2,
    'SiT-BR-B1W-DG-L/2': SiT_BR_B1W_DG_L_2,  'SiT-BR-B1W-DG-XL/2': SiT_BR_B1W_DG_XL_2,
    'SiT-SP6-B/2': SiT_SP6_B_2,  'SiT-BR-C1W-B/2': SiT_BR_C1W_B_2,
    'SiT-CTRL-B/2': SiT_CTRL_B_2,  'SiT-BR-NS-B/2': SiT_BR_NS_B_2,
    'SiT-ROSE-B/2': SiT_ROSE_B_2,  'SiT-ROSE-DP2-B/2': SiT_ROSE_DP2_B_2,
    'SiT-ROSE-NC-B/2': SiT_ROSE_NC_B_2,  'SiT-ROSE-DP2-NC-B/2': SiT_ROSE_DP2_NC_B_2,
}
