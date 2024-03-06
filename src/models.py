import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.encoder import Encoder

class MLP(nn.Sequential):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        nn.Linear(in_features, hidden_features),
        nn.LayerNorm(hidden_features),
        nn.GELU(),
        nn.Linear(hidden_features, out_features)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class IceCubeEmbedding(nn.Module):
    def __init__(self, seq_length=196, hidden_features=1536, out_features=384):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=seq_length)
        self.aux_emb = nn.Embedding(2, seq_length // 2)
        self.emb2 = SinusoidalPosEmb(dim=seq_length // 2)
        self.mlp = MLP(6 * seq_length, hidden_features, out_features)

    def forward(self, x):
        pos, charge, time, auxiliary = x["pos"], x["charge"], x["time"], x["auxiliary"]
        length = torch.log10(x["seq_length_0"].to(dtype=pos.dtype))

        x = torch.cat([
                self.emb(4096 * pos).flatten(-2),
                self.emb(1024 * charge),
                self.emb(4096 * time),
                self.aux_emb(auxiliary),
                self.emb2(length).unsqueeze(1).expand(-1, pos.shape[1], -1)
            ],
            dim=-1
        )
        x = self.mlp(x)
        return x

class IceCubeModel(nn.Module):
    def __init__(self, seq_length=196, mlp_dim=1536, hidden_dim=384, num_register_tokens=3, num_heads=12, num_layers=16, regression_or_classification="regression"):
        super().__init__()
        self.icecube_embedding = IceCubeEmbedding(seq_length, mlp_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02), requires_grad=True)

        self.class_token = nn.Parameter(torch.empty(1, 1, hidden_dim), requires_grad=True)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, hidden_dim), requires_grad=True)

        encoder_config = EncoderConfig(
            encoder_attention_heads=num_heads,
            encoder_embed_dim=hidden_dim,
            encoder_ffn_embed_dim=mlp_dim,
            encoder_layers=num_layers,
            rel_pos_buckets=32,
            max_rel_pos=256
        )
        self.encoder = Encoder(encoder_config)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.regression_or_classification = regression_or_classification
        if regression_or_classification == "regression":
            self.proj = nn.Linear(hidden_dim, 3)# x, y, z -> 3
        elif regression_or_classification == "classification":
            self.proj_azimuth = nn.Linear(hidden_dim, 128)# 128 bins for azimuth
            self.mlp_azimuth = MLP(hidden_dim, mlp_dim, 128)
            self.proj_zenith = nn.Linear(hidden_dim, 64)# 64 bins for zenith
            self.mlp_zenith = MLP(hidden_dim, mlp_dim, 64)
        else:
            raise ValueError("regression_or_classification is a string: 'regression' or 'classification'.")

    def forward(self, x):
        x = self.icecube_embedding(x)# [batch_size, seq_length, hidden_dim]
        batch_size = x.shape[0]

        x += self.pos_embedding# [batch_size, seq_length, hidden_dim]

        batch_class_token = self.class_token.expand(batch_size, -1, -1)
        batch_register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, batch_register_tokens, x], dim=1)# [batch_size, seq_length+4, hidden_dim]

        x = self.encoder(src_tokens=None, token_embeddings=x)
        x = x["encoder_out"]# [batch_size, seq_length+4, hidden_dim]

        x = self.layer_norm(x)# [batch_size, seq_length+4, hidden_dim]

        batch_class_token = x[:, 0, :]# [batch_size, hidden_dim]
        if self.regression_or_classification == "regression":
            x = self.proj(batch_class_token)# [batch_size, 3]
        elif self.regression_or_classification == "classification":
            azimuth = self.proj_azimuth(batch_class_token) + self.mlp_azimuth(batch_class_token)# [batch_size, 128]
            zenith = self.proj_zenith(batch_class_token) + self.mlp_zenith(batch_class_token)# [batch_size, 64]
            x = {"azimuth": azimuth, "zenith": zenith}

        return x

class IceCubeModel_RegA(nn.Module):
    def __init__(self, seq_length=196, hidden_dim=384, num_register_tokens=3, L=256):
        super().__init__()
        self.icecube_embedding = IceCubeEmbedding(
            seq_length=seq_length, hidden_features=2048, out_features=hidden_dim
        )
        self.pos_embedding = nn.Parameter(
            torch.empty(1, L, hidden_dim).normal_(std=0.02),requires_grad=True
        )
        self.class_token = nn.Parameter(
            torch.empty(1, 1, hidden_dim),requires_grad=True
        )
        self.register_tokens = nn.Parameter(
            torch.empty(1, num_register_tokens, hidden_dim),requires_grad=True
        )
        encoder_config = EncoderConfig(
            encoder_attention_heads=12,
            encoder_embed_dim=hidden_dim,
            encoder_ffn_embed_dim=1536,
            encoder_layers=16,
            rel_pos_buckets=32,
            max_rel_pos=256,
            subln=False,
            encoder_normalize_before=False
        )
        self.encoder = Encoder(encoder_config)
        self.proj = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        # mask
        mask = x["mask"]
        batch_size, _ = mask.shape
        # x
        x = self.icecube_embedding(x)
        # add positional embedding
        x += self.pos_embedding
        # concat cls token, register token
        batch_class_token = self.class_token.expand(batch_size, -1, -1)
        batch_register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, batch_register_tokens, x], dim=1)
        # encoder
        x = self.encoder(src_tokens=None, token_embeddings=x)
        x = x['encoder_out']
        # get cls token, then xyz
        x = self.proj(x[:, 0, :])
        return x

class IceCubeModel_RegB(nn.Module):
    def __init__(self, seq_length=196, hidden_dim=384):
        super().__init__()
        self.icecube_embedding = IceCubeEmbedding(
            seq_length=seq_length, hidden_features=8*seq_length, out_features=hidden_dim
        )
        encoder_config_1 = EncoderConfig(
            encoder_attention_heads=12,
            encoder_embed_dim=hidden_dim,
            encoder_ffn_embed_dim=1536,
            encoder_layers=6,
            rel_pos_buckets=32,
            max_rel_pos=seq_length,
            deepnorm=True,
            subln=False,
            encoder_normalize_before=False
        )
        self.encoder_1 = Encoder(encoder_config_1)
        self.cls_token = nn.Linear(hidden_dim, 1, bias=False)
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length+1, hidden_dim).normal_(std=0.02),
            requires_grad=True
        )
        encoder_config_2 = EncoderConfig(
            encoder_attention_heads=12,
            encoder_embed_dim=hidden_dim,
            encoder_ffn_embed_dim=1536,
            encoder_layers=12,
            deepnorm=True,
            subln=False,
            encoder_normalize_before=False
        )
        self.encoder_2 = Encoder(encoder_config_2)
        self.proj_out = nn.Linear(hidden_dim, 3)
        self.proj_out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 8 * seq_length),
            nn.LayerNorm(8 * seq_length),
            nn.GELU(),
            nn.Linear(8 * seq_length, 3),
        )

    def forward(self, x):
        # mask
        mask = x["mask"]
        # cls token
        batch_size, temp_ = mask.shape
        cls_token = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # x
        x = self.icecube_embedding(x)
        # resiual
        residual = x
        # encoder 1
        x = self.encoder_1(src_tokens=None, token_embeddings=x)
        x = x['encoder_out']
        # residual
        x += residual
        # concat cls token
        x = torch.cat([cls_token, x], 1)
        # add positional embedding
        x += self.pos_embedding
        # residual
        residual = x
        # encoder 2
        x = self.encoder_2(src_tokens=None, token_embeddings=x)
        x = x['encoder_out']
        # residual
        x += residual
        # get cls token, then xyz
        x = self.proj_out(x[:, 0, :]) + self.proj_out_mlp(x[:, 0, :])
        return x

class IceCubeModel_ClsB(nn.Module):
    def __init__(self, seq_length=196, hidden_dim=384):
        super().__init__()
        self.icecube_embedding = IceCubeEmbedding(
            seq_length=seq_length, hidden_features=8*seq_length, out_features=hidden_dim
        )
        encoder_config_1 = EncoderConfig(
            encoder_attention_heads=12,
            encoder_embed_dim=hidden_dim,
            encoder_ffn_embed_dim=1536,
            encoder_layers=6,
            rel_pos_buckets=32,
            max_rel_pos=seq_length,
            deepnorm=True,
            subln=False,
            encoder_normalize_before=False
        )
        self.encoder_1 = Encoder(encoder_config_1)
        self.cls_token = nn.Linear(hidden_dim, 1, bias=False)
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length+1, hidden_dim).normal_(std=0.02),
            requires_grad=True
        )
        encoder_config_2 = EncoderConfig(
            encoder_attention_heads=12,
            encoder_embed_dim=hidden_dim,
            encoder_ffn_embed_dim=1536,
            encoder_layers=12,
            deepnorm=True,
            subln=False,
            encoder_normalize_before=False
        )
        self.encoder_2 = Encoder(encoder_config_2)
        self.proj_out_azi = nn.Linear(hidden_dim, 128)
        self.mlp_azi = nn.Sequential(
            nn.Linear(hidden_dim, 8 * seq_length),
            nn.LayerNorm(8 * seq_length),
            nn.GELU(),
            nn.Linear(8 * seq_length, 12 * seq_length),
            nn.LayerNorm(12 * seq_length),
            nn.GELU(),
            nn.Linear(12 * seq_length, 128),
        )
        self.proj_out_zen = nn.Linear(hidden_dim, 64)
        self.mlp_zen = nn.Sequential(
            nn.Linear(hidden_dim, 8 * seq_length),
            nn.LayerNorm(8 * seq_length),
            nn.GELU(),
            nn.Linear(8 * seq_length, 12 * seq_length),
            nn.LayerNorm(12 * seq_length),
            nn.GELU(),
            nn.Linear(12 * seq_length, 64),
        )

    def forward(self, x):
        # mask
        mask = x["mask"]
        # cls token
        batch_size, temp_ = mask.shape
        cls_token = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # x
        x = self.icecube_embedding(x)
        # resiual
        residual = x
        # encoder 1
        x = self.encoder_1(src_tokens=None, token_embeddings=x)
        x = x['encoder_out']
        # residual
        x += residual
        # concat cls token
        x = torch.cat([cls_token, x], 1)
        # add positional embedding
        x += self.pos_embedding
        # residual
        residual = x
        # encoder 2
        x = self.encoder_2(src_tokens=None, token_embeddings=x)
        x = x['encoder_out']
        # residual
        x += residual
        # get cls token, then azi and zen classes
        azimuth = self.proj_out_azi(x[:, 0, :]) + self.mlp_azi(x[:, 0, :])
        zenith = self.proj_out_zen(x[:, 0, :]) + self.mlp_zen(x[:, 0, :])
        return {
            "azimuth": azimuth,
            "zenith": zenith
        }