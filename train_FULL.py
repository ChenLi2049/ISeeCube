import random
import os
import math
import numpy as np
import pandas as pd
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torchinfo import summary

from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.encoder import Encoder

from src.fastai_fix import *
from src.utils import WrapperAdamW
from src.dataset_icemix import (
    IceCubeCache,
    RandomChunkSampler,
    LenMatchBatchSampler,
    DeviceDataLoader,
)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class SinusoidalPosEmb(nn.Module):# Sinusoidal Position Embedding
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
    def __init__(self, dim=384, dim_base=128):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim_base)
        self.aux_emb = nn.Embedding(2, dim_base // 2)
        self.emb2 = SinusoidalPosEmb(dim=dim_base // 2)
        self.proj = nn.Sequential(
            nn.Linear(6 * dim_base, 6 * dim_base),
            nn.LayerNorm(6 * dim_base),
            nn.GELU(),
            nn.Linear(6 * dim_base, dim),
        )

    def forward(self, x, Lmax=None):
        pos = x["pos"] if Lmax is None else x["pos"][:, :Lmax]
        charge = x["charge"] if Lmax is None else x["charge"][:, :Lmax]
        time = x["time"] if Lmax is None else x["time"][:, :Lmax]
        auxiliary = x["auxiliary"] if Lmax is None else x["auxiliary"][:, :Lmax]
        length = torch.log10(x["dim_base_0"].to(dtype=pos.dtype))

        x = torch.cat(#拼接
            [
                #self.cls_token,
                self.emb(4096 * pos).flatten(-2),
                self.emb(1024 * charge),
                self.emb(4096 * time),
                self.aux_emb(auxiliary),
                self.emb2(length).unsqueeze(1).expand(-1, pos.shape[1], -1)
            ],
            -1,
        )
        x = self.proj(x)
        return x


class IceCubeModel(nn.Module):
    def __init__(
        self,
        dim=384,
        dim_base=128
    ):
        super().__init__()
        self.icecube_embedding = IceCubeEmbedding(dim, dim_base)
        encoder_config_1 = EncoderConfig(
            encoder_attention_heads=6,
            encoder_embed_dim=dim,
            encoder_ffn_embed_dim=1536,# Feed Forward Network
            encoder_layers=4,
            # relative position bias, see [1910.10683]
            rel_pos_buckets=32,
            max_rel_pos=dim_base
        )
        self.encoder_1 = Encoder(encoder_config_1)
        self.cls_token = nn.Linear(dim, 1, bias=False)
        encoder_config_2 = EncoderConfig(
            encoder_attention_heads=12,
            encoder_embed_dim=dim,
            encoder_ffn_embed_dim=1536,# Feed Forward Network
            encoder_layers=12
        )
        self.encoder_2 = Encoder(encoder_config_2)
        self.proj_out = nn.Linear(dim, 3)

    def forward(self, x0):
        # mask
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        # cls token
        batch_size, _ = mask.shape
        cls_token = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # x
        x = self.icecube_embedding(x0, Lmax)
        # encoder 1
        x = self.encoder_1(src_tokens=None, token_embeddings=x)
        x = x['encoder_out']
        # concat cls token
        x = torch.cat([cls_token, x], 1)
        # encoder 2
        x = self.encoder_2(src_tokens=None, token_embeddings=x)
        x = x['encoder_out']
        x = self.proj_out(x[:, 0, :])# get cls token, then xyz
        return x


def mse_loss(pred, y):
    pred = pred.float()
    l = torch.norm(pred.float(), dim=-1).unsqueeze(-1)
    pred = pred.float()
    
    sa2 = torch.sin(y["target"][:, 0])
    ca2 = torch.cos(y["target"][:, 0])
    sz2 = torch.sin(y["target"][:, 1])
    cz2 = torch.cos(y["target"][:, 1])
    target = torch.stack([sa2 * sz2, ca2 * sz2, cz2], -1)
    
    loss = nn.MSELoss()(pred, target)
    return loss


def competition_metric(pred, y):
    pred = F.normalize(pred.double(), dim=-1)

    sa2 = torch.sin(y["target"][:, 0])#sin(azimuth)
    ca2 = torch.cos(y["target"][:, 0])
    sz2 = torch.sin(y["target"][:, 1])#sin(zenith)
    cz2 = torch.cos(y["target"][:, 1])

    scalar_prod = (
        pred[:, 0] * sa2 * sz2 + pred[:, 1] * ca2 * sz2 + pred[:, 2] * cz2
    ).clip(-1 + 1e-8, 1 - 1e-8)
    return torch.acos(scalar_prod).abs().mean(-1).float()


def train():
    seed_everything(42)
    
    ds_train = IceCubeCache("data/",mode="train",L=128,selection="total",reduce_size=0.125,)
    ds_train_len = IceCubeCache("data/",mode="train",L=128,reduce_size=0.125,selection="total",mask_only=True,)
    sampler_train = RandomChunkSampler(ds_train_len, chunks=ds_train.chunks)
    len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=32, drop_last=True)
    dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train,batch_sampler=len_sampler_train,num_workers=4,persistent_workers=True,))

    ds_val = IceCubeCache("data/", mode="eval", L=128, selection="total")
    ds_val_len = IceCubeCache("data/",mode="eval",L=128,selection="total",mask_only=True,)
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=32, drop_last=False)
    dl_val = DeviceDataLoader(torch.utils.data.DataLoader(ds_val, batch_sampler=len_sampler_val, num_workers=0))

    data = DataLoaders(dl_train, dl_val)
    model = IceCubeModel().cuda()

    learn = Learner(
        data,
        model,
        path="BEiT3_29M_MSE_rel128_FULL",
        loss_func=mse_loss,
        metrics=competition_metric,opt_func=partial(WrapperAdamW, eps=1e-7)
    ).to_fp16()

    learn.fit_one_cycle(8, lr_max=5e-4, cbs=CSVLogger(fname='history.csv', append=True))
    learn.save("BEiT3_29M_MSE_rel128_FULL_0", with_opt=False)


if __name__ == '__main__':
    train()