from functools import partial

import numpy as np

import torch
import torch.nn as nn

from src.dataset import (
    IceCubeCache,
    RandomChunkSampler,
    LenMatchBatchSampler,
    DeviceDataLoader,
)
from src.models import (
    IceCubeModel_RegA,
    IceCubeModel_RegB,
    IceCubeModel_ClsB
)
from loss import (
    mse_loss,
    competition_metric_reg,
    loss_comb_reg
)
from src.fastai_fix import *
from src.utils import seed_everything, WrapperAdamW
from fastxtend.vision.all import EMACallback


def train():
    seed_everything(42)

    # load data
    ds_train = IceCubeCache("data/",mode="train",L=196,selection="total",reduce_size=0.125,)
    ds_train_len = IceCubeCache("data/",mode="train",L=196,reduce_size=0.125,selection="total",mask_only=True,)
    sampler_train = RandomChunkSampler(ds_train_len, chunks=ds_train.chunks)
    len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=32, drop_last=True)
    dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train,batch_sampler=len_sampler_train,num_workers=32,persistent_workers=True,))

    ds_val = IceCubeCache("data/", mode="eval", L=196, selection="total")
    ds_val_len = IceCubeCache("data/",mode="eval",L=196,selection="total",mask_only=True,)
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=32, drop_last=True)
    dl_val = DeviceDataLoader(torch.utils.data.DataLoader(ds_val, batch_sampler=len_sampler_val, num_workers=0))
    
    data = DataLoaders(dl_train, dl_val)

    # load model
    model = IceCubeModel_RegA()
    model = nn.DataParallel(model)
    model = model.cuda()
    learn = Learner(
        data,
        model,
        path="S_RegA",
        loss_func=mse_loss,
        metrics=[competition_metric_reg,],
        opt_func=partial(WrapperAdamW, eps=1e-7),
        cbs=[
            GradientClip(3.0),
            CSVLogger(append=True),
            EMACallback(),
            SaveModelCallback(
                monitor='competition_metric_reg',comp=np.less,every_epoch=True,with_opt=True
            ),
            GradientAccumulation(n_acc=4096//32)
        ]
    ).to_fp16()

    # load pretrained model
    #learn.load('S_RegA_1_opt',device='cuda',with_opt=True)

    # train and save
    learn.fit_one_cycle(
        1,
        lr_max=1e-4,
        wd=0.05,
        pct_start=0.01,
        div=25,
        div_final=25
    )
    learn.save("S_RegA_1_opt", with_opt=True)
    learn.save("S_RegA_1", with_opt=False)

if __name__ == '__main__':
    train()