import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler

from dataset import prepare_sensors


class IceCubeDatatest(Dataset):
    def __init__(self,seq_length=196,batch_id=655,train_path="./data/train",train_meta_path="./data/train_meta"):
        # seq_length for nomalize
        self.seq_length = seq_length

        # 1. load `batch_1.parquet` and concat the same event_id together
        batch_1 = pd.read_parquet(os.path.join(train_path, "batch_"+str(batch_id)+".parquet"))
        batch_1 = batch_1.groupby('event_id').agg({
            'sensor_id': list,
            'time': list,
            'charge': list,
            'auxiliary': list
        })
        batch_1.reset_index(drop=True, inplace=True)# event_id -> index
        self.batch_1 = batch_1

        # 2. geometry
        sensors = prepare_sensors()
        self.geometry = torch.from_numpy(sensors[["x", "y", "z"]].values.astype(np.float32))
        self.qe = sensors["qe"].values

        # 3. load `train_meta.parquet` to get target: azimuth and zenith
        batch_1_meta = pd.read_parquet(os.path.join(train_meta_path, "train_meta_"+str(batch_id)+".parquet"))
        batch_1_meta.reset_index(drop=True, inplace=True)# event_id -> index
        self.batch_1_meta = batch_1_meta

        assert len(self.batch_1)==len(self.batch_1_meta), "batch_1 and batch_1_meta length doesn't match"

    def __len__(self):
        return len(self.batch_1)

    def __getitem__(self, index):
        # 1. for each event in `batch_1.parquet`
        batch_1_event = self.batch_1.loc[index]
        sensor_id = np.array(batch_1_event["sensor_id"])
        time = np.array(batch_1_event["time"])
        time = (time - 1e4) / 3e4
        charge = np.array(batch_1_event["charge"])
        charge = np.log10(charge) / 3.0
        auxiliary = np.array(batch_1_event["auxiliary"])

        # normalize so that the length of each event is 128
        seq_length = len(sensor_id)
        # the output of this method
        seq_length_0 = seq_length
        if seq_length < self.seq_length:
            # assignment
            sensor_id = np.pad(sensor_id, (0, max(0, self.seq_length - seq_length)))
            time = np.pad(time, (0, max(0, self.seq_length - seq_length)))
            charge = np.pad(charge, (0, max(0, self.seq_length - seq_length)))
            auxiliary = np.pad(auxiliary, (0, max(0, self.seq_length - seq_length)))
        else:
            # generate a randomly arranged index array ids
            ids = torch.randperm(seq_length).numpy()
            # negtive index
            auxiliary_n = np.where(~auxiliary)[0]
            # positive index
            auxiliary_p = np.where(auxiliary)[0]
            # choose as many as ids in the negtive index
            ids_n = ids[auxiliary_n][: min(self.seq_length, len(auxiliary_n))]
            # the rest is positive index
            ids_p = ids[auxiliary_p][: min(self.seq_length - len(ids_n), len(auxiliary_p))]
            # concat into a new index
            ids = np.concatenate([ids_n, ids_p])
            # sort the index (from small to big)
            ids.sort()
            # assignment
            sensor_id = sensor_id[ids]
            time = time[ids]
            charge = charge[ids]
            auxiliary = auxiliary[ids]
            seq_length = len(ids)

        # 2. geometry
        sensor_id = torch.from_numpy(sensor_id).long()
        pos = self.geometry[sensor_id]
        pos[seq_length:] = 0
        qe = self.qe[sensor_id]
        qe[seq_length:] = 0

        # 3. target
        azimuth = self.batch_1_meta.loc[index].azimuth
        zenith = self.batch_1_meta.loc[index].zenith
        target = np.array([azimuth, zenith]).astype(np.float32)
        target = torch.from_numpy(target)

        # 4. mask
        mask = torch.zeros(self.seq_length, dtype=torch.bool)
        mask[:seq_length] = True

        return {
            "time": torch.from_numpy(time).float(),
            "charge": torch.from_numpy(charge).float(),
            "auxiliary": torch.from_numpy(auxiliary).long(),
            "pos": pos,
            "mask": mask,
            "qe": qe,
            "seq_length_0": seq_length_0
        }#, {"target": target}