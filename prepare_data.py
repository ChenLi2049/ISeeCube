# edited from https://github.com/DrHB/icecube-2nd-place/blob/main/prepare_data.py

import os
import polars as pl
import pickle
from tqdm import tqdm

def run():
    Nevents = {}
    for fname in tqdm(sorted(os.listdir(os.path.join("data", "train")))):
        path = os.path.join("data", "train", fname)
        df = pl.read_parquet(path)
        df = df.groupby("event_id").agg([pl.count()])

        Nevents[fname] = {}
        Nevents[fname]["total"] = len(df)
        Nevents[fname]["short"] = len(df.filter(pl.col("count") < 64))
        Nevents[fname]["medium"] = len(df.filter((pl.col("count") >= 64) & (pl.col("count") < 192)))
        Nevents[fname]["long"] = len(df.filter(pl.col("count") >= 192))

    with open(os.path.join("data", "Nevents.pickle"), "wb") as f:
        pickle.dump(Nevents, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    run()