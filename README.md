# ISeeCube

## Introduction

The goal of this project is to reconstruct the direction of neutrino events detected by [IceCube Neutrino Observatory](https://icecube.wisc.edu/). It is based on the Kaggle competition [IceCube - Neutrinos in Deep Ice](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice). And the arXiv link is [[2308.13285] Refine Neutrino Events Reconstruction with BEiT-3](https://arxiv.org/abs/2308.13285).

## Install

1. Clone the repository:

    ```bash
    git clone https://github.com/ChenLi2049/ISeeCube.git
    ```

2. Navigate to the repository folder:

    ```bash
    cd ISeeCube
    ```

3. Create conda environment `iseecube`:

    ```bash
    conda create -n iseecube python=3.8
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Please download the [dataset](https://www.kaggle.com/c/icecube-neutrinos-in-deep-ice/data) and put it in the folder of this repository, the `data` folder should look likes this:

```bash
data/
├── sample_submission.parquet
├── sensor_geometry.csv
├── test
│   └── batch_661.parquet
├── test_meta.parquet
└── train
    ├── batch_1.parquet
    └── batch_2.parquet
```

Now you can run `visualize_dataset.ipynb`.

## Train

1. Download [splitted train_meta](https://www.kaggle.com/datasets/solverworld/train-meta-parquet) and put it in the `data` folder.

2. Download [icecube_transparency](https://www.kaggle.com/datasets/anjum48/icecubetransparency) and put it in the `data` folder.

3. Run this command to create `Nevents.pickle` file in the `data` folder:

    ```bash
    python prepare_data.py
    ```

Now the `data` folder should looks like this:

```bash
data/
├── Nevents.pickle
├── ice_transparency.txt
├── sample_submission.parquet
├── sensor_geometry.csv
├── test
│   └── batch_661.parquet
├── test_meta.parquet
├── train
│   ├── batch_1.parquet
│   └── batch_2.parquet
└── train_meta
    ├── train_meta_1.parquet
    └── train_meta_2.parquet
```

To train S_RegA model on about 654 batches divided into 8 epochs, First create a folder named `S_RegA` and an empty file named `history.csv` in the created folder, then run this command:

```bash
python train.py
```

You can change the configuration of `train.py` to train a classification model, or load a pre-trained model and finetune it. For `IceCubeModel_RegA`, the 0~32 epochs are trained with `L=196` and the 33∼40 epochs are trained with `L=256`.

## Predict

Download [pretrained model](https://github.com/ChenLi2049/ISeeCube/releases/tag/v0.0.1) and put it in the `pretrained_model` folder in the folder of this repository. Then run `predict.ipynb`.

## Acknowledgements

- Thanks to IceCube and Kaggle for this amazing competition.
- Lots of code are from [2nd place solution](https://github.com/DrHB/icecube-2nd-place/) in the Kaggle competition. I really appreciate it.
- Thanks to these repositories: [`torchscale`](https://github.com/microsoft/torchscale), [`graphnet`](https://github.com/graphnet-team/graphnet), [`fastai`](https://github.com/fastai/fastai).
- Kaggle solutions, discussions and notebooks are helpful.
- Thanks to [arxiv-style](https://github.com/kourgeorge/arxiv-style) for such a beautiful LaTeX template.
- Thanks to many other developers and communicators for their dedication.

##  Citation

If you find this repository useful, please consider citing our work:

```
@article{iseecube,
  doi = {10.1088/1748-0221/19/06/T06003},
  url = {https://dx.doi.org/10.1088/1748-0221/19/06/T06003},
  year = {2024},
  publisher = {IOP Publishing},
  volume = {19},
  number = {06},
  pages = {T06003},
  author = {Chen Li and Hao Cai and Xianyang Jiang},
  title = {Refine neutrino events reconstruction with BEiT-3},
  journal = {Journal of Instrumentation}
}
```
