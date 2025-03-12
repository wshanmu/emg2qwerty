from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform
import hydra
import matplotlib.pyplot as plt
import os

def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
    return transforms.Compose([instantiate(cfg) for cfg in configs])

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    hdf5_path = Path("/data1/shanmu/emg2qwerty/data/2020-08-13-1597354281-keystrokes.hdf5")
    train_transform=_build_transform(config.transforms.train)

    train_dataset = WindowedEMGDataset(
                        hdf5_path,
                        transform=train_transform,
                        window_length=8000,
                        padding=[1800, 200],
                        jitter=False,
                    )
    train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
        )
    for batch in train_loader:
        # print(batch.keys())
        # print(batch['inputs'].max(), batch['inputs'].min())
        # exit()
        inputs = batch['inputs'][:, 0, 0, ...].detach().cpu().numpy()
        print(inputs.shape)
        import pywt
        # create a cmap based on the min and max values of the inputs
        cmap = plt.get_cmap('viridis')
        # normalize the inputs
        # inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        # apply the colormap
        print(batch['targets'])
        for i in range(16):
            plt.figure(figsize=(10, 8))
            y = inputs[:1000, i]
            coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
            plt.matshow(coef)
            plt.colorbar()
            plt.title(f'Channel {i}')
            plt.savefig(f'/data1/shanmu/emg2qwerty/emg2qwerty/data_visualization/channel_{i}_cwt.png', dpi=300)
        exit()

if __name__ == "__main__":
    main()
