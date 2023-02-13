import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets

DATA_PATH = os.getenv("DATA_PATH")


class MNISTLoss:
    def __init__(self, training=True):
        dataset = datasets.MNIST(
            DATA_PATH,
            train=True,
            download=False,
            transform=torchvision.transforms.ToTensor(),
        )
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        if training:
            indices = indices[: len(indices) // 2]
        else:
            indices = indices[len(indices) // 2 :]

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
        )

        self.batches = []
        self.cur_batch = 0

    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch
