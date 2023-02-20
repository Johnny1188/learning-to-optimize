import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets

DATA_PATH = os.getenv("DATA_PATH")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class MNIST:
    def __init__(self, training=True, batch_size=128):
        if training:
            dataset = datasets.MNIST(
                DATA_PATH,
                train=True,
                download=False,
                transform=torchvision.transforms.ToTensor(),
            )
        else:
            dataset = datasets.MNIST(
                DATA_PATH,
                train=False,
                download=False,
                transform=torchvision.transforms.ToTensor(),
            )

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
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


""" --- DEPRECATED ---
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
"""
