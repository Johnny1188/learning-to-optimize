import os
import random

import numpy as np
import torch
from torchvision import datasets, transforms

DATA_PATH = os.path.join(os.getenv("DATA_PATH"))
DEFAULT_SEED = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MNIST:
    def __init__(
        self,
        training=True,
        batch_size=128,
        only_classes=None,
        preload=False,
        normalize=False,
        seed=DEFAULT_SEED,
    ):
        ### set seed
        g = torch.Generator()
        g.manual_seed(seed)

        ### init dataset and loader
        trans = [transforms.ToTensor()]
        if normalize:
            trans.append(transforms.Normalize((0.1307,), (0.3081,)))
        if training:
            dataset = datasets.MNIST(
                DATA_PATH,
                train=True,
                download=False,
                transform=transforms.Compose(trans),
            )
        else:
            dataset = datasets.MNIST(
                DATA_PATH,
                train=False,
                download=False,
                transform=transforms.Compose(trans),
            )

        if only_classes is not None:
            idx = torch.isin(
                dataset.targets,
                only_classes
                if type(only_classes) == torch.Tensor
                else torch.tensor(only_classes),
            )
            dataset.targets = dataset.targets[idx]
            dataset.data = dataset.data[idx]

        self.training = training
        self.only_classes = only_classes
        self.batch_size = batch_size
        self.seed = seed

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.batches = []
        self.curr_batch = 0

        if preload:
            self.preload_batches()

    def count_class_samples(self):
        """Counts the number of samples for each class in the dataset, returns a dictionary."""
        counts = {i: 0 for i in range(10)}
        for batch in self.loader:
            for label in batch[1]:
                counts[label.item()] += 1
        return counts

    def preload_batches(self, clear_batches=True):
        if clear_batches:
            self.batches = []
            self.curr_batch = 0
        for b in self.loader:
            self.batches.append(b)

    def sample(self):
        if self.curr_batch >= len(self.batches):
            self.preload_batches(clear_batches=True)
        batch = self.batches[self.curr_batch]
        self.curr_batch += 1
        return batch


class FashionMNIST:
    def __init__(
        self,
        training=True,
        batch_size=128,
        only_classes=None,
        preload=False,
        normalize=False,
        seed=DEFAULT_SEED,
    ):
        ### set seed
        g = torch.Generator()
        g.manual_seed(seed)

        ### init dataset and loader
        trans = [transforms.ToTensor()]
        if normalize:
            trans.append(transforms.Normalize((0.1307,), (0.3081,)))
        if training:
            dataset = datasets.FashionMNIST(
                DATA_PATH,
                train=True,
                download=False,
                transform=transforms.Compose(trans),
            )
        else:
            dataset = datasets.FashionMNIST(
                DATA_PATH,
                train=False,
                download=False,
                transform=transforms.Compose(trans),
            )

        if only_classes is not None:
            idx = torch.isin(
                dataset.targets,
                only_classes
                if type(only_classes) == torch.Tensor
                else torch.tensor(only_classes),
            )
            dataset.targets = dataset.targets[idx]
            dataset.data = dataset.data[idx]

        self.training = training
        self.only_classes = only_classes
        self.batch_size = batch_size
        self.seed = seed

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.batches = []
        self.curr_batch = 0

        if preload:
            self.preload_batches()

    def count_class_samples(self):
        """Counts the number of samples for each class in the dataset, returns a dictionary."""
        counts = {i: 0 for i in range(10)}
        for batch in self.loader:
            for label in batch[1]:
                counts[label.item()] += 1
        return counts

    def preload_batches(self, clear_batches=True):
        if clear_batches:
            self.batches = []
            self.curr_batch = 0
        for b in self.loader:
            self.batches.append(b)

    def sample(self):
        if self.curr_batch >= len(self.batches):
            self.preload_batches(clear_batches=True)
        batch = self.batches[self.curr_batch]
        self.curr_batch += 1
        return batch


class CIFAR10:
    def __init__(
        self,
        training=True,
        batch_size=128,
        only_classes=None,
        preload=False,
        rgb=True,
        resize_to=None,
        seed=DEFAULT_SEED,
    ):
        ### set seed
        g = torch.Generator()
        g.manual_seed(seed)

        ### build transforms
        if rgb:
            img_transformation = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            img_transformation = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)),
                ]
            )
        if resize_to is not None:
            img_transformation = transforms.Compose(
                [
                    transforms.Resize(resize_to),
                    img_transformation,
                ]
            )

        ### init dataset and loader
        if training:
            dataset = datasets.CIFAR10(
                DATA_PATH,
                train=True,
                download=False,
                transform=img_transformation,
            )
        else:
            dataset = datasets.CIFAR10(
                DATA_PATH,
                train=False,
                download=False,
                transform=img_transformation,
            )

        dataset.targets = torch.tensor(dataset.targets)
        if only_classes is not None:
            idx = torch.isin(
                dataset.targets,
                only_classes
                if type(only_classes) == torch.Tensor
                else torch.tensor(only_classes),
            )
            dataset.targets = dataset.targets[idx]
            dataset.data = dataset.data[idx]

        self.training = training
        self.only_classes = only_classes
        self.batch_size = batch_size
        self.seed = seed

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.batches = []
        self.curr_batch = 0

        if preload:
            self.preload_batches()

    def count_class_samples(self):
        """Counts the number of samples for each class in the dataset, returns a dictionary."""
        counts = {i: 0 for i in range(10)}
        for batch in self.loader:
            for label in batch[1]:
                counts[label.item()] += 1
        return counts

    def preload_batches(self, clear_batches=True):
        if clear_batches:
            self.batches = []
            self.curr_batch = 0
        for b in self.loader:
            self.batches.append(b)

    def sample(self):
        if self.curr_batch >= len(self.batches):
            self.preload_batches(clear_batches=True)
        batch = self.batches[self.curr_batch]
        self.curr_batch += 1
        return batch
