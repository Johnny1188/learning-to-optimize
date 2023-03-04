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
    def __init__(self, training=True, batch_size=128, seed=DEFAULT_SEED):
        ### set seed
        g = torch.Generator()
        g.manual_seed(seed)

        ### init dataset and loader
        if training:
            dataset = datasets.MNIST(
                DATA_PATH,
                train=True,
                download=False,
                transform=transforms.ToTensor(),
            )
        else:
            dataset = datasets.MNIST(
                DATA_PATH,
                train=False,
                download=False,
                transform=transforms.ToTensor(),
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


# Data transformations and loading - MNIST
def get_mnist_data_loaders(
    batch_size=32,
    flatten=False,
    drop_last=True,
    only_classes=None,
    img_size=28,
    seed=DEFAULT_SEED,
):
    ### set seed
    g = torch.Generator()
    g.manual_seed(seed)

    ### build transforms
    img_transformation = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    if img_size < 28 and img_size >= 24:
        img_transformation.transforms.append(transforms.Resize(img_size))
    elif img_size < 24:
        img_transformation.transforms.append(transforms.CenterCrop(24))
        img_transformation.transforms.append(transforms.Resize(img_size))
    if flatten:
        img_transformation.transforms.append(
            transforms.Lambda(lambda x: torch.flatten(x))
        )

    train_dataset = datasets.MNIST(
        DATA_PATH, train=True, download=False, transform=img_transformation
    )
    if only_classes != None:  # list of classes to select from the dataset (0,1,...)
        idx = torch.isin(
            train_dataset.targets,
            only_classes
            if type(only_classes) == torch.Tensor
            else torch.tensor(only_classes),
        )
        train_dataset.targets = train_dataset.targets[idx]
        train_dataset.data = train_dataset.data[idx]
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_dataset = datasets.MNIST(
        DATA_PATH, train=False, download=False, transform=img_transformation
    )
    if only_classes != None:  # list of classes to select from the dataset (0,1,...)
        idx = torch.isin(
            test_dataset.targets,
            only_classes
            if type(only_classes) == torch.Tensor
            else torch.tensor(only_classes),
        )
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return (
        train_loader,
        test_loader,
        datasets.MNIST.classes if only_classes is None else only_classes,
    )
