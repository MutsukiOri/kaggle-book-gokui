import numpy as np
import sklearn.model_selection

import argparse
import pathlib
import torch
import torchvision
import torchvision.transforms.functional
from torchvision import transforms
import os


def setup_center_crop_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def get_labels(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return get_labels(dataset.dataset)[dataset.indices]
    else:
        return np.array([img[1] for img in dataset.imgs])

def setup_train_val_split(labels, dryrun=False, seed=0):
    x = np.arange(len(labels))
    y = np.array(labels)
    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=0.8, random_state=seed
    )
    train_indices, val_indices = next(splitter.split(x, y))

    if dryrun:
        train_indices = np.random.choice(train_indices, 100, replace=False)
        val_indices = np.random.choice(val_indices, 100, replace=False)

    return train_indices, val_indices

def setup_train_val_datasets(data_dir, dryrun=False):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=setup_center_crop_transform(),
    )
    labels = get_labels(dataset)
    train_indices, val_indices = setup_train_val_split(labels, dryrun)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dogs_vs_cats')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    device = args.device
    dryrun = args.dryrun

    train, val = setup_train_val_datasets(data_dir, dryrun)

    print(f"train: {len(train)}")
    print(f"val: {len(val)}")

if __name__ == "__main__":
    main()