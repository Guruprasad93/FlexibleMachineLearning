import logging
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import pickle
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class CIFARDataset(Dataset):

    def __init__(self, data_dir, train = True, transform = None):

        self.data_file = data_dir +"/train" if train else data_dir+"/test"
        self.meta_file = data_dir +"/meta"
        self.transform = transform

        with open(self.data_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        with open(self.meta_file, "rb") as f:
            meta = pickle.load(f, encoding="latin1")

        self.images = data["data"]
        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC
        self.labels = np.array(data["fine_labels"])
        self.names = np.array(meta["fine_label_names"])

        # self._filter_classes()
        # self._visualize_image()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = torch.tensor(self.labels[idx]).long()

        if self.transform is not None:
            img = self.transform(img)
            # label = label.view(1)
        return img, label

    def _filter_classes(self, min=0, max=10, relabel=True):
        filtered = (self.labels >= min) & (self.labels < max)
        self.images = self.images[filtered]
        self.labels = self.labels[filtered]
        if relabel:
            self.labels -= min
        self.names = self.names[min:max]
        # print(self.names)

    def _filter_size(self, data_size):
        self.images = self.images[:data_size]
        self.labels = self.labels[:data_size]

    def _visualize_image(self):
        idx = np.random.randint(len(self), size=1)[0]
        img, label = self[idx]
        class_name = self.names[int(label.item())]
        print(idx, class_name)
        plt.imshow(img)
        plt.show()

def get_loader(args, cmin = 0, cmax = 10, relabel = True, data_size = None):

    data_dir = "data/cifar-100-python"

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = CIFARDataset(data_dir, train = True, transform = transform_train)
    trainset._filter_classes(cmin, cmax, relabel)
    if data_size is not None:
        trainset._filter_size(data_size)
    testset = CIFARDataset(data_dir, train = False, transform = transform_test)
    testset._filter_classes(cmin, cmax, relabel)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
