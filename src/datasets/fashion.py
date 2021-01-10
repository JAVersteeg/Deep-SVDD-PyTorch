import torch.utils.data as data
from torch.utils.data import Subset
from PIL import Image
from .preprocessing import get_target_label_idx, global_contrast_normalization
from base import FashionDataset
from base.torchvision_dataset import TorchvisionDataset

from utils import fashion_reader

import torchvision.transforms as transforms
import numpy as np

class FASHION_Dataset(FashionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # We don't have pre-computed min and max values for new dataset.
        # min_max = [(,)]

        # FASHION preprocessing:
        transform = transforms.Compose([transforms.ToTensor()])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyFASHION(root=self.root, train=True,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyFASHION(root=self.root, train=False,
                                  transform=transform, target_transform=target_transform)

        print("FASHION Shape:", self.test_set.data.shape)

class MyFASHION(data.Dataset):
    def __init__(self, root: str, train: bool=True, transform=None,
                 target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if train:
          self.data, self.labels = fashion_reader.load_fashion('datasets_other/fashion', kind='train')
        else:
          self.data, self.labels = fashion_reader.load_fashion('datasets_other/fashion', kind='t10k')
        self.normal_classes = np.arange(60000) # indices of train_labels

    def __getitem__(self, idx):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[idx], self.labels[idx]

        # goes in as 1D np array

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(np.reshape(img, (64,64)), mode='L')
        img = Image.fromarray(np.reshape(img, (28,28)), mode='L')

        #comes out in the same shape as mnist (not sure if as cifar)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, idx

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""