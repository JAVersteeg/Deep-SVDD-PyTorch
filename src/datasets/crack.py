import torch.utils.data as data
from torch.utils.data import Subset
from PIL import Image
from .preprocessing import get_target_label_idx, global_contrast_normalization
from base.crack_dataset import CrackDataset
from sklearn.model_selection import train_test_split
from copy import deepcopy, copy
import logging
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import matplotlib.image as mpimg
import random
from time import time

class CRACK_Dataset(CrackDataset):

    def __init__(self, root:str, normal_class=0, patch_size=64):
        super().__init__(root)

        self.n_classes = 0  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)

        # Crack data preprocessing:
        transform = transforms.Compose([transforms.ToTensor()])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = MyCrack(root=self.root, train=True,
                              transform=transform, target_transform=target_transform,
                              patch_size=patch_size)
        self.train_set.set_to_train()
        
        self.test_set = deepcopy(self.train_set)
        self.test_set.set_to_test()
        
        self.test_set_corner = deepcopy(self.test_set) # copy train set object and set to corner test set
        self.test_set_corner.set_to_test(corner_cracks=True)

class MyCrack(data.Dataset):

    def __init__(self, root: str, train: bool=True, transform=None,
                 target_transform=None, patch_size=64):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.test_split_size = 0.2

        # Ensures that data is initialized only once, namely before training:
        if patch_size == 128:
            non_crack_data = self.load_images("../../patches_1119/non_crack/img")
            #crack_data = self.load_images("../../patches/crack/img-no-inv")
            crack_data = self.load_images("../../patches_1119/crack/img_ncc")
            self.crack_data_remaining = self.load_images("../../patches_1119/crack/img_ncc_remaining")            
        else:
            non_crack_data = self.load_images("../../non_crack/img")
            #crack_data = self.load_images("../../crack/img-no-inv")
            crack_data = self.load_images("../../crack/img_ncc")    # no corner cracks
            self.crack_data_remaining = self.load_images("../../crack/img_ncc_remaining")
        
        print("Test split size: ", self.test_split_size)

        self.X = np.concatenate((non_crack_data, crack_data), axis=0)
        self.y = np.concatenate((np.zeros(len(non_crack_data)), np.ones(len(crack_data))), axis=0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_split_size, shuffle=True, stratify=self.y)

    def set_to_train(self):
        self.data = self.X_train
        self.labels = self.y_train
        self.normal_classes = np.arange(len(self.data)) # indices of train_labels
        print("Length training set:", len(self.data))
        return self

    def set_to_test(self, corner_cracks=False):
        logger = logging.getLogger()
        
        self.data = self.X_test
        self.labels = self.y_test
        if corner_cracks:
            random.seed(time())
            logger.info("Creating test set without corner cracks")
            
            remaining_samples = random.sample(self.crack_data_remaining, round(len(self.crack_data_remaining)*self.test_split_size))
            outliers_data = list(self.data[self.labels == 1])
            outliers_data = random.sample(outliers_data, round(len(outliers_data)*(1-self.test_split_size)))
            outliers_data = np.concatenate((remaining_samples, outliers_data), axis=0)
            outliers_data = list(map(lambda x: x.tolist(), outliers_data))
            normals_data = self.data[self.labels == 0]
            
            self.data = np.append(outliers_data, normals_data, axis=0)
            
            t = copy(self.labels)
            self.labels = self.labels[t == 0]
            self.labels = np.concatenate((np.ones(len(outliers_data)), self.labels), axis=0)
            
        self.normal_classes = np.arange(len(self.data)) # indices of test_labels    
        
        if not corner_cracks:
            print("Length test set:    ", len(self.data))
        else:
            print("Len test set corner:", len(self.data))
        return self

    def load_images(self, folder):                  # Remove timer and comments
        logger = logging.getLogger()
        start_load = time()
        images = []
        sample_size = 500000
        counter = 0
        print("Loading", folder, "images...")
        folder_time = 0
        for filename in os.scandir(folder):
            start_folder = time()
            skip = 1
            if counter % skip == 0 and (folder == "img" or folder == "../../non_crack/img" or folder == "../../patches_1119/non_crack/img"):
                img = mpimg.imread(os.path.join(folder,filename.name))
                end_folder = time()
                folder_time += (end_folder-start_folder)
                images.append(img)  
                counter += 1
                if counter % 10000 == 0:
                    logger.info('Loading 10000 (of %s) images took: %s' % (counter/skip, folder_time))
                    folder_time = 0
            elif not folder == "img" and not folder == "../../non_crack/img": 
                counter += 1
                img = mpimg.imread(os.path.join(folder,filename.name))
                images.append(img)
           
            if counter/skip == 500000:
                break
        sampling_time = time()
        
        images = list(map(lambda x: x.tolist(), images))
        if len(images) < sample_size:
            sample_size = len(images)
        
        random.seed(time())
        sampling_time_done = time()
        logger.info("Sampling %s took %s seconds" % (len(images), (round((sampling_time_done - sampling_time), 1))))
        return random.sample(images, sample_size)

    def __getitem__(self, idx):
        """Get method of the CrackDataset class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[idx], self.labels[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None: #img must be 2D
            img = self.transform(img)

        if self.target_transform is not None: #target must be 1D
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