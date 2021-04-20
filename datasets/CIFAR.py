import os

from torchvision import datasets
from PIL import Image
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from datasets.datasets_utils import buildLabelIndex, getItem
            
            
class CIFAR100(datasets.CIFAR100):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            num_imgs_per_cat = None, training_mode = 'SSL'
    ) -> None:

        super(CIFAR100, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.training_mode = training_mode
        
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        
        if (num_imgs_per_cat is not None) and self.train:
            self._keep_first_k_examples_per_category(num_imgs_per_cat)
        
        # __getitem__ and __len__ inherited from ImageFolder
        
    def _keep_first_k_examples_per_category(self, num_imgs_per_cat):
        print('num_imgs_per_category {0}'.format(num_imgs_per_cat))
   
        labels = self.targets
        data = self.data
        label2ind = buildLabelIndex(labels)
        all_indices = []
        for cat in label2ind.keys():
            label2ind[cat] = label2ind[cat][:num_imgs_per_cat]
            all_indices += label2ind[cat]
        all_indices = sorted(all_indices)
        data = data[all_indices]
        labels = [labels[idx] for idx in all_indices]
        self.targets = labels
        self.data = data

        label2ind = buildLabelIndex(labels)
        for k, v in label2ind.items(): 
            assert(len(v)==num_imgs_per_cat)    
            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.fromarray(self.data[index])
        
        target = self.targets[index]
        
        return getItem(X, target, self.transform, self.training_mode)



class CIFAR10(datasets.CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            num_imgs_per_cat = None, training_mode = 'SSL'
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        
        self.training_mode = training_mode
            
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        
        if (num_imgs_per_cat is not None) and self.train:
            self._keep_first_k_examples_per_category(num_imgs_per_cat)
        
        # __getitem__ and __len__ inherited from ImageFolder
        
    def _keep_first_k_examples_per_category(self, num_imgs_per_cat):
        print('num_imgs_per_category {0}'.format(num_imgs_per_cat))
   
        labels = self.targets
        data = self.data
        label2ind = buildLabelIndex(labels)
        all_indices = []
        for cat in label2ind.keys():
            label2ind[cat] = label2ind[cat][:num_imgs_per_cat]
            all_indices += label2ind[cat]
        all_indices = sorted(all_indices)
        data = data[all_indices]
        labels = [labels[idx] for idx in all_indices]
        self.targets = labels
        self.data = data

        label2ind = buildLabelIndex(labels)
        for k, v in label2ind.items(): 
            assert(len(v)==num_imgs_per_cat)        
 
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.fromarray(self.data[index])
        target = self.targets[index]
        
        return getItem(X, target, self.transform, self.training_mode)