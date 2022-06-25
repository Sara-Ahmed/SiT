import os

from datasets.TinyImageNet import TinyImageNetDataset
from datasets.CIFAR import CIFAR10, CIFAR100
from datasets.STL10 import STL10
from datasets.ImageNet import ImageNetDataset
from datasets.Cars_stanford import Cars
from datasets.Flowers_stanford import Flowers
from datasets.AirCraft import Aircraft
from datasets.CUB import Cub2011
from datasets.Pets import pets

import torchvision

def build_dataset(args, is_train, trnsfrm=None, training_mode='SSL'):

    if args.data_set == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.data_location, 'CIFAR10_dataset'), 
                          download=True, train=is_train, transform=trnsfrm, 
                          training_mode = training_mode)
        nb_classes = 10


    elif args.data_set == 'MNIST':
        dataset = torchvision.datasets.MNIST(os.path.join(args.data_location, 'MNIST_dataset'), 
                                   train=is_train, transform=trnsfrm, download=True)

        nb_classes = 10
    
    elif args.data_set == 'CIFAR100':
        dataset = CIFAR100(os.path.join(args.data_location, 'CIFAR100_dataset'), 
                           download=True, train=is_train, transform=trnsfrm, 
                           training_mode = training_mode)
        
        nb_classes = 100
        
    
    elif args.data_set == 'Aircraft':
        dataset = Aircraft(os.path.join(args.data_location, 'Aircraft_dataset'), train=is_train, transform=trnsfrm)
        
        nb_classes = 100
    
    elif args.data_set == 'CUB':
        dataset = Cub2011(os.path.join(args.data_location, 'CUB_dataset'), train=is_train, transform=trnsfrm)
        
        nb_classes = 200
        
    elif args.data_set == 'Pets':
        split = 'trainval' if is_train else 'test'
        dataset = pets(os.path.join(args.data_location, 'Pets_dataset'), split=split, transform=trnsfrm)
        
        nb_classes = 37

    elif args.data_set == 'STL10':
        split = 'train+unlabeled' if training_mode=='SSL' else 'train'
        split = split if is_train else 'test'
        
        dataset = torchvision.datasets.STL10(root=os.path.join(args.data_location, 'STL10'), 
                        download=True, split=split, transform=trnsfrm)
        nb_classes = 10

    elif args.data_set == 'Cars':
        file_root = os.path.join(args.data_location, 'carsStanford/car_data/car_data/')
        
        if is_train:
            datafiles = file_root+'/train' #'TrainFiles_50Samples.csv'
        else:
            datafiles = file_root+'/test'
        dataset = Cars(datafiles, transform=trnsfrm, training_mode = training_mode)
        
        nb_classes = 196

    elif args.data_set == 'Flowers':
        file_root = os.path.join(args.data_location, 'Flowers')
        
        if is_train:
            datafiles = file_root+'/train' #'TrainFiles_50Samples.csv'
        else:
            datafiles = file_root+'/test'
        dataset = Flowers(datafiles, transform=trnsfrm, training_mode = training_mode)
        
        nb_classes = 102
   
    elif args.data_set == 'TinyImageNet':
        mode='train' if is_train else 'val'
        root_dir = os.path.join(args.data_location, 'TinyImageNet/tiny-imagenet-200/')
        dataset = TinyImageNetDataset(root_dir=root_dir, mode=mode, transform=trnsfrm, 
                          training_mode = training_mode)
        nb_classes = 200

    
    elif args.data_set == 'ImageNet':
        file_root = 'datasets/ImageNet_files/'

        if is_train:
            datafiles = file_root + 'TrainFiles_1300Samples_shuffled_abs.csv'
        else:
            datafiles = file_root + 'INet_val.csv'
            
        file_loc = os.path.join(args.data_location, 'still/ImageNet/ILSVRC2012')
        dataset = ImageNetDataset(datafiles, dataset_path=file_loc, transform=trnsfrm, training_mode = training_mode)
        
        nb_classes = 1000

    return dataset, nb_classes


