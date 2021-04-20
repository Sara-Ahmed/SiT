import os

from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from datasets.TinyImageNet import TinyImageNetDataset
from datasets.CIFAR import CIFAR10, CIFAR100
from datasets.STL10 import STL10
    
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.dataset_location, 'CIFAR10_dataset'), 
                          download=True, train=is_train, transform=transform, 
                          num_imgs_per_cat=args.num_imgs_per_cat,
                          training_mode = args.training_mode)
        nb_classes = 10

    
    elif args.data_set == 'CIFAR100':
        dataset = CIFAR100(os.path.join(args.dataset_location, 'CIFAR100_dataset'), 
                           download=True, train=is_train, transform=transform, 
                           num_imgs_per_cat=args.num_imgs_per_cat,
                           training_mode = args.training_mode)
        
        nb_classes = 100
        

    elif args.data_set == 'STL10':
        #### Note num_imgs_per_cat is not implemented in this dataset as it has unlabeled data
        split = 'train+unlabeled' if args.training_mode=='SSL' else 'train'
        split = split if is_train else 'test'
        
        dataset = STL10(root=os.path.join(args.dataset_location, 'STL10'), 
                        download=True, split=split, transform=transform,
                          training_mode = args.training_mode)
        nb_classes = 10
        
    elif args.data_set == 'TinyImageNet':
        mode='train' if is_train else 'val'
        root_dir = os.path.join(args.dataset_location, 'TinyImageNet/tiny-imagenet-200/')
        dataset = TinyImageNetDataset(root_dir=root_dir, download=True, mode=mode, transform=transform, 
                          num_imgs_per_cat=args.num_imgs_per_cat,
                          training_mode = args.training_mode)
        nb_classes = 200


    return dataset, nb_classes





def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


