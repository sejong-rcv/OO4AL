import os
from torchvision import datasets
import torchvision.transforms as T

from data.cifar import CIFAR100, CIFAR10, CIFAR10im
from data.imagenet import ImageNet



mean = {
'cifar10': (0.4914, 0.4822, 0.4465),
'cifar10im': (0.4914, 0.4822, 0.4465),
'cifar100': (0.5071, 0.4867, 0.4408),
'imagenet': (0.485, 0.456, 0.406)
}

std = {
'cifar10': (0.2023, 0.1994, 0.2010),
'cifar10im': (0.2023, 0.1994, 0.2010),
'cifar100': (0.2675, 0.2565, 0.2761),
'imagenet' : (0.229, 0.224, 0.225)
}


def get_dataloader(data_name, args):
    if data_name == 'cifar10':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize(mean[args.dataset], std[args.dataset])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean[args.dataset], std[args.dataset])
        ]) 
        train_custom = CIFAR10(args.data_path, train=True, download=True, transform=train_transform)
        unlabeled_custom   = CIFAR10(args.data_path, train=True, download=True, transform=test_transform)
        test_custom  = CIFAR10(args.data_path, train=False, download=True, transform=test_transform)
        if args.approach == 'VAAL':
            train_custom.tavaal = True
            unlabeled_custom.tavaal = True
            test_custom.tavaal = True
    elif data_name == 'cifar100':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize(mean[args.dataset], std[args.dataset])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean[args.dataset], std[args.dataset])
        ]) 
        train_custom = CIFAR100(args.data_path, train=True, download=True, transform=train_transform)
        unlabeled_custom   = CIFAR100(args.data_path, train=True, download=True, transform=test_transform)
        test_custom  = CIFAR100(args.data_path, train=False, download=True, transform=test_transform)
        if args.approach == 'VAAL':
            train_custom.tavaal = True
            unlabeled_custom.tavaal = True
            test_custom.tavaal = True
    elif args.dataset == 'imagenet':
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean[args.dataset], std[args.dataset])
        ])
        test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean[args.dataset], std[args.dataset])
        ]) 
        train_path = os.path.join(args.data_path, 'train_task12')
        test_path = os.path.join(args.data_path, 'validation')
        train_custom = ImageNet(train_path, transform=train_transform)
        unlabeled_custom   = ImageNet(train_path, transform=test_transform)
        test_custom  = datasets.ImageFolder(test_path, transform=test_transform)
    elif args.dataset == 'cifar10im':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize(mean[args.dataset], std[args.dataset])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean[args.dataset], std[args.dataset])
        ]) 
        train_custom = CIFAR10im(args.data_path, train=True, download=True, transform=train_transform)
        unlabeled_custom   = CIFAR10im(args.data_path, train=True, download=True, transform=test_transform)
        test_custom  = CIFAR10(args.data_path, train=False, download=True, transform=test_transform)
        if args.approach == 'VAAL':
            train_custom.tavaal = True
            unlabeled_custom.tavaal = True
            test_custom.tavaal = True
    
    return train_custom, unlabeled_custom, test_custom
