import os
from torchvision import datasets
import torchvision.transforms as T

from data.cifar import CIFAR100, CIFAR10, CIFAR10im
from data.imagenet import ImageNet
from data.randaugment import RandAugmentMC



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
    if args.dataset == 'imagenet':
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
        fixmatch_custom   = ImageNet(train_path, 
                                      transform=TransformFixMatch(mean=mean['imagenet'],
                                                                  std=std['imagenet']))
    
    else:
        print("ERROR: Fixmatch are only for ImageNet benchmarks.")
        print("line 47 (data/get_fixmatch_loader.py)\n")
        return 0, 0, 0
    
    return train_custom, unlabeled_custom, test_custom, fixmatch_custom



class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(224)])
        self.strong = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(224),
            RandAugmentMC(n=2, m=10)])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
    
