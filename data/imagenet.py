from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy

    
class ImageNet(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.imagenet = datasets.ImageFolder(root=path, transform=self.transform)
        self.mode_tavaal = False

    def change_mode_tavaal(self, value):
        self.mode_tavaal = value # true or false

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.imagenet[index]

        if self.mode_tavaal:
            return data, target, index
    
        return data, target

    def __len__(self):
        return len(self.imagenet)