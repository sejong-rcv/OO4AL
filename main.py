'''
"Overcoming Overconfidence for Active Learning" Procedure in PyTorch.
'''
# Python
import os
import random
# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
# Torchvison
import torchvision.models as models
# Custom
from models import vaal_model
from models import vgg
from utils.utils import *
from utils.train import *
from utils.argument import *
from data.get_dataloader import *
import models.lossnet as lossnet
import models.resnet_fusion as resnet
from utils.selection_methods import query_samples
from utils.temperature_scaling import ModelWithTemperature
from utils.hierarchical_sampling import *

if __name__ == '__main__':
    args = get_argument()
    logger = get_logger(args)
    # Fix the seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    # Make the directory for saving
    project_dir = os.path.join(*args.save_name.split('/')+['train','weights'])
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    # Dataloader
    train_custom, unlabeled_custom, test_custom = get_dataloader(args.dataset, args)
    indices = list(range(len(train_custom)))
    labeled_set, unlabeled_set = initial_budget_sampling(indices, args)
    train_loader = DataLoader(train_custom, batch_size=args.batch_size, num_workers=args.workers,
                                sampler=SubsetRandomSampler(labeled_set), 
                                pin_memory=True)
    test_loader  = DataLoader(test_custom, batch_size=args.batch_size, num_workers=args.workers)
    dataloaders  = {'train': train_loader, 'test': test_loader}
    # Model
    resnet18    = resnet.ResNet18(num_classes=args.class_num).cuda()
    torch.backends.cudnn.benchmark = False
    if args.approach == 'LL4AL' or args.approach == 'TA-VAAL':
        loss_module = lossnet.LossNet().cuda()
        models = {'backbone': resnet18, 'module': loss_module}    
    else:
        models      = {'backbone': resnet18}        

    # Active learning cycles
    for cycle in range(args.cycle_num):
        if args.approach == 'LL4AL' or args.approach == 'TA-VAAL':
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args.lr, 
                                    momentum=args.momentum, weight_decay=args.wdecay)
            optim_module   = optim.SGD(models['module'].parameters(), lr=args.lr, 
                                    momentum=args.momentum, weight_decay=args.wdecay)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.milestones)
            sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=args.milestones)
            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}
        else:
            criterion      = nn.CrossEntropyLoss()
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args.lr, 
                                    momentum=args.momentum, weight_decay=args.wdecay)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.milestones)
            optimizers = {'backbone': optim_backbone }
            schedulers = {'backbone': sched_backbone }
        
        # Training and test
        train(models, criterion, optimizers, schedulers, dataloaders, args)
        acc = test(models, dataloaders, mode='test')
        logger.info('Cycle {}/{} || Label set size {}: Test acc {}'.format(cycle+1, args.cycle_num, len(labeled_set), acc))

        if 'TS' in args.approach:
            calibrated_model = ModelWithTemperature(models['backbone'])
            calibrated_model.set_temperature(dataloaders['train'])
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:args.sampling_num]
            arg, flag = query_samples({'backbone':calibrated_model}, args.approach, unlabeled_custom, subset, labeled_set, cycle, args, len(labeled_set))
        else:
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:args.sampling_num]
            arg, flag = query_samples(models, args.approach, unlabeled_custom, subset, labeled_set, cycle, args, len(labeled_set))
        
        if flag == 1:
            selected_set = list(torch.tensor(subset)[arg][:args.selected].numpy())
            labeled_set += selected_set
            unlabeled_set = list(torch.tensor(subset)[arg][args.selected:].numpy()) + unlabeled_set[args.sampling_num:]
        elif flag == 2: # Hierarchical
            selected_set = np.array(subset)[arg].tolist()
            labeled_set += selected_set
            unlabeled_set = list(set(subset) - set(selected_set)) + unlabeled_set[args.sampling_num:]
        else:
            selected_set = list(torch.tensor(subset)[arg][-args.selected:].numpy())
            labeled_set += selected_set
            unlabeled_set = list(torch.tensor(subset)[arg][:-args.selected].numpy()) + unlabeled_set[args.sampling_num:]

        # Create a new dataloader for the updated labeled dataset
        dataloaders['train'] = DataLoader(train_custom, batch_size=args.batch_size, num_workers=args.workers,
                                        sampler=SubsetRandomSampler(labeled_set), 
                                        pin_memory=True)
        
        # Save a checkpoint
        torch.save({
                    'cycle': cycle + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                },
                './{}/train/weights/{}_{}_cycle{}.pth'.format(args.save_name, args.dataset, args.approach, cycle))
