import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# Custom
from utils.config_tavaal import *
from models.query_models import VAE, Discriminator, GCN
from data.sampler import SubsetSequentialSampler
from utils.kcenterGreedy import kCenterGreedy

def get_RankedMS(models, unlabeled_loader, class_num):
    models['backbone'].eval()
    gradient_scores = torch.tensor([])
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    # https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            scores = models['backbone'](inputs)
            # output calibration
            calibration_mask = temperature.unsqueeze(1).expand(scores.size(0), scores.size(1)).cuda()
            calib_scores = scores/calibration_mask
            # output calibration
            sorted_scores, sorted_indexs = torch.sort(calib_scores, dim=1, descending=True)
            sorted_scores = torch.nn.functional.softmax(sorted_scores)
            scores_gap = sorted_scores[:, :class_num-1] - sorted_scores[:, 1:] 
            position_map = torch.tensor(range(1, class_num)).expand_as(scores_gap)
            g_score = torch.sum(scores_gap.cpu() / position_map, dim=1)
            gradient_scores = torch.cat((gradient_scores, g_score), 0)

    return gradient_scores.cpu()

def get_margin(models, unlabeled_loader):
    models['backbone'].eval()
    margin_scores = torch.tensor([])
    temperature = nn.Parameter(torch.ones(1) * 1.5)

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            scores = models['backbone'](inputs)
            # output calibration
            calibration_mask = temperature.unsqueeze(1).expand(scores.size(0), scores.size(1)).cuda()
            calib_scores = scores/calibration_mask
            # output calibration
            sorted_scores, sorted_indexs = torch.sort(calib_scores, dim=1, descending=True)
            sorted_scores = torch.nn.functional.softmax(sorted_scores)
            scores_gap = sorted_scores[:, 0] - sorted_scores[:, 1]
            margin_scores = torch.cat((margin_scores, scores_gap.cpu()), 0)

    return margin_scores.cpu()

def get_entropy(models, unlabeled_loader, class_num):
    models['backbone'].eval()
    entropy_scores = []
    temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            scores = models['backbone'](inputs)
            # output calibration
            calibration_mask = temperature.unsqueeze(1).expand(scores.size(0), scores.size(1)).cuda()
            calib_scores = scores/calibration_mask
            # output calibration
            scores = torch.nn.functional.softmax(calib_scores)
            scores = scores.cpu()
            entropy = - np.nansum(scores * np.log(scores), axis=1)
            entropy_scores.append(entropy)
    
    entropy_scores = np.concatenate(entropy_scores)
    return entropy_scores


def get_entropy_no_softmax(models, unlabeled_loader, class_num):
    models['backbone'].eval()
    entropy_scores = []
    temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            scores = models['backbone'](inputs)
            # output calibration
            calibration_mask = temperature.unsqueeze(1).expand(scores.size(0), scores.size(1)).cuda()
            calib_scores = scores/calibration_mask
            # output calibration
            # scores = torch.nn.functional.softmax(calib_scores)
            calib_scores = calib_scores.cpu()
            entropy = - np.nansum(calib_scores * np.log(calib_scores), axis=1)
            entropy_scores.append(entropy)
    
    entropy_scores = np.concatenate(entropy_scores)
    return entropy_scores

        
def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl) 
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss

def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label in dataloader:
                yield img, label
    else:
        while True:
            for img, _ in dataloader:
                yield img

def read_data_tavaal(dataloader, labels=True):
    if labels:
        while True:
            for img, label, _ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            inputs = inputs.cuda()
            scores, features = models['backbone'].forward_ll4al(inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()

def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    features = torch.tensor([]).cuda()    
    with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                inputs = inputs.cuda()
                _, features_batch, _ = models['backbone'].forward_feature_ll4al(inputs)
                features = torch.cat((features, features_batch), 0)
            feat = features #.detach().cpu().numpy()
    return feat

def get_kcg(models, labeled_data_size, unlabeled_loader, args):
    models['backbone'].eval()
    features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            inputs = inputs.cuda()
            _, features_batch = models['backbone'].forward_feature(inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(args.sampling_num,(args.sampling_num + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, args.selected)
        other_idx = [x for x in range(args.sampling_num) if x not in batch]
    return  other_idx + batch


# Select the indices of the unlablled data according to the methods
def query_samples_calibration(model, method, data_unlabeled, subset, labeled_set, cycle, args, num_labeled):
    flag = 0
    if method == 'Random' or method == 'Mixup' or method == 'CMaM':
        arg = range(args.sampling_num)
    
    if method == 'rankedms':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    num_workers=args.workers,
                                    pin_memory=True)
        # Measure uncertainty of each data points in the subset
        gradient_score = get_RankedMS(model, unlabeled_loader, args.class_num)
        arg = np.argsort(gradient_score)
        flag = 1

    # select the points which the discriminator things are the most likely to be unlabeled
    if method == 'margin':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    num_workers=args.workers,
                                    pin_memory=True)
        # Measure uncertainty of each data points in the subset
        gradient_score = get_margin(model, unlabeled_loader)
        arg = np.argsort(gradient_score)
        flag = 1
    
    if method == 'entropy':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset),
                                    num_workers=args.workers, 
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        entropy_score = get_entropy(model, unlabeled_loader, args.class_num)
        arg = np.argsort(entropy_score)

    if method == 'entropy_no_softmax':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset),
                                    num_workers=args.workers, 
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        entropy_score = get_entropy_no_softmax(model, unlabeled_loader, args.class_num)
        arg = np.argsort(entropy_score)

    return arg, flag