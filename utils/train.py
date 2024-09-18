from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

## OO4AL
def mixed_criterion_cross_fusion(criterion, pred, y_a, y_b, lam1, lam2):
    return lam2 * (lam1 * criterion(pred, y_a) + (1 - lam1) * criterion(pred, y_b)) \
            + (1 - lam2) * ((1 - lam1) * criterion(pred, y_a) + lam1 * criterion(pred, y_b))

def cross_fusion_data(x, y, index=None, alpha=0.8, use_cuda=True, lam=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if lam == None:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

    batch_size = x.size()[0]
    if index == None:
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

    mixed_x1 = lam * x + (1 - lam) * x[index, :]
    mixed_x2 = (1 - lam) * x + lam * x[index, :]

    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam

def train_epoch_for_CMaM(models, criterion, optimizers, dataloaders, alpha=0.8, layer=1):
    models['backbone'].train() 

    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        # make the pair
        inputs1, inputs2, targets_a, targets_b, lam1 = cross_fusion_data(inputs, labels, alpha=alpha, use_cuda=True)
        inputs1, inputs2, targets_a, targets_b = map(Variable, (inputs1, inputs2, targets_a, targets_b))
        lam2 = np.random.beta(alpha, alpha)

        # start the model training
        optimizers['backbone'].zero_grad()
        scores = models['backbone'].layer_fusion_multi(inputs1, inputs2, lam2, layer=layer)
        loss = mixed_criterion_cross_fusion(criterion, scores, targets_a, targets_b, lam1, lam2)
        loss.backward()
        optimizers['backbone'].step()
## OO4AL

## Random
def train_epoch_forRandom(models, criterion, optimizers, dataloaders):
    models['backbone'].train()

    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()
        
        # start training
        scores = models['backbone'](inputs)
        loss = criterion(scores, labels)
        loss.backward()
        optimizers['backbone'].step()
## Random

## VAAL
def vae_loss(x, recon, mu, logvar, beta, mse_loss):
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_epoch_for_VAAL(models, ce_loss, mse_loss, bce_loss, optimizers, dataloaders, args):
    models['backbone'].train()
    models['vae'].train()
    models['discriminator'].train()
    labeled_data = iter(dataloaders['train'])
    unlabeled_data = iter(dataloaders['unlabeled'])
    
    train_iterations = len(dataloaders['train'])
    for iter_count in range(train_iterations):
        try:
            labeled_imgs, labels, index = next(labeled_data)
        except:
            labeled_data = iter(dataloaders['train'])
            labeled_imgs, labels, index = next(labeled_data)
        try:
            unlabeled_imgs, _, _ = next(unlabeled_data)
        except:
            unlabeled_data = iter(dataloaders['unlabeled'])
            unlabeled_imgs, _, _ = next(unlabeled_data)
        unlabeled_imgs, labeled_imgs, labels = unlabeled_imgs.cuda(), labeled_imgs.cuda(), labels.cuda()

        optimizers['backbone'].zero_grad()
        
        # start training
        scores = models['backbone'](labeled_imgs)
        loss = ce_loss(scores, labels)
        loss.backward()
        optimizers['backbone'].step()

        # VAE step
        
        beta = 1 # https://github.com/sinhasam/vaal/blob/master/arguments.py
        adversary_param = 1 # https://github.com/sinhasam/vaal/blob/master/arguments.py
        for count in range(args.num_vae_steps):
            recon, z, mu, logvar = models['vae'](labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta, mse_loss)
            unlab_recon, unlab_z, unlab_mu, unlab_logvar = models['vae'](unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta, mse_loss)
        
            labeled_preds = models['discriminator'](mu)
            unlabeled_preds = models['discriminator'](unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0)).reshape(-1, 1).cuda()
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0)).reshape(-1, 1).cuda()
            lab_real_preds = lab_real_preds.cuda()
            unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds, lab_real_preds) + \
                    bce_loss(unlabeled_preds, unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (args.num_vae_steps - 1):
                try:
                    labeled_imgs, _, _ = next(labeled_data)
                except:
                    labeled_data = iter(dataloaders['train'])
                    labeled_imgs, _, _ = next(labeled_data)
                try:
                    unlabeled_imgs, _, _ = next(unlabeled_data)
                except:
                    unlabeled_data = iter(dataloaders['unlabeled'])
                    unlabeled_imgs, _, _ = next(unlabeled_data)
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

        # Discriminator step
        for count in range(args.num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = models['vae'](labeled_imgs)
                _, _, unlab_mu, _ = models['vae'](unlabeled_imgs)
            
            labeled_preds = models['discriminator'](mu)
            unlabeled_preds = models['discriminator'](unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0)).reshape(-1, 1).cuda()
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0)).reshape(-1, 1).cuda()
            lab_real_preds = lab_real_preds.cuda()
            unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds, lab_real_preds) + \
                    bce_loss(unlabeled_preds, unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (args.num_adv_steps - 1):
                try:
                    labeled_imgs, _, _ = next(labeled_data)
                except:
                    labeled_data = iter(dataloaders['train'])
                    labeled_imgs, _, _ = next(labeled_data)
                try:
                    unlabeled_imgs, _, _ = next(unlabeled_data)
                except:
                    unlabeled_data = iter(dataloaders['unlabeled'])
                    unlabeled_imgs, _, _ = next(unlabeled_data)
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()


def train_epoch_for_VAAL_vgg(models, ce_loss, mse_loss, bce_loss, optimizers, dataloaders, lr_change, epoch, args):
    models['backbone'].train()
    models['vae'].train()
    models['discriminator'].train()
    labeled_data = iter(dataloaders['train'])
    unlabeled_data = iter(dataloaders['unlabeled'])
    
    train_iterations = len(dataloaders['train'])
    for iter_count in range(train_iterations):
        total_iter_count = train_iterations * epoch + iter_count
        if total_iter_count is not 0 and total_iter_count % lr_change == 0:
            for param in optimizers['backbone'].param_groups:
                param['lr'] = param['lr'] / 10
            
        try:
            labeled_imgs, labels, index = next(labeled_data)
        except:
            labeled_data = iter(dataloaders['train'])
            labeled_imgs, labels, index = next(labeled_data)
        try:
            unlabeled_imgs, _, _ = next(unlabeled_data)
        except:
            unlabeled_data = iter(dataloaders['unlabeled'])
            unlabeled_imgs, _, _ = next(unlabeled_data)
        unlabeled_imgs, labeled_imgs, labels = unlabeled_imgs.cuda(), labeled_imgs.cuda(), labels.cuda()

        optimizers['backbone'].zero_grad()
        
        # start training
        scores = models['backbone'](labeled_imgs)
        loss = ce_loss(scores, labels)
        loss.backward()
        optimizers['backbone'].step()

        # VAE step
        
        beta = 1 # https://github.com/sinhasam/vaal/blob/master/arguments.py
        adversary_param = 1 # https://github.com/sinhasam/vaal/blob/master/arguments.py
        for count in range(args.num_vae_steps):
            recon, z, mu, logvar = models['vae'](labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta, mse_loss)
            unlab_recon, unlab_z, unlab_mu, unlab_logvar = models['vae'](unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta, mse_loss)
        
            labeled_preds = models['discriminator'](mu)
            unlabeled_preds = models['discriminator'](unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0)).reshape(-1, 1).cuda()
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0)).reshape(-1, 1).cuda()
            lab_real_preds = lab_real_preds.cuda()
            unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds, lab_real_preds) + \
                    bce_loss(unlabeled_preds, unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (args.num_vae_steps - 1):
                try:
                    labeled_imgs, _, _ = next(labeled_data)
                except:
                    labeled_data = iter(dataloaders['train'])
                    labeled_imgs, _, _ = next(labeled_data)
                try:
                    unlabeled_imgs, _, _ = next(unlabeled_data)
                except:
                    unlabeled_data = iter(dataloaders['unlabeled'])
                    unlabeled_imgs, _, _ = next(unlabeled_data)
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

        # Discriminator step
        for count in range(args.num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = models['vae'](labeled_imgs)
                _, _, unlab_mu, _ = models['vae'](unlabeled_imgs)
            
            labeled_preds = models['discriminator'](mu)
            unlabeled_preds = models['discriminator'](unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0)).reshape(-1, 1).cuda()
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0)).reshape(-1, 1).cuda()
            lab_real_preds = lab_real_preds.cuda()
            unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds, lab_real_preds) + \
                    bce_loss(unlabeled_preds, unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (args.num_adv_steps - 1):
                try:
                    labeled_imgs, _, _ = next(labeled_data)
                except:
                    labeled_data = iter(dataloaders['train'])
                    labeled_imgs, _, _ = next(labeled_data)
                try:
                    unlabeled_imgs, _, _ = next(unlabeled_data)
                except:
                    unlabeled_data = iter(dataloaders['unlabeled'])
                    unlabeled_imgs, _, _ = next(unlabeled_data)
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()
## VAAL

## Mixup
def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mix_data(x, y, index=None, alpha=0.8, use_cuda=True, lam=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if lam == None:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
    batch_size = x.size()[0]
    if index == None:
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch_for_Mixup(models, criterion, optimizers, dataloaders, alpha=0.4):
    models['backbone'].train() 
    total_loss = 0

    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        # make the pair
        inputs, targets_a, targets_b, lam = mix_data(inputs, labels, alpha=alpha, use_cuda=True)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

        # start the model training
        optimizers['backbone'].zero_grad()
        scores = models['backbone'](inputs)
        loss = mixed_criterion(criterion, scores, targets_a, targets_b, lam)
        total_loss += loss.item()

        loss.backward()
        optimizers['backbone'].step()
## Mixup

## LL4AL 
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

## LL4AL 
def train_epoch_for_LL4AL(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()

    # for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'].forward_ll4al(inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=1.0)
  
        loss            = m_backbone_loss + 1.0 * m_module_loss
        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

## LL4AL with fixmatch 
def train_epoch_for_LL4AL_FixMatch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()

    # for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'].forward_ll4al(inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=1.0)
  
        loss            = m_backbone_loss + 1.0 * m_module_loss
        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()



def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    logits_ = torch.tensor([])
    labels_ = torch.tensor([])

    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloaders[mode]:
            inputs = data[0].cuda()
            labels = data[1].cuda()

            scores = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            logits_ = torch.cat((logits_, scores.cpu()), 0)
            labels_ = torch.cat((labels_, labels.cpu()), 0)

    return 100 * correct / total

def train(models, criterion, optimizers, schedulers, dataloaders, args):
    print('>> Train a Model.')
    for epoch in tqdm(range(args.epoch_num)):
        schedulers['backbone'].step()
        if args.approach in ['LL4AL', 'TA-VAAL']:
            schedulers['module'].step()

        if args.approach in ['OO4AL', 'CMaM']:
            train_epoch_for_CMaM(models, criterion, optimizers, dataloaders, alpha=args.alpha, layer=args.layer)
        elif args.approach in ['OO4AL_SELF']:
            train_epoch_for_LL4AL_FixMatch(models, criterion, optimizers, dataloaders, alpha=args.alpha, layer=args.layer)
        elif args.approach in ['LL4AL', 'TA-VAAL']:
            train_epoch_for_LL4AL(models, criterion, optimizers, dataloaders, epoch, args.epochl)
        elif args.approach in ['Mixup']:
            train_epoch_for_Mixup(models, criterion, optimizers, dataloaders, alpha=args.alpha)
        elif args.approach in ['VAAL']:
            # cross entropy loss : criterion
            bce_loss = nn.BCELoss()
            mse_loss = nn.MSELoss()
            train_epoch_for_VAAL(models, criterion, mse_loss, bce_loss, optimizers, dataloaders, args)
        else:
            train_epoch_forRandom(models, criterion, optimizers, dataloaders)      

    print('>> Finished.')
    

def train_vgg(models, criterion, optimizers, schedulers, dataloaders, args):
    print('>> Train a Model.')
    train_iterations = len(dataloaders['train']) * args.epoch_num
    lr_change = train_iterations // 4
    for epoch in tqdm(range(args.epoch_num)):
        if args.approach in ['VAAL']:
            # cross entropy loss : criterion
            bce_loss = nn.BCELoss()
            mse_loss = nn.MSELoss()
            train_epoch_for_VAAL_vgg(models, criterion, mse_loss, bce_loss, optimizers, dataloaders, lr_change, epoch, args)
        else:
            train_epoch_forRandom(models, criterion, optimizers, dataloaders)      

    print('>> Finished.')
    