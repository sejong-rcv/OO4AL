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
from utils.hierarchical_sampling import *

def get_RankedMS(models, unlabeled_loader, class_num):
    models['backbone'].eval()
    gradient_scores = torch.tensor([])

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            scores = models['backbone'](inputs)
            sorted_scores, sorted_indexs = torch.sort(scores, dim=1, descending=True)
            sorted_scores = torch.nn.functional.softmax(sorted_scores)
            scores_gap = sorted_scores[:, :class_num-1] - sorted_scores[:, 1:] 
            position_map = torch.tensor(range(1, class_num)).expand_as(scores_gap)
            g_score = torch.sum(scores_gap.cpu() / position_map, dim=1)
            gradient_scores = torch.cat((gradient_scores, g_score), 0)

    return gradient_scores.cpu()

def get_margin(models, unlabeled_loader):
    models['backbone'].eval()
    margin_scores = torch.tensor([])

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            scores = models['backbone'](inputs)
            sorted_scores, sorted_indexs = torch.sort(scores, dim=1, descending=True)
            sorted_scores = torch.nn.functional.softmax(sorted_scores)
            scores_gap = sorted_scores[:, 0] - sorted_scores[:, 1]
            margin_scores = torch.cat((margin_scores, scores_gap.cpu()), 0)

    return margin_scores.cpu()

def get_entropy(models, unlabeled_loader, class_num):
    models['backbone'].eval()
    entropy_scores = []

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            scores = models['backbone'](inputs)
            scores = torch.nn.functional.softmax(scores)
            scores = scores.cpu()
            entropy = - np.nansum(scores * np.log(scores), axis=1)
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

def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle, args):
    
    vae = models['vae']
    discriminator = models['discriminator']
    task_model = models['backbone']
    ranker = models['module']
    
    task_model.eval()
    ranker.eval()
    vae.train()
    discriminator.train()
    vae = vae.cuda()
    discriminator = discriminator.cuda()
    task_model = task_model.cuda()
    ranker = ranker.cuda()
    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 1

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data_tavaal(labeled_dataloader)
    unlabeled_data = read_data_tavaal(unlabeled_dataloader)

    train_iterations = int( (args.selected*cycle+ args.sampling_num) * EPOCHV / BATCH )

    for iter_count in tqdm.tqdm(range(train_iterations)):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        labeled_imgs = labeled_imgs.cuda()
        unlabeled_imgs = unlabeled_imgs.cuda()
        labels = labels.cuda()
        if iter_count == 0 :
            r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0],1))).type(torch.FloatTensor).cuda()
            r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0],1))).type(torch.FloatTensor).cuda()
        else:
            with torch.no_grad():
                _,_,features_l = task_model.forward_feature_ll4al(labeled_imgs)
                _,_,feature_u = task_model.forward_feature_ll4al(unlabeled_imgs)
                r_l = ranker(features_l)
                r_u = ranker(feature_u)
        if iter_count == 0:
            r_l = r_l_0.detach()
            r_u = r_u_0.detach()
            r_l_s = r_l_0.detach()
            r_u_s = r_u_0.detach()
        else:
            r_l_s = torch.sigmoid(r_l).detach()
            r_u_s = torch.sigmoid(r_u).detach()                 
        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(r_l_s,labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s,unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            lab_real_preds = lab_real_preds.cuda()
            unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(r_l_s,labeled_imgs)
                _, _, unlab_mu, _ = vae(r_u_s,unlabeled_imgs)
            
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))
            lab_real_preds = lab_real_preds.cuda()
            unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()
            if iter_count % 100 == 0:
                print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))
                
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

def get_VAALScore(models, unlabeled_loader):
    all_preds = []

    for images, _, indices in unlabeled_loader:
        images = images.cuda()

        with torch.no_grad():
            _, _, mu, _ = models['vae'](images)
            preds = models['discriminator'](mu)

        preds = preds.cpu().data
        all_preds.extend(preds)

    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)
    # need to multiply by -1 to be able to use torch.topk 
    all_preds *= -1

    # import pdb;pdb.set_trace()
    # _, querry_indices = torch.topk(all_preds, int(self.budget))

    return all_preds

def get_HierarchicalScore(models, unlabeled_loader, data_unlabeled, subset, labeled_set):
    models['backbone'].eval()
    features = torch.tensor([]).cuda()
    
    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            inputs = inputs.cuda()
            _, features_batch = models['backbone'].forward_feature(inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
    
    targets = np.array(data_unlabeled.targets)[subset+labeled_set]
    labeled_targets = np.array(data_unlabeled.targets)[labeled_set]
    classes = list(set(targets))
    # Masking Unlabeled Dataset (to None)
    masked_targets = [None] * len(subset) + list(labeled_targets)
    
    ds = Dataset(feat, masked_targets) ## Labeled set    
    qs = HierarchicalSampling(ds, classes, active_selecting=True, random_state=1126)
    feat_idx = run_qs(ds, qs, targets, 1000)
    return feat_idx


# Select the indices of the unlablled data according to the methods
def query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args, num_labeled):
    flag = 0
    if method == 'Random' or method == 'Mixup' or method == 'CMaM' or method == 'TS':
        arg = range(args.sampling_num)
    
    if method == 'OO4AL' or method == 'rankedms' or method == 'OO4AL_SELF' or method == 'OO4AL_Input':
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

    if method == 'UncertainGCN':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    num_workers=args.workers,
                                    pin_memory=True)
        binary_labels = torch.cat((torch.zeros([args.sampling_num, 1]),(torch.ones([len(labeled_set),1]))),0)

        features = get_features(model, unlabeled_loader)
        features = nn.functional.normalize(features)
        adj = aff_to_adj(features)

        gcn_module = GCN(nfeat=features.shape[1],
                         nhid=args.hidden_units,
                         nclass=1,
                         dropout=args.dropout_rate).cuda()
                                
        models      = {'gcn_module': gcn_module}

        optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=LR_GCN, weight_decay=WDECAY)
        optimizers = {'gcn_module': optim_backbone}

        lbl = np.arange(args.sampling_num, args.sampling_num+(cycle+1)*args.selected, 1)
        nlbl = np.arange(0, args.sampling_num, 1)
        
        ############
        for _ in range(200):

            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](features, adj)
            lamda = args.lambda_loss 
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers['gcn_module'].step()


        models['gcn_module'].eval()
        with torch.no_grad():
            inputs = features.cuda()
            labels = binary_labels.cuda()
            scores, _, feat = models['gcn_module'](inputs, adj)
            s_margin = args.s_margin 
            scores_median = np.squeeze(torch.abs(scores[:args.sampling_num] - s_margin).detach().cpu().numpy())
            arg = np.argsort(-(scores_median))

            print("Max confidence value: ",torch.max(scores.data))
            print("Mean confidence value: ",torch.mean(scores.data))
            preds = torch.round(scores)
            correct_labeled = (preds[args.sampling_num:,0] == labels[args.sampling_num:,0]).sum().item() / ((cycle+1)*args.selected)
            correct_unlabeled = (preds[:args.sampling_num,0] == labels[:args.sampling_num,0]).sum().item() / args.sampling_num
            correct = (preds[:,0] == labels[:,0]).sum().item() / (args.sampling_num + (cycle+1)*args.selected)
            print("Labeled classified: ", correct_labeled)
            print("Unlabeled classified: ", correct_unlabeled)
            print("Total classified: ", correct) 
    
    if method == 'CoreSet':
        # Create dataloader
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    num_workers=args.workers,
                                    pin_memory=True)

        arg = get_kcg(model, num_labeled, unlabeled_loader, args)

    if method == 'LL4AL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    num_workers=args.workers,
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader)
        arg = np.argsort(uncertainty)        

    if method == 'TA-VAAL':
        data_unlabeled.tavaal = True
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    num_workers=args.workers,
                                    pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    num_workers=args.workers,
                                    pin_memory=True)
        vae = VAE()
        discriminator = Discriminator(32)
     
        models      = {'backbone': model['backbone'], 'module': model['module'],'vae': vae, 'discriminator': discriminator}
        
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

        train_vaal(models,optimizers, labeled_loader, unlabeled_loader, cycle+1, args)
        task_model = models['backbone']
        ranker = models['module']        
        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:                       
            images = images.cuda()
            with torch.no_grad():
                _,_,features = task_model.forward_feature_ll4al(images)
                r = ranker(features)
                _, _, mu, _ = vae(torch.sigmoid(r),images)
                preds = discriminator(r,mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds) 

    if method == 'VAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    num_workers=args.workers,
                                    pin_memory=True)
        # Measure uncertainty of each data points in the subset
        all_preds = get_VAALScore(model, unlabeled_loader)
        arg = np.argsort(all_preds)
        flag = - 1

    if method == 'Hierarchical':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), 
                                    num_workers=args.workers,
                                    pin_memory=True)
        # Measure uncertainty of each data points in the subset
        arg = get_HierarchicalScore(model, unlabeled_loader, data_unlabeled, subset, labeled_set)
        flag = 2

    return arg, flag