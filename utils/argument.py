import argparse

def get_argument():
    ## parser
    parser = argparse.ArgumentParser(description='Overcoming Overconfidence for Active Learning')
    # argument for train data.
    parser.add_argument('--data_path', default="/data1/dataset/cifar10", type=str, help='the path for data')
    parser.add_argument('--dataset', default="cifar10", choices=['cifar10', 'cifar100', 'imagenet', 'cifar10im'], help='the name of training data(cifar10)')
    parser.add_argument('--class_num', default=10, type=int, help='the number of the target classes')
    # argument for active learning.
    parser.add_argument('--initial', default=1000, type=int, help='the number of initial labeled images for model training')
    parser.add_argument('--selected', default=1000, type=int, help='the number of selected unlabeled images each cycle') 
    parser.add_argument('--sampling_num', default=10000, type=int, help='the number of selected unlabeled image in a cycle')
    parser.add_argument('--cycle_num', default=10, type=int, help='the number of cycles for the active learning algorithm') 
    # argument for training.
    # parser.add_argument('--approach', default='OO4AL', type=str, \
    #                     choices=['OO4AL', 'rankedms', 'CMaM', 'margin', 'entropy', 'VAAL', 'OO4AL_SELF', 'OO4AL_Input', \
    #                             'Random', 'CoreSet', 'LL4AL', 'TA-VAAL', 'UncertainGCN', 'Mixup', 'entropy_no_softmax', 'Hierarchical'] )
    parser.add_argument('--approach', default='OO4AL', type=str)
    parser.add_argument('--seed', default=0, type=int, help='random seed to use')
    parser.add_argument('--save_name', default="jobs", type=str, help='The name of the results-saving path')
    parser.add_argument("--milestones", nargs="+", type=int, help="List of milestones")
    parser.add_argument('--epoch_num', default=200, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wdecay', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--arch', default='resnet18', type=str)
    # argument for ablation study.
    parser.add_argument("--layer", type=int, default=1, help="fusion layer location")
    parser.add_argument("--alpha", type=float, default=0.4, help="fusion parameter of beta distribution")
    # argument for other methods.
    parser.add_argument("-n","--hidden_units", type=int, default=128, help="Number of hidden units of the graph")
    parser.add_argument("-r","--dropout_rate", type=float, default=0.3, help="Dropout rate of the graph neural network")
    parser.add_argument("-l","--lambda_loss",type=float, default=1.2,  help="Adjustment graph loss parameter between the labeled and unlabeled")
    parser.add_argument("-s","--s_margin", type=float, default=0.1, help="Confidence margin of graph")
    parser.add_argument("--epochl", type=int, default=120, help="LL4AL parameter (the epoch to stop the training of the loss predict module) ")
    parser.add_argument("--lossnet_feature", nargs="+", type=int, help="List of feature size for loss prediction networks")
    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension') # VAAL
    parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
    # argument for darkdata
    parser.add_argument('--pretrained_weight', default="None", type=str, help='the path for pretrained weight')
    args = parser.parse_args()


    return args
