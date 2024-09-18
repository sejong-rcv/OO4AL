'''
"Overcoming Overconfidence for Active Learning" Procedure in PyTorch.
'''
# Torch
import torch
from torch.utils.data import DataLoader

# Utils
import argparse
import statistics
from glob import glob

# Custom
from data.get_dataloader import *
import models.resnet_fusion as resnet


## parser
parser = argparse.ArgumentParser(description='Evaluation code for Overcoming Overconfidence for Active Learning')
parser.add_argument('--data_path', default="cifar10", type=str, help='the path for training data')
parser.add_argument('--dataset', default="cifar10", type=str, help='the name of training data(cifar10)')
parser.add_argument('--class_num', default=10, type=int, help='the number of the target classes')
parser.add_argument('--checkpoint', default="jobs/cifar10", type=str, help='a checkpoint for evaluation')
parser.add_argument('--approach', default='OO4AL', type=str, \
                    choices=['OO4AL', 'rankedms', 'CMaM', 'margin', 'entropy', \
                            'Random', 'CoreSet', 'LL4AL', 'TA-VAAL', 'UncertainGCN', 'Mixup'] )
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--num_of_cycle', default=10, type=int)
args = parser.parse_args()


def compute_calibration_metrics(num_bins=100, net=None, loader=None, device='cuda'):
    """
    Computes the calibration metrics ECE and OE along with the acc and conf values
    :param num_bins: Taken from email correspondence and 100 is used
    :param net: trained network
    :param loader: dataloader for the dataset
    :param device: cuda or cpu
    :return: ECE, OE, acc, conf
    """
    acc_counts = [0 for _ in range(num_bins+1)]
    conf_counts = [0 for _ in range(num_bins+1)]
    overall_conf = []
    n = float(len(loader.dataset))
    counts = [0 for i in range(num_bins+1)]
    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            images = data[0].to(device)
            labels = data[1].to(device)
            outputs = net(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confs, preds = probabilities.max(1)
            for (conf, pred, label) in zip(confs, preds, labels):
                bin_index = int(((conf * 100) // (100/num_bins)).cpu())
                try:
                    if pred == label:
                        acc_counts[bin_index] += 1.0
                    conf_counts[bin_index] += conf.cpu()
                    counts[bin_index] += 1.0
                    overall_conf.append(conf.cpu())
                except:
                    print(bin_index, conf)
                    raise AssertionError('Bin index out of range!')


    avg_acc = [0 if count == 0 else acc_count / count for acc_count, count in zip(acc_counts, counts)]
    avg_conf = [0 if count == 0 else conf_count / count for conf_count, count in zip(conf_counts, counts)]
    ECE, OE = 0, 0
    for i in range(num_bins):
        ECE += (counts[i] / n) * abs(avg_acc[i] - avg_conf[i])
        OE += (counts[i] / n) * (avg_conf[i] * (max(avg_conf[i] - avg_acc[i], 0)))

    return ECE, OE, avg_acc, avg_conf, sum(acc_counts) / n, counts

# Main
if __name__ == '__main__':
    print(f"\n\n\n {args.checkpoint} \n")
    _, _, test_custom = get_dataloader(args.dataset, args)
    test_loader  = DataLoader(test_custom, batch_size=args.batch_size, num_workers=args.workers)

    if args.dataset=="cifar10":
        prefix = "1e-3"
    elif args.dataset=="cifar100":
        prefix = "1e-2"
    elif args.dataset=="cifar10im":
        prefix = "1e-2"
    elif args.dataset=="imagenet":
        prefix = "1e-4"
    else:
        raise ValueError("Not Support!")
    # Model
    resnet18    = resnet.ResNet18(num_classes=args.class_num).cuda()
    torch.backends.cudnn.benchmark = False
    checkpoint_format = args.checkpoint
    for cycle in range(args.num_of_cycle):
        acc_list = []
        oe_list = []
        ece_list = []
        checkpoint_paths = glob(checkpoint_format.format('*', cycle))
        checkpoint_paths = sorted(checkpoint_paths)
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path)
            try:
                resnet18.load_state_dict(checkpoint['state_dict_backbone'])
            except:
                import pdb;pdb.set_trace()
                pass
            accuracy = []
            winning_score = []    
            ece, oe, bin_acc, bin_conf, acc, bin_count = compute_calibration_metrics(num_bins=100, net=resnet18, loader=test_loader, device='cuda')
            for i, c in enumerate(bin_count):
                if c > 0:
                    for _ in range(int(c)):
                        accuracy.append(bin_acc[i])
                        winning_score.append(bin_conf[i])
            acc_list.append(round(acc*100, 2))
            oe_list.append(oe.item()/float(prefix))
        print('Cycle: {} | ACC(%): {:.1f}±{:.1f} | OE({}): {:.1f}±{:.1f}'.format(cycle, statistics.mean(acc_list), statistics.stdev(acc_list), prefix, \
                                                                            statistics.mean(oe_list), statistics.stdev(oe_list)))