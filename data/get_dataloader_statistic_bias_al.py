import os
from torchvision import datasets
import torchvision.transforms as T

from data.cifar_statistic_bias import CIFAR10, CIFAR10im
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


class ActiveLearningData:
    def __init__(self, dataset, active_learning_hypers=None):
        super().__init__()
        active_learning_hypers = {
            "num_to_acquire": 1,  # Number of points acquired each time we do acquisition
            "starting_points": 10,  # Number of points we start with (randomly selected)
            "balanced_start": False,  # Ensure there's an equal number from each class at initialization
            "scoring_variational_samples": 100,  # Number of samples from posterior to take when we acquire
            "score_strategy": "mutual_information",  # May be "mutual_information", "entropy", or "random_acquisition"
            "weighting_scheme": "refined",  # May be "none", "naive", or "refined"
            "warm_start": False,  # Use the last trained model to initialize the new training loop
            "mi_plotting": True,  # Plot the MI's of the pool for diagnostic purposes
            "proposal": "softmax",  # Can be "softmax" or "proportional" for how we sample w.r.t. the scores
            "temperature": 15000  # Temperature of the softmax distribution if being used
        }
        self.dataset = dataset
        self.total_num_points = len(self.dataset)
        self.num_initial_points = active_learning_hypers["starting_points"]
        self.weighting_scheme = active_learning_hypers["weighting_scheme"]
        self.active_learning_hypers = active_learning_hypers

        # At the beginning, we have acquired no points
        # the acquisition mask is w.r.t. the training data (not validation)
        self.acquisition_mask = np.full((len(dataset),), False)
        self.num_acquired_points = 0

        self.active_dataset = Subset(self.dataset, None)
        self.available_dataset = Subset(self.dataset, None)

        # # Now we randomly select num_initial_points uniformly
        self.num_acquired_points = self.num_initial_points
        self._update_indices()

        for initial_idx in range(self.num_initial_points):
            scores = torch.ones(len(self.available_dataset))
            self.acquire_points_and_update_weights(scores, self.active_learning_hypers)

    def _update_indices(self):
        self.active_dataset.indices = np.where(self.acquisition_mask)[0]
        self.available_dataset.indices = np.where(~self.acquisition_mask)[0]

    def _update_weights(self, _run):
        if self.weighting_scheme == "none":
            pass
        elif self.weighting_scheme == "refined":
            self._update_refined_weight_scheme(_run)
        elif self.weighting_scheme == "naive":
            self._update_naive(_run)
        else:
            raise NotImplementedError

    def acquire_points_and_update_weights(
        self, scores, active_learning_hypers, _run=None, logging=None
    ):
        probability_masses = scores / torch.sum(scores)
        proposal = active_learning_hypers["proposal"]
        if proposal == "proportional":
            idxs, masses = sample_proportionally(
                probability_masses, active_learning_hypers
            )
        elif proposal == "softmax":
            idxs, masses = sample_softmax(probability_masses, active_learning_hypers)
        else:
            raise NotImplementedError

        # This index is on the set of points in "available_dataset"
        # This maps onto an index in the train_dataset (as opposed to valid)
        train_idxs = get_subset_base_indices(self.available_dataset, idxs)
        # Then that maps onto the index in the original union of train and validation
        true_idxs = get_subset_base_indices(
            self.dataset, train_idxs
        )  # These are the 'canonical' indices to add to the acquired points

        self.dataset.dataset.proposal_mass[true_idxs] = masses
        self.dataset.dataset.sample_order[true_idxs] = int(
            torch.max(self.dataset.dataset.sample_order) + 1
        )
        self.acquisition_mask[train_idxs] = True
        self._update_weights(_run)

        if _run is not None:
            if logging is not None:
                if logging["images"]:
                    # save the mnist image to a jpg
                    acquired_pixels = self.dataset.dataset.data[true_idxs]
                    Image.fromarray(acquired_pixels[0].numpy(), "L").save(
                        "tmp/temp.jpg"
                    )
                    _run.add_artifact(
                        "tmp/temp.jpg", f"{len(self.active_dataset.indices)}.jpg"
                    )
                if logging["classes"]:
                    _run.log_scalar(
                        "acquired_class", f"{self.dataset.dataset.targets[true_idxs]}"
                    )
                    #print(f"Picked: {self.dataset.dataset.targets[true_idxs]}")
                    num_acquired_points = len(self.dataset.dataset.targets[self.dataset.dataset.sample_order > 0])
                    class_distribution = [torch.sum(c==self.dataset.dataset.targets[self.dataset.dataset.sample_order > 0], dtype=torch.float32) / num_acquired_points for c in range(0,10)]
                    print(f"Classes: {class_distribution}")

        self._update_indices()

    def _update_refined_weight_scheme(self, _run):
        """
        This does the work for the method known as R_lure"""
        N = len(self.active_dataset) + len(
            self.available_dataset
        )  # Note that self.dataset.datset includes validation, which should not be in N!
        M = torch.sum(self.dataset.dataset.sample_order > 0)
        active_idxs = self.dataset.dataset.sample_order > 0
        m = self.dataset.dataset.sample_order[active_idxs]
        q = self.dataset.dataset.proposal_mass[active_idxs]

        weight = (N - m + 1) * q
        weight = 1 / weight - 1
        weight = (N - M) * weight
        weight = weight / (N - m)
        weight = weight + 1
        #print(f"New weight {weight.cpu().numpy()[m.numpy().argsort()]}")
        self.dataset.dataset.weights[active_idxs] = weight.float()
        self.log_weights(_run, weight, m, M)

    def _update_naive(self, _run):
        """
        This does the work for the method known as R_pure"""
        N = len(self.active_dataset) + len(
            self.available_dataset
        )  # Note that self.dataset.datset includes validation, which should not be in N!
        M = torch.sum(self.dataset.dataset.sample_order > 0)
        active_idxs = self.dataset.dataset.sample_order > 0
        m = self.dataset.dataset.sample_order[active_idxs]
        q = self.dataset.dataset.proposal_mass[active_idxs]

        weight = (1 / q) + M - m
        weight = weight / N
        #print(f"New weight {weight.cpu().numpy()[m.numpy().argsort()]}")
        self.dataset.dataset.weights[active_idxs] = weight.float()
        self.log_weights(_run, weight, m, M)


    def log_weights(self, _run, weight, m, M):
        # Lets log the new weights for the datapoints:
        if _run is not None:
            if "weights" not in _run.info:
                _run.info["weights"] = collections.OrderedDict()
            M = str(M.numpy())
            ordering = m.numpy().argsort()
            _run.info["weights"][M] = weight.numpy()[ordering].tolist()
            

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



    return train_custom, ActiveLearningData(unlabeled_custom), test_custom
