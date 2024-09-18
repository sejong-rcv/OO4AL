import sys

import random
import numpy as np
import scipy.sparse as sp


from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

from sklearn.utils import shuffle

from six import with_metaclass
from abc import ABCMeta, abstractmethod

"""
libact/utils/__init__.py
"""
def inherit_docstring_from(cls):
    """Decorator for class methods to inherit docstring from :code:`cls`
    """
    # https://groups.google.com/forum/#!msg/comp.lang.python/HkB1uhDcvdk/lWzWtPy09yYJ
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls, fn.__name__).__doc__
        return fn
    return docstring_inheriting_decorator


def seed_random_state(seed):
    """Turn seed into np.random.RandomState instance
    """
    if (seed is None) or (isinstance(seed, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r can not be used to generate numpy.random.RandomState"
                     " instance" % seed)


"""
libact/base/interfaces.py
    Base interfaces for use in the package.
    The package works according to the interfaces defined below.
"""

class QueryStrategy(with_metaclass(ABCMeta, object)):

    """Pool-based query strategy

    A QueryStrategy advices on which unlabeled data to be queried next given
    a pool of labeled and unlabeled data.
    """

    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        dataset.on_update(self.update)

    @property
    def dataset(self):
        """The Dataset object that is associated with this QueryStrategy."""
        return self._dataset

    def update(self, entry_id, label):
        """Update the internal states of the QueryStrategy after each queried
        sample being labeled.

        Parameters
        ----------
        entry_id : int
            The index of the newly labeled sample.

        label : float
            The label of the queried sample.
        """
        pass

    def _get_scores(self):
        """Return the score used for making query, the larger the better. Read-only.

        No modification to the internal states.

        Returns
        -------
        (ask_id, scores): list of tuple (int, float)
            The index of the next unlabeled sample to be queried and the score assigned.
        """
        pass

    @abstractmethod
    def make_query(self):
        """Return the index of the sample to be queried and labeled. Read-only.

        No modification to the internal states.

        Returns
        -------
        ask_id : int
            The index of the next unlabeled sample to be queried and labeled.
        """
        pass




""" 
libact/query_strategies/multiclass/hierarchical_sampling.py
    Hierarchical Sampling for Active Learning (HS)

    This module contains a class that implements Hierarchical Sampling for Active
    Learning (HS).
"""

NO_NODE = -1
NO_LABEL = -1


class HierarchicalSampling(QueryStrategy):

    """Hierarchical Sampling for Active Learning (HS)

    HS is an active learning scheme that exploits cluster structure in data.
    The original C++ implementation by the authors can be found at:
    http://www.cs.columbia.edu/~djhsu/code/HS.tar.gz

    Parameters
    ----------
    classes: list
        List of distinct classes in data.

    active_selecting: {True, False}, optional (default=True)
        False (random selecting): sample weight of a pruning is its number of
        unseen leaves.
        True (active selecting): sample weight of a pruning is its weighted
        error bound.

    subsample_qs: {:py:class:`libact.base.interfaces.query_strategies`, None}, optional (default=None)
        Subsample query strategy used to sample a node in the selected pruning.
        RandomSampling is used if None.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    m : int
        number of nodes

    classes: list
        List of distinct classes in data.

    n : int
        number of leaf nodes

    num_class : int
        number of classes

    parent : np.array instance, shape = (m)
        parent indices

    left_child : np.array instance, shape = (m)
        left child indices

    right_child : np.array instance, shape = (m)
        right child indices

    size : np.array instance, shape = (m)
        number of leaves in subtree

    depth : np.array instance, shape = (m)
        maximum depth in subtree

    count : np.array instance, shape = (m, num_class)
        node class label counts

    total : np.array instance, shape = (m)
        total node class labels seen (total[i] = Sum_j count[i][j])

    lower_bound : np.array instance, shape = (m, num_class)
        upper bounds on true node class label counts

    upper_bound : np.array instance, shape = (m, num_class)
        lower bounds on true node class label counts

    admissible: np.array instance, shape = (m, num_class)
        flag indicating if (node,label) is admissible

    best_label: np.array instance, shape = (m)
        best admissible label

    random_states\_ : np.random.RandomState instance
        The random number generator using.


    Examples
    --------
    Here is an example of declaring a HierarchicalSampling query_strategy
    object:

    .. code-block:: python

       from libact.query_strategies import UncertaintySampling
       from libact.query_strategies.multiclass import HierarchicalSampling

       sub_qs = UncertaintySampling(
           dataset, method='sm', model=SVM(decision_function_shape='ovr'))

       qs = HierarchicalSampling(
                dataset, # Dataset object
                dataset.get_num_of_labels(),
                active_selecting=True,
                subsample_qs=sub_qs
            )


    References
    ----------

    .. [1] Sanjoy Dasgupta and Daniel Hsu. "Hierarchical sampling for active
           learning." ICML 2008.
    """

    def __init__(self, dataset, classes, active_selecting=True,
            subsample_qs=None, random_state=None):
        super(HierarchicalSampling, self).__init__(dataset)
        
        # qs = HierarchicalSampling(ds, classes, active_selecting=True, random_state=1126)
        X, _ = self.dataset.get_entries() #
        cluster = AgglomerativeClustering()
        cluster.fit(X)
        childrens = cluster.children_

        if subsample_qs is not None:
            if not isinstance(subsample_qs, QueryStrategy):
                raise TypeError("subsample_qs has to be a QueryStrategy")
            self.sub_qs = subsample_qs
        else:
            self.sub_qs = None

        self.active_selecting = active_selecting
        self.random_state_ = seed_random_state(random_state)
        self.n = len(childrens) + 1
        self.m = self.n * 2 - 1
        self.num_class = len(classes)
        self.classes = list(classes)
        self.class_id = dict(zip(self.classes, range(self.num_class)))

        self.parent = np.full(self.m, NO_NODE, dtype=int)
        self.size = np.zeros(self.m, dtype=int)
        self.depth = np.zeros(self.m, dtype=int)
        for i, (left_child, right_child) in enumerate(childrens):
            parent = i + self.n
            self.parent[left_child] = parent
            self.parent[right_child] = parent
        self.left_child = np.concatenate([np.full(self.n, NO_NODE), childrens[:,0]]).astype(int)
        self.right_child = np.concatenate([np.full(self.n, NO_NODE), childrens[:,1]]).astype(int)

        for i in range(self.n):
            node = i
            cur_depth = 0
            while node != NO_NODE:
                assert node >= 0 and node < self.m
                self.size[node] += 1
                self.depth[node] = max(self.depth[node], cur_depth)
                cur_depth += 1
                node = self.parent[node]

        self.count = np.zeros((self.m, self.num_class), dtype=int)
        self.total = np.zeros(self.m, dtype=int)
        self.upper_bound = np.ones((self.m, self.num_class), dtype=float)
        self.lower_bound = np.zeros((self.m, self.num_class), dtype=float)
        self.admissible = np.zeros((self.m, self.num_class), dtype=bool)
        self.best_label = np.full(self.m, NO_LABEL, dtype=int)
        self.split = np.zeros(self.m, dtype=bool)
        self.cost = self.size.copy()

        self.prunings = [self.m-1]

        for i, entry in enumerate(self.dataset.data):
            if entry[1] != None:
                self.update(i, entry[1])
        
    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        if label not in self.class_id:
            raise ValueError(
                    'Unknown class of entry %d: %s, expected: %s' %
                    (entry_id, label, list(self.class_id.keys()))
                    )
        class_id = self.class_id[label]
        root_pruning = self._find_root_pruning(entry_id)
        self._update(entry_id, class_id, root_pruning)
        self._prune_node(root_pruning)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        pruning = self._select_pruning()
        if self.sub_qs is None:
            ask_id = int(self._sample_node(pruning))
        else:
            _, scores = self.sub_qs.make_query(return_score=True)
            leaves = set(self._find_leaves(pruning))
            leaf_scores = [(score, node) for node, score in scores if node in leaves]
            ask_id = max(leaf_scores)[1]
        return ask_id

    def report_entry_label(self, entry_id):
        """
        Return the best label of the asked entry.

        Parameters
        ----------
        entry_id : int
            The index of the sample to ask.

        Returns
        -------
        label: object
            The best label of the given sample.
        """

        pruning = self._find_root_pruning(entry_id)
        return self.classes[self._best_label(pruning)]

    def report_all_label(self):
        """
        Return the best label of the asked entry.

        Parameters
        ----------

        Returns
        -------
        labels: list of object, shape=(m)
            The best label of all samples.
        """

        labels = np.empty(len(self.dataset), dtype=int)
        for pruning in self.prunings:
            best_label = self._best_label(pruning)
            leaves = self._find_leaves(pruning)
            labels[leaves] = best_label
        return labels

    def _best_label(self, pruning):
        if self.best_label[pruning] != NO_LABEL:
            return self.best_label[pruning]
        if self.parent[pruning] != NO_NODE:
            return self.best_label[self.parent[pruning]]
        return 0 # default label is 0 if no admissble label for root

    def _find_root_pruning(self, entry_id):
        node = entry_id
        while node != NO_NODE and node not in self.prunings:
            node = self.parent[node]
        return node

    def _find_leaves(self, node):
        if node == NO_NODE:
            return []
        if self.size[node] == 1:
            return [node]
        return (self._find_leaves(self.left_child[node]) +
                self._find_leaves(self.right_child[node]))

    def _select_pruning(self):
        if self.active_selecting:
            sample_weight = []
            for pruning in self.prunings:
                best_label = self.best_label[pruning]
                if best_label == NO_LABEL:
                    w = self.size[pruning]
                else:
                    w = self.size[pruning] - self.lower_bound[pruning][best_label]
                sample_weight.append(w)
        else:
            sample_weight = self.size[self.prunings] - self.total[self.prunings]
        sample_weight = sample_weight / sum(sample_weight)
        return self.random_state_.choice(self.prunings, p=sample_weight)

    def _sample_node(self, node):
        num_unseen_leaves = self.size[node] - self.total[node]
        if num_unseen_leaves == 0:
            return NO_NODE
        if self.size[node] == 1:
            return node
        assert self.left_child[node] != NO_NODE and self.right_child[node] != NO_NODE
        p_left = (self.size[self.left_child[node]] - self.total[self.left_child[node]]) / num_unseen_leaves
        if self.random_state_.rand() < p_left:
            return self._sample_node(self.left_child[node])
        else:
            return self._sample_node(self.right_child[node])

    def _update(self, entry_id, label, root_pruning):
        node = entry_id
        while node != NO_NODE:
            self.count[node, label] += 1
            self.total[node] += 1
            assert self.total[node] <= self.size[node]

            for l in range(self.num_class):
                frac = self.count[node, l] / self.total[node]
                delta = self._get_delta(frac, node)
                mean = frac * self.size[node]
                err = delta * self.size[node]
                self.lower_bound[node][l] = max(self.count[node][l], mean - err)
                self.upper_bound[node][l] = min(self.size[node] - (self.total[node] - self.count[node, l]), mean + err)

            max_count = 0
            for l in range(self.num_class):
                self.admissible[node, l] = True
                for k in range(self.num_class):
                    if l != k and self.lower_bound[node, l] <= 2 * self.upper_bound[node, k] - self.size[node]:
                        self.admissible[node, l] = False
                if self.admissible[node, l] and self.count[node, l] > max_count:
                    max_count = self.count[node, l]
                    self.best_label[node] = l

            if self.best_label[node] != NO_LABEL:
                basic_cost = self.size[node] - self.lower_bound[node][self.best_label[node]]
            else:
                basic_cost = self.size[node]

            if self.size[node] == 1:
                self.cost[node] = basic_cost
            else:
                split_cost = self.cost[self.left_child[node]] + self.cost[self.right_child[node]]
                if split_cost < basic_cost and self.best_label[node] != NO_LABEL:
                    self.cost[node] = split_cost
                    self.split[node] = True
                else:
                    self.cost[node] = basic_cost

            if node != root_pruning:
                node = self.parent[node]
            else:
                break

    def _prune_node(self, root_pruning):
        self.prunings.remove(root_pruning)
        node_set = [root_pruning]
        while len(node_set) > 0:
            node = node_set.pop()
            if self.split[node]:
                node_set.append(self.left_child[node])
                node_set.append(self.right_child[node])
            else:
                self.prunings.append(node)

    def _get_delta(self, frac, node):
        fs_corr = 1.0 - self.total[node] / self.size[node]
        return fs_corr / self.total[node] + \
               np.sqrt(fs_corr * frac * (1. - frac) / self.total[node])


"""
libact/base/dataset.py
    The dataset class used in this package.
    Datasets consists of data used for training, represented by a list of
    (feature, label) tuples.
    May be exported in different formats for application on other libraries.
"""
class Dataset(object):

    """libact dataset object

    Parameters
    ----------
    X : {array-like}, shape = (n_samples, n_features)
        Feature of sample set.

    y : list of {int, None}, shape = (n_samples)
        The ground truth (label) for corresponding sample. Unlabeled data
        should be given a label None.

    Attributes
    ----------
    data : list, shape = (n_samples)
        List of all sample feature and label tuple.

    """

    def __init__(self, X=None, y=None):
        if X is None: X = np.array([])
        elif not isinstance(X, sp.csr_matrix):
            X = np.array(X)

        if y is None: y = []
        y = np.array(y)

        self._X = X
        self._y = y
        self.modified = True
        self._update_callback = set()

    def __len__(self):
        """
        Number of all sample entries in this object.

        Returns
        -------
        n_samples : int
        """
        return self._X.shape[0]

    def __getitem__(self, idx):
        # still provide the interface to direct access the data by index
        return self._X[idx], self._y[idx]

    @property
    def data(self): return self

    def get_labeled_mask(self):
        """
        Get the mask of labeled entries.

        Returns
        -------
        mask: numpy array of bool, shape = (n_sample, )
        """
        return ~np.fromiter((e is None for e in self._y), dtype=bool)

    def len_labeled(self):
        """
        Number of labeled data entries in this object.

        Returns
        -------
        n_samples : int
        """
        return self.get_labeled_mask().sum()

    def len_unlabeled(self):
        """
        Number of unlabeled data entries in this object.

        Returns
        -------
        n_samples : int
        """
        return (~self.get_labeled_mask()).sum()

    def get_num_of_labels(self):
        """
        Number of distinct lebels in this object.

        Returns
        -------
        n_labels : int
        """
        return np.unique(self._y[self.get_labeled_mask()]).size

    def append(self, feature, label=None):
        """
        Add a (feature, label) entry into the dataset.
        A None label indicates an unlabeled entry.

        Parameters
        ----------
        feature : {array-like}, shape = (n_features)
            Feature of the sample to append to dataset.

        label : {int, None}
            Label of the sample to append to dataset. None if unlabeled.

        Returns
        -------
        entry_id : {int}
            entry_id for the appened sample.
        """
        if isinstance(self._X, np.ndarray):
            self._X = np.vstack([self._X, feature])
        else: # sp.csr_matrix
            self._X = sp.vstack([self._X, feature])
        self._y = np.append(self._y, label)

        self.modified = True
        return len(self) - 1

    def update(self, entry_id, new_label):
        """
        Updates an entry with entry_id with the given label

        Parameters
        ----------
        entry_id : int
            entry id of the sample to update.

        label : {int, None}
            Label of the sample to be update.
        """
        self._y[entry_id] = new_label
        self.modified = True
        for callback in self._update_callback:
            callback(entry_id, new_label)

    def on_update(self, callback):
        """
        Add callback function to call when dataset updated.

        Parameters
        ----------
        callback : callable
            The function to be called when dataset is updated.
        """
        self._update_callback.add(callback)

    def format_sklearn(self):
        """
        Returns dataset in (X, y) format for use in scikit-learn.
        Unlabeled entries are ignored.

        Returns
        -------
        X : numpy array, shape = (n_samples, n_features)
            Sample feature set.

        y : numpy array, shape = (n_samples)
            Sample labels.
        """
        # becomes the same as get_labled_entries
        X, y = self.get_labeled_entries()
        return X, np.array(y)

    def get_entries(self):
        """
        Return the list of all sample feature and ground truth tuple.

        Returns
        -------
        X: numpy array or scipy matrix, shape = ( n_sample, n_features )
        y: numpy array, shape = (n_samples)
        """
        return self._X, self._y

    def get_labeled_entries(self):
        """
        Returns list of labeled feature and their label

        Returns
        -------
        X: numpy array or scipy matrix, shape = ( n_sample labeled, n_features )
        y: list, shape = (n_samples lebaled)
        """
        return self._X[self.get_labeled_mask()], self._y[self.get_labeled_mask()].tolist()

    def get_unlabeled_entries(self):
        """
        Returns list of unlabeled features, along with their entry_ids

        Returns
        -------
        idx: numpy array, shape = (n_samples unlebaled)
        X: numpy array or scipy matrix, shape = ( n_sample unlabeled, n_features )
        """
        return np.where(~self.get_labeled_mask())[0], self._X[~self.get_labeled_mask()]

    def labeled_uniform_sample(self, sample_size, replace=True):
        """Returns a Dataset object with labeled data only, which is
        resampled uniformly with given sample size.
        Parameter `replace` decides whether sampling with replacement or not.

        Parameters
        ----------
        sample_size
        """
        idx = np.random.choice(np.where(self.get_labeled_mask())[0],
                               size=sample_size, replace=replace )
        return Dataset(self._X[idx], self._y[idx])


"""
libact/query_strategies/tests/utils.py
    This module includes some functions to be reused in query strategy testing.
"""
def run_qs(trn_ds, qs, truth, quota):
    """Run query strategy on specified dataset and return quering sequence.

    Parameters
    ----------
    trn_ds : Dataset object
        The dataset to be run on.

    qs : QueryStrategy instance
        The active learning algorith to be run.

    truth : array-like
        The true label.

    quota : int
        Number of iterations to run

    Returns
    -------
    qseq : numpy array, shape (quota,)
        The numpy array of entry_id representing querying sequence.
    """
    ret = []
    for _ in range(quota):
        ask_id = qs.make_query()
        trn_ds.update(ask_id, truth[ask_id])

        ret.append(ask_id)
    return np.array(ret)
