import numpy as np
import torch

from .learning import LearnSPN, fit
from .nodes import (eval_root, eval_root_children, eval_root_class, sample,
                    sample_conditional, delete)
from .utils import logsumexp3

class PC:
    """
        Class that defines and evaluates an PC.

        Attributes
        ----------

        root: Node object
            The root node of the PC.
        ncat: numpy
            The number of categories of each variable. One for continuous variables.
        learner: object
            Defines the learning method of the PC.
            Currently, only LearnSPN (Gens and Domingos, 2013).
    """

    def __init__(self, ncat=None):
        self.ncat = ncat
        self.root = None
        self.n_nodes = 0
        self.is_ensemble = False  # Whether the PC was learned as an ensemble
        self.gate_weights = None
        self.gate_split_values = None
        self.sum_weights = None

    def learnspn(self, data, ncat=None, thr=0.001, nclusters=2, max_height=1000000, classcol=None):
        if ncat is not None:
            self.ncat = ncat
        assert self.ncat is not None, "You must provide `ncat`, the number of categories of each class."
        learner = LearnSPN(self.ncat, thr, nclusters, max_height, classcol)
        self.root = fit(learner, data)

    #
    def set_topological_order(self):
        """
            Updates the ids of the nodes so that they match their topological
            order.
        """
        def get_topological_order(node, order=[]):
            if order == []:
                node.id = len(order)
                order.append(node)
            for child in node.children:
                child.id = len(order)
                order.append(child)
            for child in node.children:
                get_topological_order(child, order)
            return order
        self.order = get_topological_order(self.root, [])
        self.n_nodes = len(self.order)

    #
    def log_likelihood(self, data, avg=False):
        """
            Computes the log-likelihood of data.

            Parameters
            ----------

            data: numpy array
                Input data including the class variable.
                Missing values should be set to numpy.nan
        """
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        if avg:
            nchildren = self.root.nchildren
            logs_avg = np.empty(shape=(data.shape[0], nchildren))
            for i in range(nchildren):
                logs_avg[:, i] = eval_root(self.root.children[i], data)
            logs_avg = np.mean(logs_avg, axis=1)
            return logs_avg
        return eval_root(self.root, data)

    def likelihood(self, data):
        """
            Computes the likelihood of data.

            Parameters
            ----------

            data: numpy array
                Input data including the class variable.
                Missing values should be set to numpy.nan
        """
        ll = self.log_likelihood(data)
        return np.exp(ll)

    def classify(self, X, classcol=None, return_prob=False):
        """
            Classifies instances running proper PC inference, that is,
            argmax_y P(X, Y=y).

            Parameters
            ----------

            X: numpy array
                Input data not including the variable to be predicted.
                The data should be ordered as in training, excluding the
                variable to be predicted (see example).
                Missing values should be set to numpy.nan
            classcol: int
                The index of the class to be predicted. If None, the model
                predicts the original target variable y (last column).
            return_prob: boolean
                Whether to return the conditional probability of each class.
            Example
            -------
            If a model is defined over 5 variables and one wants to predict
            variable 2, then the columns of X should contain observations of
            variables 0, 1, 3, and 4 in that order.
        """
        
        # perturbation
        eps = 1e-6

        # check the index of feature to be predicted
        if classcol is None:
            classcol = len(self.ncat)-1
        elif classcol != len(self.ncat)-1:
            # If not predicting the default target class (assumed to be the
            # last column), use the other classify function.
            return self.classify_lspn(X, classcol, return_prob)
        
        # number of classes for the target feature
        nclass = int(self.ncat[classcol])
        
        # generative decision tree can only make prediction where labels are categorical
        assert nclass > 1, "Only categorical variables can be classified."

        if type(X) == torch.Tensor:
            XX = X.clone()

            # the input of the GeDT should has 2 dimensions
            if XX.ndim == 1:
                XX = torch.unsqueeze(XX, dim=0)

            # log-probabilities with weight of all instances for each class and child of the root
            joints = eval_root_class(self.root, XX, classcol, nclass, naive=False)
            joints = torch.logsumexp(joints,dim=2)

            #   report this 'bug' to author and mentor
            expjoints = torch.exp(joints)

            # P(X)
            prob_sum = torch.sum(expjoints, keepdims=True, axis=1)

            # Index of cases where all classes have zero probability.
            unlikely_ones = torch.where(prob_sum == 0)[0]

            if len(unlikely_ones) > 0:
                print("Low density samples found at indices ", unlikely_ones, ". Probably out-of-domain samples.")
                prob_sum[unlikely_ones, :] += 1e-12

            # P(Y=y|X) = P(X,Y=y)/P(X)
            probs = expjoints / prob_sum
            if 0 in prob_sum:
                print(prob_sum)

            if return_prob:
                return torch.argmax(probs, axis=1), probs

            return torch.argmax(probs, axis=1)

        else:
            X = X.copy()

            # the input of the GeDT should has 2 dimensions
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)

            # log-probabilities with weight of all instances for each class and child of the root
            joints = eval_root_class(self.root, X, classcol, nclass, naive=False)
            # log-probabilities by summing all children
            # after logsumexp3: size=(number of instances, number of classes)
            joints = logsumexp3(joints, axis=2)

            # the maximum log-probability/log_count of target class

            max_values = np.max(joints, axis=1, keepdims=True)

            max_values = np.where(np.isfinite(max_values), max_values, 0)

            # the joint_minus_max is used for correcting the probability here
            joints_minus_max = joints - max_values

            probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)

            # P(X)
            prob_sum = np.sum(probs, keepdims=True, axis=1)

            # Index of cases where all classes have zero probability.
            unlikely_ones = np.where(prob_sum == 0)[0]

            if len(unlikely_ones) > 0:
                print("Low density samples found at indices ", unlikely_ones, ". Probably out-of-domain samples.")
                prob_sum[unlikely_ones, :] += 1e-12

            # P(Y=y|X) = P(X,Y=y)/P(X)
            probs = probs / prob_sum

            if return_prob:
                return np.argmax(probs, axis=1), probs

            return np.argmax(probs, axis=1)


    def classify_avg(self, X, classcol=None, return_prob=False, naive=False):
        """
            Classifies instances by taking the average of the conditional
            probabilities defined by each PC, that is,
            argmax_y sum_n P_n(Y=y|X)/n

            This is only makes sense if the PC was learned as an ensemble, where
            each model is the child of the root.

            Parameters
            ----------

            X: numpy array
                Input data not including the variable to be predicted.
                The data should be ordered as in training, excluding the
                variable to be predicted (see example).
                Missing values should be set to numpy.nan
            classcol: int
                The index of the class to be predicted. If None, the model
                predicts the original target variable y (last column).
            return_prob: boolean
                Whether to return the conditional probability of each class.
            naive: boolean
                Whether to treat missing values as suggested by  Friedman in 1975,
                that is, by taking the argmax over the counts of all pertinent
                cells.
            Example
            -------
            If a model is defined over 5 variables and one wants to predict
            variable 2, then the columns of X should contain observations of
            variables 0, 1, 3, and 4 in that order.
        """
        eps = 1e-6
        if classcol is None:
            classcol = len(self.ncat)-1
            
        elif classcol != len(self.ncat)-1:
            # If not predicting the default target class (assumed to be the
            # last column), use the other classify function.
            return self.classify_avg_lspn(X, classcol, return_prob)
        if not self.is_ensemble:
            return self.classify(X, classcol, return_prob)
        nclass = int(self.ncat[classcol])
        assert nclass > 1, "Only categorical variables can be classified."
        X = X.copy()
        
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        joints = eval_root_class(self.root, X, classcol, nclass, naive)
        
        if naive:
            counts = np.exp(joints).astype(int)  # int to filter out the smoothing
            conditional = counts/np.sum(counts, axis=1, keepdims=True)
        else:
            # Convert from log to probability space
            max_values = np.max(joints, axis=1, keepdims=True)
            max_values = np.where(np.isfinite(max_values), max_values, 0)
            joints_minus_max = joints - max_values
            probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
            # Normalize to sum out X: we get P(Y|X) by dividing by P(X)
            prob_sum = np.sum(probs, keepdims=True, axis=1)
            # Treat cases where all classes have low probability.
            unlikely_ones = np.where(prob_sum == 0)[0]
            if len(unlikely_ones) > 0:
                print("Low density samples found at indices ", unlikely_ones, ". Probably out-of-domain samples.")
                prob_sum[unlikely_ones, :] += 1e-12
            conditional = probs/prob_sum
            
        # Average over the trees
        agg = np.mean(conditional, axis=2)
        maxclass = np.argmax(agg, axis=1)
        if return_prob:
            return maxclass, agg
        return maxclass


    def classify_lspn(self, X, classcol=None, return_prob=False):
        """
            Classifies instances running proper PC inference, that is,
            argmax_y P(X, Y=y).

            Parameters
            ----------

            X: numpy array
                Input data not including the variable to be predicted.
                The data should be ordered as in training, excluding the
                variable to be predicted (see example).
                Missing values should be set to numpy.nan
            classcol: int
                The index of the class to be predicted. If None, the model
                predicts the original target variable y (last column).
            return_prob: boolean
                Whether to return the conditional probability of each class.
            naive: boolean
                Whether to treat missing values as suggested in Friedman1975,
                that is, by taking the argmax over the counts of all pertinent
                cells.

            Example
            -------
            If a model is defined over 5 variables and one wants to predict
            variable 2, then the columns of X should contain observations of
            variables 0, 1, 3, and 4 in that order.
        """
        # we use classify_lspn when there are many spn leaf nodes because it is faster
        eps = 1e-6
        if classcol is None:
            classcol = len(self.ncat)-1
        nclass = int(self.ncat[classcol])
        assert nclass > 1, "Only categorical variables can be classified."
        X = X.copy()
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        maxclass = np.zeros(X.shape[0])-1
        maxlogpr = np.zeros(X.shape[0])-np.Inf

        # shape:(nsamples,nclass)
        joints = np.zeros((X.shape[0], nclass))

        # predict each class joint probability
        for i in range(nclass):
            iclass = np.zeros(X.shape[0]) + i
            Xi = np.insert(X, classcol, iclass, axis=1)
            joints[:, i] = np.squeeze(eval_root(self.root, Xi))
        max_values = np.max(joints, axis=1, keepdims=True)
        max_values = np.where(np.isfinite(max_values), max_values, 0)
        joints_minus_max = joints - max_values
        probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
        prob_sum = np.sum(probs, keepdims=True, axis=1)
        # Treat cases where all classes have low probability.
        unlikely_ones = np.where(prob_sum == 0)[0]
        if len(unlikely_ones) > 0:
            print("Low density samples found at indices ", unlikely_ones, ". Probably out-of-domain samples.")
            prob_sum[unlikely_ones, :] += 1e-12
        probs = probs/prob_sum
        if return_prob:
            return np.argmax(probs, axis=1), probs
        return np.argmax(probs, axis=1)

    def classify_avg_lspn(self, X, classcol=None, return_prob=False):
        """
            Classifies instances by taking the average of the conditional
            probabilities defined by each tree, that is,
            argmax_y sum_n P_n(Y=y|X)/n
            This is only makes sense if the PC was learned as an ensemble, where
            each model is the child of the root.

            Parameters
            ----------

            X: numpy array
                Input data not including the variable to be predicted.
                The data should be ordered as in training, excluding the
                variable to be predicted (see example).
                Missing values should be set to numpy.nan
            classcol: int
                The index of the class to be predicted. If None, the model
                predicts the original target variable y (last column).
            return_prob: boolean
                Whether to return the conditional probability of each class.

            Example
            -------
            If a model is defined over 5 variables and one wants to predict
            variable 2, then the columns of X should contain observations of
            variables 0, 1, 3, and 4 in that order.
        """
        eps = 1e-6
        if classcol is None:
            classcol = len(self.ncat)-1
        nclass = int(self.ncat[classcol])
        assert nclass > 1, "Only categorical variables can be classified."
        if not self.is_ensemble:
            return self.classify_lspn(X, classcol, return_prob)
        X = X.copy()
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        joints = np.zeros((X.shape[0], self.root.nchildren, nclass))
        for i in range(nclass):
            iclass = np.zeros(X.shape[0]) + i
            Xi = np.insert(X, classcol, iclass, axis=1)
            joints[:, :, i] = eval_root_children(self.root, Xi)
        max_values = np.max(joints, axis=2, keepdims=True)
        max_values = np.where(np.isfinite(max_values), max_values, 0)
        joints_minus_max = joints - max_values
        probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
        prob_sum = np.sum(probs, keepdims=True, axis=2)
        # Treat cases where all classes have low probability.
        unlikely_ones = np.where(prob_sum == 0)[0]
        if len(unlikely_ones) > 0:
            print("Low density samples found at indices ", unlikely_ones, ". Probably out-of-domain samples.")
            prob_sum[unlikely_ones, :] += 1e-12
        normalized = probs/prob_sum
        agg = np.mean(normalized, axis=1)
        maxclass = np.argmax(agg, axis=1)
        if return_prob:
            return maxclass, agg
        return maxclass

    def sample(self, n_samples=1):
        """ Returns samples from the distribution defined by the PC. """
        return sample(self.root, n_samples)

    def sample_conditional(self, evi):
        """ Returns samples from the distribution defined by the PC
            given the evidence in `evi`.

            Parameters
            ----------
            evi: numpy array (n_samples, n_variables)
                Non-observed variables should be set to numpy.nan.
        """
        return sample_conditional(self.root, evi)

    def clear(self):
        """ Deletes the structure of the PC. """
        if self.root is not None:
            self.root.remove_children(*self.root.children)
            self.root = None

    def get_node(self, id):
        """ Fetchs node by its id. """
        queue = [self.root]
        while queue != []:
            node = queue.pop(0)
            if node.id == id:
                return node
            if node.type not in ['L', 'G', 'U']:
                queue.extend(node.children)
        print("Node %d is not part of the network.", id)

    def get_node_of_type(self, type):
        """ Fetchs all nodes of a given type. """
        queue = [self.root]
        res = []
        while queue != []:
            node = queue.pop(0)
            if node.type == type:
                res.append(node)
            if node.type not in ['L', 'G', 'U']:
                queue.extend(node.children)
        return res

    def delete(self):
        """
            Calls the delete function of the root node, which in turn deletes
            the rest of the nodes in the PC. Given that nodes in an PC point
            to each other, they are always referenced and never automatically
            deleted by the Python interpreter.
        """
        delete(self.root)
