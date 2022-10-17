import numba as nb
import numpy as np
import torch
from .signed import (signed, signed_prod, signed_econtaminate, signed_join)
from .utils import (bincount, logtrunc_phi, isin, lse, logsumexp2,
                    logsumexp3, nb_argmax, categorical, sample_trunc_phi)

class Node:
    def __init__(self, scope, type, n):
        self.id = np.random.randint(0, 10000000) # Random identification number
        # initialize left and right children as None
        self.left_child = None
        self.right_child = None
        self.sibling = None
        self.nchildren = 0
        self.scope = scope
        self.type = type
        # n is the number of instances with Laplace smoothing
        self.n = n 
        # Sum params
        self.w = None
        self.logw = None
        self.tempw = None
        self.theta = None # re-parameterization to replace w
        # Leaf params
        self.value = np.zeros(1, dtype=np.float64)
        self.comparison = -1
        # Gaussian leaf params
        self.mean = 0.
        self.std = 1.
        self.a = -np.Inf
        self.b = np.Inf
        self.p = None
        self.logp = None
        self.logcounts = None
        self.upper = np.ones(len(scope))*(np.Inf)
        self.lower = np.ones(len(scope))*(-np.Inf)
        # Gate params
        self.gate_split_value = torch.tensor([0.], requires_grad=True)
        self.gate_weight = torch.tensor([-1.], requires_grad=True)
    @property
    def children(self):
        """ A list with all children. """
        children = []
        child = self.left_child
        while child is not None:
            children.append(child)
            child = child.sibling
        return children

    def add_sibling(self, sibling):
        self.sibling = sibling

    def add_child(self, child):
        # from left most child to right most child, a chain is built here
        
        # if parent has no children
        if self.left_child is None:
            # this node is it first child
            self.left_child = child
            self.right_child = child
        else:
            # the last (now right) child will have this node as sibling
            self.right_child.add_sibling(child)
            self.right_child = child
        self.nchildren += 1
        if self.type == 'S':
            self.reweight()

    def add_child_torch(self, child):
        # from left most child to right most child, a chain is built here

        # if parent has no children
        if self.left_child is None:
            # this node is it first child
            self.left_child = child
            self.right_child = child
        else:
            # the last (now right) child will have this node as sibling
            self.right_child.add_sibling(child)
            self.right_child = child
        self.nchildren += 1
        if self.type == 'S':
            self.reweight_torch()

    def reweight(self):
        children_n = np.array([c.n for c in self.children])
        n = np.sum(children_n)
        if n > 0:
            self.n = n
            self.w = np.divide(children_n.ravel(), self.n)
        else:
            self.n = 0
            self.w = np.ones(self.nchildren) * (1/self.nchildren)
        self.logw = np.log(self.w.ravel())

    # reweight when adding child to a node
    def reweight_torch(self):
        children_n = torch.tensor([c.n for c in self.children])
        n = torch.sum(children_n)
        if n > 0:
            self.n = n
            self.theta = torch.log(torch.div(children_n, self.n))
            self.theta.requires_grad = True

            # re-parameterizaiton, the initial value of w is children_n/n
            self.w = torch.exp(self.theta)/torch.sum(torch.exp(self.theta))
        else:
            self.n = 0
            self.theta = torch.log(torch.ones(self.nchildren) * (1/self.nchildren))
            self.theta.requires_grad = True
            self.w = torch.exp(self.theta)/torch.sum(torch.exp(self.theta))

        self.logw = torch.log(self.w)

    # set temperary weight
    def set_tempw(self, array):
        self.tempw = array



###########################
### INTERFACE FUNCTIONS ###
###########################

def ProdNode(scope, n):
    return Node(scope, 'P', n)

def SumNode(scope, n):
    return Node(scope, 'S', n)

def Leaf(scope, n, value, comparison):
    node = Node(scope, 'L', n)
    fit_indicator(node, value, comparison)
    return node

def GaussianLeaf(scope, n):
    return Node(scope, 'G', n)

def MultinomialLeaf(scope, n):
    return Node(scope, 'M', n)

def UniformLeaf(scope, n, value):
    node = Node(scope, 'U', n)
    node.value = value
    return node

def Gate_L(scope, n,value, comparison):
    node = Node(scope, 'LG', n)
    fit_gate_l(node,value,comparison)
    return node

def Gate_R(scope, n,value, comparison, weight= -1.0):
    node = Node(scope, 'LG', n)
    fit_gate_r(node,value,weight,comparison)
    return node


#################################
###    AUXILIARY FUNCTIONS    ###
#################################


def n_nodes(node):
    if node.type in ['L', 'G', 'M']:
        return 1
    if node.type in ['S', 'P']:
        n = 1 + np.sum([n_nodes(c) for c in node.children])
    return n


def delete(node):
    for c in node.children:
        delete(c)
    node.left_child = None
    node.right_child = None
    node.sibling = None
    node = None


###########################
###    FIT FUNCTIONS    ###
###########################

def fit_gaussian(node, data, upper, lower, minstd=1):
    assert node.type == 'G', "Only gaussian leaves fit data."
    assert minstd > 0, "Minimum standard deviation should be positive."
    node.n = data.shape[0]

    # mean value
    m = np.nanmean(data[:, node.scope])
    if np.isnan(m):  # No data was observed for this variable (node.scope)
        node.mean = 0.  # Assuming the data has been standardized
        node.std = np.sqrt(1.)
    else:
        node.mean = m
        if node.n > 1:  # Avoid runtimewarning
            node.std = np.std(data[:, node.scope])
        else:
            node.std = np.sqrt(1.)  # Probably not the best solution here
        node.std = max(minstd, node.std)
        # Compute the tresholds to truncate the Gaussian.
        # The Gaussian has support [a, b]
    node.a = lower
    node.b = upper


def fit_multinomial(node, data, k, smoothing=1e-6):
    assert node.type == 'M', "Node is not a multinomial leaf."
    d = data[~np.isnan(data[:, node.scope].ravel()), :]  # Filter missing
    d = data[:, node.scope].ravel()  # Project to scope
    d = np.asarray(d, np.int64)
    # k is the number of categories of the scope
    # bincount the data in each category
    # node.p is the probability distribution
    if d.shape[0] > 0:
        counts = bincount(d, k) + smoothing
        node.logcounts = np.log(counts)
        node.p = counts/(d.shape[0] + k*smoothing)
        
    # If there is no data, each category should has the same probability.
    # data has shape 0 because all values on this scope is missing
    else:
        counts = np.ones(k)
        node.logcounts = np.log(counts)
        node.p = counts * (1/k)
    # log-probability of the node
    node.logp = np.log(np.asarray(node.p))


def fit_multinomial_with_counts(node, counts):
    assert node.type == 'M', "Node is not a multinomial leaf."
    node.logcounts = np.log(np.asarray(counts))
    node.p = counts/np.sum(counts)
    node.logp = np.log(np.asarray(node.p))


def fit_indicator(node, value, comparison):
    node.value = value
    node.comparison = comparison

def fit_gate_l(node, value,comparison):
    node.comparison = comparison
    node.gate_split_value.data = torch.tensor([value])

def fit_gate_r(node, value, weight,comparison):
    node.comparison = comparison
    node.gate_split_value = value
    node.gate_weight = weight

###########################
### EVALUATE FUNCTIONS  ###
###########################

def eval_eq(scope, value, evi):
    """
        Evaluates an indicator of the type 'equal'.
        True if evi[scope] == value, False otherwise.
    """
    if type(evi) == torch.Tensor:
        s, v = scope[0], value[0]
        res = torch.zeros(evi.shape[0], dtype=np.float64)-np.inf
        for i in range(evi.shape[0]):
            if (evi[i, s] == v) or torch.isnan(evi[i, s]):
                res[i] = 0
        return res
    else:
        s, v = scope[0], value[0]
        res = np.zeros(evi.shape[0], dtype=np.float64)-np.Inf
        for i in range(evi.shape[0]):
            if (evi[i, s] == v) or np.isnan(evi[i, s]):
                res[i] = 0
        return res


def eval_leq(scope, value, evi):
    """
        Evaluates an indicator of the type 'less or equal'.
        True if evi[scope] <= value, False otherwise.
    """
    if type(evi) == torch.Tensor:
        s, v = scope[0], value[0]
        res = torch.zeros(evi.shape[0], dtype=torch.float64)
        for i in range(evi.shape[0]):
            if evi[i, s] > v:
                res[i] = -np.inf
        return res
    else:
        s, v = scope[0], value[0]
        res = np.zeros(evi.shape[0], dtype=np.float64)
        for i in range(evi.shape[0]):
            if evi[i, s] > v:
                res[i] = -np.Inf
        return res


def eval_g(scope, value, evi):
    """
        Evaluates an indicator of the type 'greater'.
        True if evi[scope] > value, False otherwise.
    """

    if type(evi) == torch.Tensor:
        s, v = scope[0], value[0]
        res = torch.zeros(evi.shape[0], dtype=torch.float64)
        for i in range(evi.shape[0]):
            if evi[i, s] <= v:
                res[i] = -np.inf
        return res
    else:
        s, v = scope[0], value[0]
        res = np.zeros(evi.shape[0], dtype=np.float64)
        for i in range(evi.shape[0]):
            if evi[i, s] <= v:
                res[i] = -np.Inf
        return res


def eval_in(scope, value, evi):
    """
        Evaluates an indicator of the type 'in':
        True if evi[scope] in value, False otherwise.
    """
    if type(evi) == torch.Tensor:
        s, v = scope[0], value[0]
        res = torch.zeros(evi.shape[0], dtype=np.float64)
        for i in range(evi.shape[0]):
            if not isin(evi[i, s], torch.from_numpy(value)):
                res[i] = -np.inf
        return res
    else:
        s, v = scope[0], value[0]
        res = np.zeros(evi.shape[0], dtype=np.float64)
        for i in range(evi.shape[0]):
            if not isin(evi[i, s], value):
                res[i] = -np.Inf
        return res

def eval_gate_leq(scope,value,gate_split_value,gate_weight,evi):
    """
        Evaluates the gate of the type 'less or equal'.
        True if evi[scope] <= value, False otherwise.
    """
    s, v = scope[0], value[0]
    if evi[0,s] != np.nan:
        res = torch.log(torch.sigmoid((evi[:, s]-gate_split_value) * gate_weight))
    else:
        res = torch.zeros(evi.shape[0], dtype=torch.float64)
    return res


def eval_gate_g(scope, value,gate_split_value,gate_weight,evi):
    """
        Evaluates the gate of the type 'greater'.
        True if evi[scope] > value, False otherwise.
    """
    s, v = scope[0], value[0]
    if evi[0,s] != np.nan:
        res = torch.log(1 - torch.sigmoid((evi[:, s]-gate_split_value) * gate_weight))
    else:
        res = torch.zeros(evi.shape[0], dtype=torch.float64)
    return res


def eval_gaussian(node, evi):
    """ Evaluates a Gaussian leaf. """
    if type(evi) == torch.Tensor:
        return logtrunc_phi(evi[:, node.scope], node.mean, node.std, node.a, node.b)
    else:
        return logtrunc_phi(evi[:, node.scope].ravel(), node.mean, node.std, node.a, node.b).reshape(-1)



def eval_m(node, evi):
    """ Evaluates a multinomial leaf. """
    if type(evi) == torch.Tensor:
        s = node.scope[0]
        res = torch.zeros(evi.shape[0], dtype=torch.float64)
        for i in range(evi.shape[0]):
            res[~torch.isnan(evi[:,s])]
        return res
    else:
        s = node.scope[0]
        res = np.zeros(evi.shape[0], dtype=np.float64)

        # traverse all samples
        for i in range(evi.shape[0]):
            obs = evi[i, s]
            if not np.isnan(obs):
                if obs >= len(node.p):
                    print("Previously unobserved category for variable ", s)
                    res[i] = np.log(1e-6)  # return a low probability value
                else:
                    # node.logp is already computed
                    # int(evi[i,s]) is the class of instance i on scope s
                    res[i] = node.logp[int(evi[i, s])]
        return res


def compute_batch_size(n_points, n_features):
    maxmem = max(n_points * n_features + (n_points)/10, 10 * 2 ** 17)
    batch_size = (-n_features + np.sqrt(n_features ** 2 + 4 * maxmem)) / 2
    return int(batch_size)


def eval_root(node, evi):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        This function only applies to the root of a PC.
    """
    if node.type == 'S':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        for i in nb.prange(node.nchildren):
            logprs[:, i] = evaluate(node.children[i], evi) + node.logw[i]
        res = logsumexp2(logprs, axis=1)
        return res
    elif node.type == 'P':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        logprs[:, 0] = evaluate(node.children[0], evi)
        nonzero = np.where(logprs[:, 0] != -np.Inf)[0]
        if len(nonzero) > 0:
            for i in nb.prange(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[nonzero, i] = evaluate(node.children[i], evi[nonzero, :])
        return np.sum(logprs, axis=1)
    else:
        return evaluate(node, evi)


def eval_root_children(node, evi):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        This function only applies to the root of a PC.
    """
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS
    sizes = np.full(n_threads, node.nchildren // n_threads, dtype=np.int32)
    sizes[:node.nchildren % n_threads] += 1
    offset_in_buffers = np.zeros(n_threads, dtype=np.int32)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])
    logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
    for thread_idx in nb.prange(n_threads):
        start = offset_in_buffers[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            logprs[:, i] = evaluate(node.children[i], evi) + node.logw[i]
    return logprs


def evaluate(node, evi):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        This function applies to any node in a PC and is called recursively.
    """
    if node.type == 'L':
        if node.comparison == 0:  # IN
            return eval_in(node.scope, node.value.astype(np.int64), evi)
        elif node.comparison == 1:  # EQ
            return eval_eq(node.scope, node.value.astype(np.float64), evi)
        elif node.comparison == 3:  # LEQ
            return eval_leq(node.scope, node.value, evi)
        elif node.comparison == 4:  # G
            return eval_g(node.scope, node.value, evi)
    elif node.type == 'M':
        return eval_m(node, evi)
    elif node.type == 'G':
        return eval_gaussian(node, evi)
    elif node.type == 'U':
        return np.ones(evi.shape[0]) * node.value
    elif node.type == 'P':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        logprs[:, 0] = evaluate(node.children[0], evi)
        nonzero = np.where(logprs[:, 0] != -np.Inf)[0]
        if len(nonzero) > 0:
            for i in nb.prange(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[nonzero, i] = evaluate(node.children[i], evi[nonzero, :])
        return np.sum(logprs, axis=1)
    elif node.type == 'S':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        for i in nb.prange(node.nchildren):
            logprs[:, i] = evaluate(node.children[i], evi) + node.logw[i]
        res = logsumexp2(logprs, axis=1)
        return res
    return np.zeros(evi.shape[0])


def eval_root_class(node, evi, class_var, n_classes, naive):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        Same as `eval_root` but evaluates all possibles instantiations of the
        class variable at once. For that three extra parameters are required.

        Parameters
        ----------
        class_var: int
            Index of the class variable
        n_classes: int
            Number of classes in the data
        naive: boolean
            Whether to simply propagate the counts (Friedman method).

        Returns
        -------
        logprs: numpy array (float) of shape (n_samples, n_classes, n_trees)
    """
    
    # default number of threads is the number of cpu cores
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS

    # number of children for each thread
    sizes = np.full(n_threads, node.nchildren // n_threads, dtype=np.int32)

    # deal with remainder, now its the actual number of children per thread
    sizes[:node.nchildren % n_threads] += 1
    
    # calculate the offset for each thread in buffer
    offset_in_buffers = np.zeros(n_threads, dtype=np.int32)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])

    if type(evi) == torch.Tensor:
        # log probabilities for each instance, each class of the target feature and each child
        logprs = torch.zeros((evi.shape[0], n_classes, node.nchildren), dtype=torch.float64)
        # run threads in parallel
        for thread_idx in nb.prange(n_threads):
            start = offset_in_buffers[thread_idx]
            stop = start + sizes[thread_idx]

            # loop through all children in this thread
            for i in range(start, stop):
                if naive:
                    logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes,naive)  # no weights here
                else:
                    logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes,naive) + torch.log(torch.exp(node.theta[i])/torch.sum(torch.exp(node.theta)))

        return logprs



    else:
        logprs = np.zeros((evi.shape[0], n_classes, node.nchildren), dtype=np.float64)
        # run threads in parallel
        for thread_idx in nb.prange(n_threads):
            start = offset_in_buffers[thread_idx]
            stop = start + sizes[thread_idx]

            # loop through all children in this thread
            for i in range(start, stop):
                if naive:
                    logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes,
                                                     naive)  # no weights here
                else:
                    logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive) + node.logw[i]
        return logprs


def evaluate_class(node, evi, class_var, n_classes, naive):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        This function applies to any node in a PC and is called recursively.
        Same as `evaluate` but evaluates all possibles instantiations of the
        class variable at once. For that three extra parameters are required.

        Parameters
        ----------
        class_var: int
            Index of the class variable
        n_classes: int
            Number of classes in the data
        naive: boolean
            Whether to simply propagate the counts (Friedman method).
    """
    if type(evi) == torch.Tensor:
        if node.type == 'L':#Indicator Node
            res = torch.zeros((evi.shape[0], 1),dtype=torch.float64)
            if node.comparison == 0:  # IN
                res[:, 0] = eval_in(node.scope, node.value.astype(np.int64), evi)
            elif node.comparison == 1:  # EQ
                res[:, 0] = eval_eq(node.scope, node.value.astype(np.float64), evi)
            elif node.comparison == 3:  # LEQ
                res[:, 0] = eval_leq(node.scope, node.value, evi)
            elif node.comparison == 4:  # G
                res[:, 0] = eval_g(node.scope, node.value, evi)
            return res
        elif node.type == 'LG':#Gating Node
            res = torch.zeros((evi.shape[0], 1), dtype=torch.float64)
            if node.comparison == 3:  # LEQ
                res[:, 0] = eval_gate_leq(node.scope, node.value, node.gate_split_value, node.gate_weight, evi)
            elif node.comparison == 4:  # G
                res[:, 0] = eval_gate_g(node.scope, node.value,node.gate_split_value, node.gate_weight, evi)
            return res
        elif node.type == 'M':#Multinomial Node
            # check if the scope is the target feature
            if isin(class_var, node.scope):
                if naive:
                    return torch.zeros((evi.shape[0], n_classes),dtype=torch.float64) + node.logcounts
                else:
                    res = torch.zeros((evi.shape[0], n_classes), dtype=torch.float64) + node.logp
                    return res
            if naive:
                return torch.zeros((evi.shape[0], 1),dtype=torch.float64)
            res = torch.zeros((evi.shape[0], 1),dtype=torch.float64)
            res[:, 0] = eval_m(node, evi)
            return res
        elif node.type == 'G':#Gaussian Node
            res = torch.zeros((evi.shape[0], 1),dtype=torch.float64)
            if naive:
                return res
            res = eval_gaussian(node, evi)
            return res
        elif node.type == 'U':#Uniform Node
            if naive:
                return torch.zeros((evi.shape[0], 1))
            return torch.ones((evi.shape[0], 1)) * torch.from_numpy(node.value)
        elif node.type == 'P':#Product Node
            logprs = torch.zeros((evi.shape[0], n_classes, node.nchildren), dtype=torch.float64)

            # the first child of the product node is the gate, so this line tries to get the value of gate
            logprs[:, :, 0] = evaluate_class(node.children[0], evi, class_var, n_classes, naive)

            for i in range(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive)

            return torch.sum(logprs, dim=2,dtype=torch.float64)

        elif node.type == 'S':#Sum Node
            logprs = torch.zeros((evi.shape[0], n_classes, node.nchildren), dtype=torch.float64)
            if naive:
                for i in range(node.nchildren):
                    logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes,
                                                     naive)  # no weights here
            else:
                for i in range(node.nchildren):
                    if node.theta != None:
                        # the weight of sum node is replaced by softmax on theta
                        logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes,
                                                     naive) + torch.log(torch.exp(node.theta[i])/torch.sum(torch.exp(node.theta)))
                    else: #if it is a sum node in SPN leaf
                        logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes,
                                                     naive) + node.logw[i]
            return torch.logsumexp(logprs, 2)
        return torch.zeros((evi.shape[0], 1),dtype=torch.float64)
    else:
        if node.type == 'L':
            res = np.zeros((evi.shape[0], 1))
            if node.comparison == 0:  # IN
                res[:, 0] = eval_in(node.scope, node.value.astype(np.int64), evi)
            elif node.comparison == 1:  # EQ
                res[:, 0] = eval_eq(node.scope, node.value.astype(np.float64), evi)
            elif node.comparison == 3:  # LEQ
                res[:, 0] = eval_leq(node.scope, node.value, evi)
            elif node.comparison == 4:  # G
                res[:, 0] = eval_g(node.scope, node.value, evi)
            return res
        elif node.type == 'M':
            # check if the scope is the target feature
            if isin(class_var, node.scope):
                if naive:
                    return np.zeros((evi.shape[0], n_classes)) + node.logcounts
                else:
                    res = np.zeros((evi.shape[0], n_classes)) + node.logp
                    return res
            if naive:
                return np.zeros((evi.shape[0], 1))
            res = np.zeros((evi.shape[0], 1))
            res[:, 0] = eval_m(node, evi)
            return res
        elif node.type == 'G':
            res = np.zeros((evi.shape[0], 1))
            if naive:
                return res
            res[:, 0] = eval_gaussian(node, evi)
            return res
        elif node.type == 'U':
            if naive:
                return np.zeros((evi.shape[0], 1))
            return np.ones((evi.shape[0], 1)) * node.value
        elif node.type == 'P':

            logprs = np.zeros((evi.shape[0], n_classes, node.nchildren), dtype=np.float64)

            # the first child of the product node is the indicator, so this line tries to get the value of indicator
            # which log-probability is 0 for those that has 1 probability and -Inf for those that has 0 probability
            logprs[:, :, 0] = evaluate_class(node.children[0], evi, class_var, n_classes, naive)

            # use this to filterout cases that removed by the indicator
            nonzero = ~np.isinf(logprs[:, 0, 0])

            if np.sum(nonzero) > 0:
                for i in range(1, node.nchildren):
                    # Only evaluate nonzero examples to save computation
                    logprs[nonzero, :, i] = evaluate_class(node.children[i], evi[nonzero, :], class_var, n_classes,
                                                           naive)
            return np.sum(logprs, axis=2)

        elif node.type == 'S':
            logprs = np.zeros((evi.shape[0], n_classes, node.nchildren), dtype=np.float64)
            if naive:
                for i in range(node.nchildren):
                    logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes,
                                                     naive)  # no weights here
            else:
                for i in range(node.nchildren):
                    logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive) + node.logw[i]
            return logsumexp3(logprs, axis=2)
        return np.zeros((evi.shape[0], 1))


##################################
###    ROBUSTNESS FUNCTIONS    ###
##################################


def eval_m_rob(node, evi, n_classes, eps, ismax):
    s = node.scope[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if not np.isnan(evi[i, s]):
            logprs = np.zeros(node.p.shape)-np.Inf
            logprs[int(evi[i, s])] = 0.
            econt = econtaminate(node.p, logprs, eps, ismax)
            res[i] = lse(logprs + np.log(econt))
    return res


def econtaminate(vec, logprs, eps, ismax):
    """
        Returns a `eps`-contaminated version of `vec`.

        Parameters
        ----------
        vec: numpy array
            The original array of parameters.
        logprs: numpy array (same dimension as `vec`)
            The log-density associated to each parameter in vec.
            Typically, logprs[i] is the log-density (at a given evidence) of
            the PC rooted at the ith child of the sum node with parameters `vec`.
        eps: float between 0 and 1.
            The epsilon used to contaminate the parameters.
            See https://arxiv.org/abs/2007.05721
        ismax: boolean
            If True, the parameters in `vec` are perturbed so that the dot
            product between the `eps`-contaminated version of `vec` and `logprs`
            is maximised. If False, this dot product is minimised.
    """
    econt = np.asarray(vec) * (1-eps)
    room = 1 - np.sum(econt)
    if ismax:
        order = np.argsort(-1*logprs)
    else:
        order = np.argsort(logprs)
    for i in order:
        if room > eps:
            econt[i] = econt[i] + eps
            room -= eps
        else:
            econt[i] = econt[i] + room
            break
    return econt


def evaluate_rob_class(node, evi, class_var, n_classes, eps, maxclass):
    """
        Computes the expected value of min[P(evi, y') - P(evi, maxclass)] and
        min[P(evi, y') - P(evi, maxclass)], where maxclass is the predicted
        class and y' is any other possible class. Because this difference might
        be negative, we need to propagate negative values through the network
        which required signed values (see signed.py).

        Parameters
        ----------
        node: Node object (nodes.py)
            The root of the Probabilistic Circuit.
        evi: numpy array of size m
            Single realisation of m variables.
        class_var: int
            Index of the class variable
        n_classes: int
            Number of classes in the data
        eps: float between 0 and 1.
            The epsilon used to contaminate the parameters.
            See https://arxiv.org/abs/2007.05721
        maxclass: int
            Index of the predicted class.

        Returns
        -------
        res_min, res_max: signed arrays of size n_class
            The minimum and maximum values the density function assumes within
            the epsilon-contaminated set.
    """
    one = np.array([1.])
    if node.type == 'L':
        if node.comparison == 0:  # IN
            res = eval_in(node.scope, node.value.astype(np.int64), evi)
        elif node.comparison == 1:  # EQ
            res = eval_eq(node.scope, node.value.astype(np.float64), evi)
        elif node.comparison == 3:  # LEQ
            res = eval_leq(node.scope, node.value, evi)
        elif node.comparison == 4:  # G
            res = eval_g(node.scope, node.value, evi)
        res = signed(res, one)
        return res, res  # min, max are the same
    elif node.type == 'M':
        if isin(class_var, node.scope):
            indicators = np.ones(n_classes, dtype=nb.boolean) # np.bool
            indicators[maxclass] = 0
            # Min
            econt_min = np.asarray(node.p)*(1-eps)
            econt_min[indicators] = econt_min[indicators] + eps
            econt_min = np.asarray(econt_min[maxclass]) - econt_min
            res_min = signed(econt_min, None)
            # Max
            econt_max = np.asarray(node.p)*(1-eps)
            econt_max[~indicators] = econt_max[~indicators] + eps
            econt_max = np.asarray(econt_max[maxclass]) - econt_max
            res_max = signed(econt_max, None)
        else:
            # Min
            res_min = eval_m_rob(node, evi, n_classes, eps, False)
            # Max
            res_max = eval_m_rob(node, evi, n_classes, eps, True)
            res_min, res_max = signed(res_min, one), signed(res_max, one)
        return res_min, res_max
    elif node.type == 'G':
        res = np.zeros((evi.shape[0], 1))
        delta = eps/2
        point = evi[:, node.scope].ravel()
        left = logtrunc_phi(point, node.mean-delta, node.std, node.a, node.b).reshape(-1)
        right = logtrunc_phi(point, node.mean+delta, node.std, node.a, node.b).reshape(-1)
        if left[0] >= right[0]:
            res_min = signed(right, one)
            res_max = signed(left, one)
        else:
            res_min = signed(left, one)
            res_max = signed(right, one)
        if (point[0] + delta >= node.mean) & (point[0] - delta <= node.mean):
            res_max = signed(np.asarray([node.mean]), one)
        return res_min, res_max
    elif node.type == 'P':
        res_min = signed(np.array([1]), None)
        res_max = signed(np.array([1]), None)
        res_min_cl, res_max_cl = None, None
        for i in range(node.nchildren):
            # Only evaluate nonzero examples to save computation
            child_min, child_max = evaluate_rob_class(node.children[i], evi, class_var, n_classes, eps, maxclass)
            if (i == 0) and (np.all(child_max.nonzero()) == False) and (np.all(child_min.nonzero()) == False):
                return child_max, child_max
            if isin(class_var, node.children[i].scope):
                res_min_cl, res_max_cl = child_min, child_max
            else:
                res_min = signed_prod(res_min, child_min)
                res_max = signed_prod(res_max, child_max)
        new_res_min = None
        new_res_max = None
        if res_min_cl != None and res_max_cl != None:
            for j in range(n_classes):
                if res_min_cl.sign[j] < 0:
                    res_min_j = signed_prod(res_max, res_min_cl.get(j))
                else:
                    res_min_j = signed_prod(res_min, res_min_cl.get(j))
                new_res_min = signed_join(new_res_min, res_min_j)
                if res_max_cl.sign[j] < 0:
                    res_max_j = signed_prod(res_min, res_max_cl.get(j))
                else:
                    res_max_j = signed_prod(res_max, res_max_cl.get(j))
                new_res_max = signed_join(new_res_max, res_max_j)
            return new_res_min, new_res_max
        return res_min, res_max
    elif node.type == 'S':
        values_min, values_max = np.zeros((node.nchildren, n_classes)), np.zeros((node.nchildren, n_classes))
        signs_min, signs_max = np.zeros((node.nchildren, n_classes)), np.zeros((node.nchildren, n_classes))
        for i in range(node.nchildren):
            res_min, res_max = evaluate_rob_class(node.children[i], evi, class_var, n_classes, eps, maxclass)
            values_min[i, :], signs_min[i, :] = res_min.value, res_min.sign
            values_max[i, :], signs_max[i, :] = res_max.value, res_max.sign
        res_min, res_max = signed(np.zeros(n_classes), None), signed(np.zeros(n_classes), None)
        for j in range(n_classes):
            min_j = signed(values_min[:, j], signs_min[:, j])
            econt_min = signed_econtaminate(node.w, min_j, eps, False)
            res_min_j = signed_prod(min_j, econt_min)
            res_min.insert(res_min_j.reduce(), j)
            max_j = signed(values_max[:, j], signs_max[:, j])
            econt_max = signed_econtaminate(node.w, max_j, eps, True)
            res_max_j = signed_prod(max_j, econt_max)
            res_max.insert(res_max_j.reduce(), j)
        return res_min, res_max
    res = signed(np.ones(evi.shape[0]), None)  # Just so numba compiles
    return res, res


def compute_rob_class(node, evi, class_var, n_classes):
    """
        Compute the robustness of the PC rooted at `node` for each instance in `evi`.

        Parameters
        ----------
        node: Node object (nodes.py)
            The root of the Probabilistic Circuit.
        evi: numpy array n x m
            Data with n samples and m variables.
        class_var: int
            Index of the class variable
        n_classes: int
            Number of classes in the data
    """
    rob = np.zeros(evi.shape[0])
    logprobs = evaluate_class(node, evi, class_var, n_classes, False)
    maxclass = nb_argmax(logprobs, axis=1)
    for i in nb.prange(evi.shape[0]):
        rob[i] = rob_loop_class(node, evi[i:i+1, :], class_var, n_classes, maxclass[i])
    return maxclass, rob


def rob_loop_class(node, evi, class_var, n_classes, maxclass):
    """
        Same as `compute_rob_class` but for a single instance.

        Performs a binary search, increasing the value of eps until
        min[P(evi, y') - P(evi, maxclass)] becomes negative. A negative value
        here indicates that not every PC in the class of PCs defined by a
        contamination of eps yields the same classification, and hence the
        robustness value should be less than eps.
        See https://arxiv.org/abs/2007.05721 for details.
    """
    lower = 0
    upper = 1
    it = 0
    while (lower < upper - .005) & (it <= 200):
        ok = True
        rob = (lower + upper)/2
        min_values, max_values = evaluate_rob_class(node, evi, class_var, n_classes, rob, maxclass)
        for j in range(n_classes):
            if j != maxclass:
                if min_values.get(j).sign[0] <= 0:
                    ok = False
                    break
        if ok:
            lower = rob
        else:
            upper = rob
        it += 1
    return rob


def sample(node, n_samples=1):
    res = np.zeros((n_samples, len(node.scope)))
    for i in nb.prange(n_samples):
        sample_aux(node, res[i, :])
    return res


def sample_aux(node, res=None):
    """
        Returns one sample from the distribution defined by the PC rooted at node.
    """
    if res is None:
        res = np.zeros(len(node.scope))
    if node.type == 'S':
        index = categorical(node.w)[0]
        return sample_aux(node.children[index], res)
    elif node.type == 'P':
        for child in node.children:
            resi = sample_aux(child, res)
            res[child.scope] = resi[child.scope]
        return res
    elif node.type == 'G':
        res[node.scope] = sample_trunc_phi(node.mean, node.std, node.a, node.b)
        return res
    elif node.type == 'M':
        res[node.scope] = categorical(node.p)[0]
        return res
    return res


def sample_conditional(node, evi):
    """
        Returns samples from the distribution defined by the PC given the
        evidence `evi`.

        Parameters
        ----------
        evi: numpy array (n_samples, n_variables)
            Non-observed variables should be set to numpy.nan.
    """
    if evi.ndim == 1:
        evi_local = np.expand_dims(evi, axis=0).copy()
    else:
        evi_local = evi.copy()
    for i in range(evi_local.shape[0]):
        evi_i = evi_local[i, :]
        set_tempw(node, evi_i.reshape(1, -1))
        sample_conditional_aux(node, evi_i)
    return evi_local


def sample_conditional_aux(node, evi):
    """
        Returns one sample from the distribution defined by the PC rooted at
        node, given the evidence in `evi`.
    """
    if node.type == 'S':
        index = categorical(node.tempw)[0]
        node.set_tempw(node.w)  # reset tempw
        return sample_conditional_aux(node.children[index], evi)
    elif node.type == 'P':
        for child in node.children:
            evi_i = sample_conditional_aux(child, evi)
            evi[child.scope] = evi_i[child.scope]
        return evi
    elif node.type == 'G':
        if np.isnan(evi[node.scope]):
            evi[node.scope] = sample_trunc_phi(node.mean, node.std, node.a, node.b)
        return evi
    elif node.type == 'M':
        if np.isnan(evi[node.scope]):
            evi[node.scope] = categorical(node.p)[0]
        return evi
    return evi


def set_tempw(node, evi):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        The temporary weights `tempw` of the sum nodes are updated to reflect
        the probability of the evidence `evi`.
    """
    eps = 1e-6
    if node.type == 'L':
        if node.comparison == 0:  # IN
            return eval_in(node.scope, node.value.astype(np.int64), evi)
        elif node.comparison == 1:  # EQ
            return eval_eq(node.scope, node.value.astype(np.float64), evi)
        elif node.comparison == 3:  # LEQ
            return eval_leq(node.scope, node.value, evi)
        elif node.comparison == 4:  # G
            return eval_g(node.scope, node.value, evi)
    elif node.type == 'M':
        return eval_m(node, evi)
    elif node.type == 'G':
        return eval_gaussian(node, evi)
    elif node.type == 'U':
        return np.ones(evi.shape[0]) * node.value
    elif node.type == 'P':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        logprs[:, 0] = evaluate(node.children[0], evi)
        nonzero = np.where(logprs[:, 0] != -np.Inf)[0]
        if len(nonzero) > 0:
            for i in nb.prange(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[nonzero, i] = set_tempw(node.children[i], evi[nonzero, :])
        return np.sum(logprs, axis=1)
    elif node.type == 'S':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        for i in nb.prange(node.nchildren):
            logprs[:, i] = set_tempw(node.children[i], evi) + node.logw[i]
        res = logsumexp2(logprs, axis=1)
        probs = np.exp(logprs - res)[0]
        node.set_tempw(probs)
        return res
    return np.zeros(evi.shape[0])
