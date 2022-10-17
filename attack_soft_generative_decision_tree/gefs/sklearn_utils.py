import numpy as np
import torch
from sklearn.tree import _tree
from sklearn.ensemble._forest import (_generate_sample_indices,
                                      _generate_unsampled_indices,
                                      _get_n_samples_bootstrap)

from .learning import LearnSPN, fit
from .nodes import (SumNode, ProdNode, Leaf, GaussianLeaf, MultinomialLeaf, Gate_L, Gate_R,
                   UniformLeaf, fit_multinomial, fit_multinomial_with_counts,
                   fit_gaussian)
from .pc import PC
from .utils import bincount


def calc_inbag(n_samples, rf):
    """
        Recovers samples used to create trees in scikit-learn RandomForest objects.

        See https://github.com/scikit-learn-contrib/forest-confidence-interval

        Parameters
        ----------
        n_samples : int
            The number of samples used to fit the scikit-learn RandomForest object.
        forest : RandomForest
            Regressor or Classifier object that is already fit by scikit-learn.

        Returns
        -------
        sample_idx: list
            The indices of the samples used to train each tree.
    """
    # the default sampling mehtod for each tree is bootstrap
    assert rf.bootstrap == True, "Forest was not trained with bootstrapping."

    n_trees = rf.n_estimators
    sample_idx = []
    
    # get the size of dataset after bootstrap
    n_samples_bootstrap = _get_n_samples_bootstrap(
        n_samples, rf.max_samples
    )
    
    # get indices of samples for each decision tree
    for t_idx in range(n_trees):
        sample_idx.append(
            _generate_sample_indices(rf.estimators_[t_idx].random_state,
                                     n_samples, n_samples_bootstrap))
        
    return sample_idx


def calc_outofbag(n_samples, rf):
    """
        Recovers samples used to create trees in scikit-learn RandomForest objects.

        See https://github.com/scikit-learn-contrib/forest-confidence-interval

        Parameters
        ----------
        n_samples : int
            The number of samples used to fit the scikit-learn RandomForest object.
        forest : RandomForest
            Regressor or Classifier object that is already fit by scikit-learn.

        Returns
        -------
        sample_idx: list
            The indices of the samples used to train each tree.
    """

    assert rf.bootstrap == True, "Forest was not trained with bootstrapping."

    n_trees = rf.n_estimators
    sample_idx = []
    n_samples_bootstrap = _get_n_samples_bootstrap(
        n_samples, rf.max_samples
    )

    for t_idx in range(n_trees):
        sample_idx.append(
            _generate_unsampled_indices(rf.estimators_[t_idx].random_state,
                                        n_samples, n_samples_bootstrap))
    return sample_idx


def tree2pc(tree, X, y, ncat, learnspn=np.Inf, max_height=100000,
            thr=0.01, minstd=1, smoothing=1e-6, return_pc=True):
    """
        Parses a sklearn DecisionTreeClassifier to a Generative Decision Tree.
        Note that X, y do not need to match the data used to train the
        decision tree exactly. However, if they do not match you might get
        branches of the tree with no data, and hence poor models of the
        distribution at the leaves.

        Parameters
        ----------
        tree: DecisionTreeClassifier
        X: numpy array
            Explanatory variables.
        y: numpy array
            Target variable.
        ncat: numpy array (int64)
            The number of categories for each variable. 1 for continuous variables.
        learnsnp: int
            The number of samples (at a given leaf) required to run LearnSPN.
            Set to infinity by default, so as not to run LearnSPN anywhere.
        max_height: int
            Maximum height (depth) of the LearnSPN models at the leaves.
        thr: float
            p-value threshold for independence tests in product nodes.
        return_pc: boolean
            If True returns a PC object, if False returns a Node object (root).
        minstd: float
            The minimum standard deviation of gaussian leaves.
        smoothing: float
            Additive smoothing (Laplace smoothing) for categorical data.

    https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    """

    # used to store soft GeDT parameters
    gate_weights = []
    gate_split_values = []
    sum_weights = []

    # used to mark whether the input is a torch tensor
    torch_flag = False

    # turn the input from tensor to numpy and set the flag to be true
    if type(X) == torch.Tensor:
        X = X.numpy()
        y = y.numpy()
        torch_flag = True

    # the scope include the label, the initial scope is the index of all features and labels
    scope = np.array([i for i in range(X.shape[1]+1)]).astype('int64')

    # combine features X and label y into data
    data = np.concatenate([X, np.expand_dims(y, axis=1)], axis=1)
    
    # LaPlace smoothing term
    lp = np.sum(np.where(ncat==1, 0, ncat)) * smoothing # LaPlace counts

    # recursively parse decision tree nodes to PC nodes.
    def recurse(node, node_ind, depth, data, upper, lower):
        # If split node 
        if tree_.feature[node_ind] != _tree.TREE_UNDEFINED:
            
            # get the feature of the split node
            split_var = feature_name[node_ind]
            
            # get the threshold of the split node
            split_value = np.array([tree_.threshold[node_ind]], dtype=np.float64)
            
            # create sumnode, n is the number of instances with Laplace smoothing
            sumnode = SumNode(scope=scope, n=data.shape[0]+lp)

            # avoid root node, then other nodes are product nodes
            if node is not None:
                node.add_child(sumnode)



            # Parse left node <=
            upper1 = upper.copy()
            lower1 = lower.copy()

            # set the upper bound of split variable
            # for the left node, for both continuous and categorical data, the upper bound changes to the split value
            # in some cases, the original upper bound is smaller than the split value(categorical data)
            upper1[split_var] = min(split_value, upper1[split_var])
            
            # get the instances for the left children
            split1 = data[np.where(data[:, split_var] <= split_value)]
            
            # the child of a sum node is a product node
            p1 = ProdNode(scope=scope, n=split1.shape[0]+lp)
            
            # add product node as child, this is the product of a condition and a sumnode
            if torch_flag:
                sumnode.add_child_torch(p1)
            else:
                sumnode.add_child(p1)

            # condition node on the split feature, 'comparison 3' means 'less or equal to'
            # the scope here now shrink to a single feature
            if torch_flag:
                ind1 = Gate_L(scope=np.array([split_var], dtype='int64'), n=split1.shape[0] + lp, value=split_value,
                            comparison=3)  # Comparison <=
                gate_weights.append(ind1.gate_weight)
                gate_split_values.append(ind1.gate_split_value)
            else:
                ind1 = Leaf(scope=np.array([split_var], dtype='int64'), n=split1.shape[0] + lp, value=split_value,
                            comparison=3)  # Comparison <=

            # add indicator to the left product node
            p1.add_child(ind1)
            
            # recurse to the product node
            recurse(p1, tree_.children_left[node_ind], depth + 1, split1.copy(), upper1, lower1)



            # Parse right node >
            upper2 = upper.copy()
            lower2 = lower.copy()

            lower2[split_var] = max(split_value, lower2[split_var])
            split2 = data[np.where(data[:, split_var] > split_value)]
            p2 = ProdNode(scope=scope, n=split2.shape[0]+lp)
            if torch_flag:
                sumnode.add_child_torch(p2)
            else:
                sumnode.add_child(p2)

            # the sum weights are replaced by theta, which is a re-parameterization trick
            # please keep the position of this line, this line of code can only appear after sumnode.add_child(p2)
            sum_weights.append(sumnode.theta)

            if torch_flag:
                ind2 = Gate_R(scope=np.array([split_var],dtype='int64'), n=split2.shape[0]+lp,
                              value=ind1.gate_split_value, comparison=4,weight=ind1.gate_weight)  # Comparison >
            else:
                ind2 = Leaf(scope=np.array([split_var],dtype='int64'), n=split2.shape[0]+lp,
                            value=split_value, comparison=4)  # Comparison >

            p2.add_child(ind2)
            recurse(p2, tree_.children_right[node_ind], depth + 1, split2.copy(), upper2, lower2)

            return sumnode
        # Leaf node
        else:
            assert node is not None, "Tree has no splits."
            
            # if GeDT has enough sample then create SPN leaf node
            if data.shape[0] >= learnspn:
                learner = LearnSPN(ncat, thr, 2, max_height, None)
                
                fit(learner, data, node)
                
            # if GeDT does not have enough sample
            else:
                for var in scope:
                    if ncat[var] > 1:  # Categorical variable is parsed into a multinomial leaf
                        leaf = MultinomialLeaf(scope=np.array([var],dtype='int64'), n=data.shape[0]+lp)
                        node.add_child(leaf)
                        fit_multinomial(leaf, data, int(ncat[var]), smoothing)
                    else:  # Continuous variable is parsed into a gaussian leaf
                        leaf = GaussianLeaf(scope=np.array([var], dtype='int64'), n=data.shape[0]+lp)
                        node.add_child(leaf)
                        fit_gaussian(leaf, data, upper[var], lower[var], minstd)
                return None

    # upper bound for categorical data
    upper = ncat.copy().astype(float)

    # lower bound for categorical data
    lower = ncat.copy().astype(float)

    # upper bound for continuous data
    upper[upper == 1] = np.Inf
    
    # lower bound for continuous data
    lower[ncat == 1] = -np.Inf

    # rename sklearn tree.tree_ as tree_
    tree_ = tree.tree_

    # check whether it is split node or leaf node
    # get feature name for each node
    feature_name = [
        i if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    root = recurse(None, 0, 1, data, upper, lower)
    
    if return_pc:
        pc = PC(ncat)
        pc.root = root

        # pass soft GeDT parameters to root pc
        pc.gate_weights = gate_weights
        pc.gate_split_values = gate_split_values
        pc.sum_weights = sum_weights
        return pc

    return root


def rf2pc(rf, X_train, y_train, ncat, learnspn=np.Inf, max_height=10000,
          thr=0.01, minstd=1, smoothing=1e-6):
    """
        Parses a sklearn RandomForestClassifier to a Generative Forest.
        Note that X, y do not need to match the data used to train the
        decision tree exactly. However, if they do not match you might get
        branches of the tree with no data, and hence poor models of the
        distribution at the leaves.

        Parameters
        ----------
        tree: DecisionTreeClassifier
        X: numpy array
            Explanatory variables.
        y: numpy array
            Target variable.
        ncat: numpy array (int64)
            The number of categories for each variable. 1 for continuous variables.
        learnsnp: int
            The number of samples (at a given leaf) required to run LearnSPN.
            Set to infinity by default, so as not to run LearnSPN anywhere.
        max_height: int
            Maximum height (depth) of the LearnSPN models at the leaves.
        thr: float
            p-value threshold for independence tests in product nodes.
        minstd: float
            The minimum standard deviation of gaussian leaves.
        smoothing: float
            Additive smoothing (Laplace smoothing) for categorical data.
    """

    # get the scope, including all variables
    scope = np.arange(len(ncat)).astype('int64')
    # get indices of samples for each decision tree
    sample_idx = calc_inbag(X_train.shape[0], rf)

    # flag used to test whether input is a pytorch tensor
    torch_flag = False
    if type(X_train) == torch.Tensor:
        torch_flag = True

    pc = PC(ncat)
    
    # the root node that weighted average the output of each tree
    pc.root = SumNode(scope=scope, n=1)
    
    # turn each decision tree into a generative decision tree
    for i, tree in enumerate(rf.estimators_):
        # get the training data for a decision tree
        X_tree = X_train[sample_idx[i], :]
        y_tree = y_train[sample_idx[i]]
        si = tree2pc(tree, X_tree, y_tree, ncat, learnspn, max_height,
                         thr, minstd, smoothing, return_pc=False)

        if torch_flag:
            pc.root.add_child_torch(si)
        else:
            pc.root.add_child(si)
    
    # set flag to be true
    pc.is_ensemble = True
    return pc
