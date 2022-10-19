import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats


def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, L1_REG=None, randomize=False, omega=None):
    """ Construct PGD adversarial examples on the examples X

        Parameters:
        ----------
        model: gefs.pc.PC
            Soft generative decision tree
        X: numpy array
            The features
        y: numpy array
            The classes
        epsilon: float
            epsilon parameter for epsilon ball of PGD attack
        alpha: float
            learning rate
        num_iter: int
            number of iterations
        L1_REG:
            The weight of L1 regularization
        randomize: boolean
            Whether generate perturbation randomly
        omega: pytorch tensor
            Used for feature importance
    """
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    if omega != None:
        for t in range(num_iter):
            pred, prob = model.classify(X + delta, return_prob=True)
            if L1_REG != None:
                loss = F.nll_loss(torch.log(prob), y.type(torch.int64), reduction='sum') \
                       + L1_REG * (sum([torch.abs(x) for x in model.gate_weights])) + (
                           sum(sum([torch.abs(x) for x in model.sum_weights])))
            else:
                loss = F.nll_loss(torch.log(prob), y.type(torch.int64), reduction='sum')
            loss.backward(retain_graph=True)
            delta.data = torch.clamp((omega * (delta + alpha * delta.grad.detach().sign())), -epsilon, epsilon) / omega
            delta.grad.zero_()
    else:
        for t in range(num_iter):
            pred, prob = model.classify(X + delta, return_prob=True)
            loss = F.nll_loss(torch.log(prob), y.type(torch.int64), reduction='sum')
            loss.backward(retain_graph=True)
            delta.data = torch.clamp((delta + alpha * delta.grad.detach().sign()), -epsilon, epsilon)
            delta.grad.zero_()

    return delta.detach()


def epoch_adversarial_training(loader, model, attack, return_adv=False, L1_REG=None, non_editable_vector=None,
                               clipping_max=None, clipping_min=None, datatypes=None, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset, designed for soft GeDT

    Parameters
    ----------
    loader: torch.utils.data.dataloader.DataLoader
        Pytorch dataset loader
    model: gefs.pc.PC
        Soft generative decision tree
    attack:  function
        adversarial attack method
    return_adv: boolean
        Whether return adversarial examples
    L1_REG: float
        The weight of L1 regularization
    non_editable_vector: numpy array
        editability vector
    clipping_max: int or float
        Upper bound for clipping
    clipping_min: int or float
        Lower bound for clipping
    datatypes: list
        list of strings representing data types
    opt: torch.optim.adam.Adam
        pytorch optimizer
    **kwargs:
        other keywords for adversarial attacks

    https://adversarial-ml-tutorial.org/adversarial_training/
    """
    total_loss, total_err = 0., 0.

    if return_adv:
        X_advs = []
        y_advs = []

    for X, y in loader:
        # calculate perturbation using adversarial attack
        delta = attack(model, X, y, **kwargs)

        # add perturbation to generate adversarial data
        if non_editable_vector != None:
            # non-editable vector
            X_adv = X + delta * non_editable_vector
        else:
            X_adv = X + delta

        # clipping
        if clipping_max != None and clipping_min != None:
            for i in range(X_adv.shape[1]):
                X_adv[:, i] = torch.clamp(X_adv[:, i], clipping_min[i], clipping_max[i])

        # correction of data types
        if datatypes != None:
            for i in range(X_adv.shape[1]):
                if datatypes[i] == 'pos_int':
                    X_adv[:, i] = torch.round(X_adv[:, i])
                    X_adv[:, i] = torch.clamp(X_adv[:, i], min=0)
                elif datatypes[i] == 'int':
                    X_adv[:, i] = torch.round(X_adv[:, i])
                elif datatypes[i] == 'pos_float':
                    X_adv[:, i] = torch.clamp(X_adv[:, i], min=0)
                elif datatypes[i] == 'bool':
                    X_adv[X_adv[:, i] <= 0.5, i] = 0
                    X_adv[X_adv[:, i] > 0.5, i] = 1

        pred, prob = model.classify(X_adv, return_prob=True)

        # L1 regularization
        if L1_REG != None:
            loss = F.nll_loss(torch.log(prob), y.type(torch.int64), reduction='sum') \
                   + L1_REG * (sum([torch.abs(x) for x in model.gate_weights])) + (
                       sum(sum([torch.abs(x) for x in model.sum_weights])))
        else:
            loss = F.nll_loss(torch.log(prob), y.type(torch.int64), reduction='sum')

        if return_adv:
            X_advs.append(X_adv)
            y_advs.append(y)

        # if no optimizer is found, no adversarial training will be performed but only adversarial attack
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (pred != y).sum().item()

        # print(sum([torch.abs(x) for x in model.sum_weights]),loss-sum([torch.abs(x) for x in model.sum_weights]))
        total_loss += loss.item()
    if return_adv:
        return total_err / len(loader.dataset), total_loss / len(loader.dataset), X_advs, y_advs
    else:
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def pgd_linf_nn(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False, omega=None):
    """ Construct PGD adversarial examples on the examples X

        Parameters:
        ----------
        model: gefs.pc.PC
            Soft generative decision tree
        X: numpy array
            The features
        y: numpy array
            The classes
        epsilon: float
            epsilon parameter for epsilon ball of PGD attack
        alpha: float
            learning rate
        num_iter: int
            number of iterations
        L1_REG:
            The weight of L1 regularization
        randomize: boolean
            Whether generate perturbation randomly
        omega: pytorch tensor
            Used for feature importance
    """
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    if omega != None:
        for t in range(num_iter):
            y_pred = model(X + delta)
            loss = F.cross_entropy(y_pred, y.long(), reduction='sum')
            loss.backward()
            delta.data = torch.clamp((omega * (delta + alpha * delta.grad.detach().sign())), -epsilon, epsilon) / omega
            delta.grad.zero_()
    else:
        for t in range(num_iter):
            y_pred = model(X + delta)
            loss = F.cross_entropy(y_pred, y.long(), reduction='sum')
            loss.backward()
            delta.data = torch.clamp((delta + alpha * delta.grad.detach().sign()), -epsilon, epsilon)
            delta.grad.zero_()
    return delta.detach()


def epoch_adversarial_training_nn(loader, model, attack, opt=None, non_editable_vector=None, clipping_max=None,
                                  clipping_min=None, datatypes=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset, designed for neural networks

        Parameters
        ----------
        loader: torch.utils.data.dataloader.DataLoader
          Pytorch dataset loader
        model: gefs.pc.PC
          Soft generative decision tree
        attack:  function
          adversarial attack method
        return_adv: boolean
          Whether return adversarial examples
        L1_REG: float
          L1 regularization
        non_editable_vector: numpy array
          editability vector
        clipping_max: int or float
          Upper bound for clipping
        clipping_min: int or float
          Lower bound for clipping
        datatypes: list
          list of strings representing data types
        opt: torch.optim.adam.Adam
          pytorch optimizer
        **kwargs:
          other keywords for adversarial attacks
        https://adversarial-ml-tutorial.org/adversarial_training/
    """
    total_loss, total_err = 0., 0.

    for X, y in loader:
        delta = attack(model, X, y, **kwargs)
        if non_editable_vector != None:
            # non-editable vector
            X_adv = X + delta * non_editable_vector
        else:
            X_adv = X + delta
        # clipping
        if clipping_max != None and clipping_min != None:
            for i in range(X_adv.shape[1]):
                X_adv[:, i] = torch.clamp(X_adv[:, i], clipping_min[i], clipping_max[i])

        # correction of datatypes
        if datatypes != None:
            for i in range(X_adv.shape[1]):
                if datatypes[i] == 'pos_int':
                    X_adv[:, i] = torch.round(X_adv[:, i])
                    X_adv[:, i] = torch.clamp(X_adv[:, i], min=0)
                elif datatypes[i] == 'int':
                    X_adv[:, i] = torch.round(X_adv[:, i])
                elif datatypes[i] == 'pos_float':
                    X_adv[:, i] = torch.clamp(X_adv[:, i], min=0)
                elif datatypes[i] == 'bool':
                    X_adv[X_adv[:, i] <= 0.5, i] = 0
                    X_adv[X_adv[:, i] > 0.5, i] = 1

        y_pred = model(X_adv)

        loss = F.cross_entropy(y_pred, y.long(), reduction='sum')
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (torch.argmax(y_pred, dim=1) != y).sum().item()
        total_loss += loss.item()

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def mean_confidence_interval(data, confidence=0.95):
    """
    This function is used to calculate confidence interval of multiple experimental results

    :param data: list
        list of numbers
    :param confidence:
        set the confidence
    :return:
        m: the mean value
        h: quantile of t-distribution
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h
