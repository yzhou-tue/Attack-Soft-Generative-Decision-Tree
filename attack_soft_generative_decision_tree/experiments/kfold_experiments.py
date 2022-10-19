import numpy as np
import shap
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.tree import DecisionTreeClassifier
from gefs.sklearn_utils import tree2pc
from prep import get_stats, standardize_data
from sklearn.model_selection import KFold
from gefs.adversarial import pgd_linf, epoch_adversarial_training, mean_confidence_interval, \
    epoch_adversarial_training_nn, pgd_linf_nn
import math


def kfold_normal_training_softgedt(filename, data, ncat, non_editable_vector, datatypes, include_sum_weight=True):
    """
    This function perform 5 experiments and 5-fold cross validation to test the accuracy and robustness of soft GeDT
    before adversarial training

    :param filename: filename for storing parameters of model
    :param data: input data
    :param ncat: number of categories
    :param non_editable_vector: editability vector
    :param datatypes: data types of features
    :param include_sum_weight: whether include weights of sum nodes in the optimizer
    """
    experiments_accs = []
    experiments_adt_errs = []
    experiments_hard_accs = []
    experiments_hard_adt_errs = []
    L1_REG = 0.1
    for i in range(5):
        print('experiment:', i)
        # K-fold parameters
        K_FOLDS = 5
        kfold = KFold(n_splits=K_FOLDS, shuffle=True)

        # counters
        acc_fold_train = [0 for i in range(K_FOLDS)]
        acc_fold_valid = [0 for i in range(K_FOLDS)]
        acc_fold_test = [0 for i in range(K_FOLDS)]
        acc_fold_origin = [0 for i in range(K_FOLDS)]
        acc_fold_origin_hard = [0 for i in range(K_FOLDS)]
        err_fold_adt = [0 for i in range(K_FOLDS)]
        err_fold_hard_adt = [0 for i in range(K_FOLDS)]

        training_fold_num_correct = [0 for i in range(K_FOLDS)]
        training_fold_num_total = [0 for i in range(K_FOLDS)]
        validation_fold_num_correct = [0 for i in range(K_FOLDS)]
        validation_fold_num_total = [0 for i in range(K_FOLDS)]
        test_fold_num_correct = [0 for i in range(K_FOLDS)]
        test_fold_num_total = [0 for i in range(K_FOLDS)]

        for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
            # split dataset for each fold
            print("Fold:", fold)
            shuffle = np.random.choice(train_ids, train_ids.shape[0], replace=False)
            valid_ids = shuffle[int(train_ids.shape[0] * 0.75):]
            train_ids = shuffle[:int(train_ids.shape[0] * 0.75)]

            # standardize dataset
            _, maxv, minv, mean, std = get_stats(data[train_ids, :], ncat)
            data_train_tree = standardize_data(data[train_ids, :], mean, std)
            data_valid_tree = standardize_data(data[valid_ids, :], mean, std)
            data_test_tree = standardize_data(data[test_ids, :], mean, std)

            data_train_tree = torch.from_numpy(data_train_tree)
            data_valid_tree = torch.from_numpy(data_valid_tree)
            data_test_tree = torch.from_numpy(data_test_tree)

            data_train_tree = torch.utils.data.TensorDataset(data_train_tree[:, :-1], data_train_tree[:, -1])
            data_valid_tree = torch.utils.data.TensorDataset(data_valid_tree[:, :-1], data_valid_tree[:, -1])
            data_test_tree = torch.utils.data.TensorDataset(data_test_tree[:, :-1], data_test_tree[:, -1])

            trainloader = torch.utils.data.DataLoader(data_train_tree, batch_size=32, shuffle=True)
            validloader = torch.utils.data.DataLoader(data_valid_tree, batch_size=32, shuffle=True)
            testloader = torch.utils.data.DataLoader(data_test_tree, batch_size=32, shuffle=True)

            # create model
            clf = DecisionTreeClassifier()
            clf.fit(data_train_tree[:][0].numpy(), data_train_tree[:][1].numpy())
            SoftGeDT = tree2pc(clf, data_train_tree[:][0], data_train_tree[:][1], ncat, learnspn=30, minstd=.1,
                               smoothing=0.1)
            HardGeDT = tree2pc(clf, data_train_tree[:][0], data_train_tree[:][1], ncat, learnspn=30, minstd=.1,
                               smoothing=0.1)
            for k in range(len(HardGeDT.gate_weights)):
                HardGeDT.gate_weights[k].data = torch.tensor([-np.inf])

            # optimization set up
            if include_sum_weight:
                opt = torch.optim.Adam(SoftGeDT.gate_weights + SoftGeDT.gate_split_values + SoftGeDT.sum_weights,
                                       lr=0.1)
            else:
                opt = torch.optim.Adam(SoftGeDT.gate_weights + SoftGeDT.gate_split_values,
                                       lr=0.1)

            EPOCHS = 10

            # early stopping parameter
            patience = 4
            trigger_times = 0

            # original prediction
            pred = SoftGeDT.classify(data_test_tree[:][0])
            acc_fold_origin[fold] = (torch.sum(pred == data_test_tree[:][1]) / data_test_tree[:][1].shape[0])

            pred = HardGeDT.classify(data_test_tree[:][0])
            acc_fold_origin_hard[fold] = (torch.sum(pred == data_test_tree[:][1]) / data_test_tree[:][1].shape[0])

            # counters
            acc_epoch_train = [0 for i in range(EPOCHS)]
            acc_epoch_valid = [0 for i in range(EPOCHS)]
            training_epoch_num_correct = [0 for i in range(EPOCHS)]
            training_epoch_num_total = [0 for i in range(EPOCHS)]
            validation_epoch_num_correct = [0 for i in range(EPOCHS)]
            validation_epoch_num_total = [0 for i in range(EPOCHS)]

            torch.save(SoftGeDT.gate_weights, f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{-1}_weights')
            torch.save(SoftGeDT.gate_split_values,
                       f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{-1}_split_values')
            torch.save(SoftGeDT.sum_weights,
                       f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{-1}_sum_weights')

            # shap values
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(data_test_tree[:][0].numpy())
            shap_values = np.array(shap_values)
            shap_values = np.mean(np.mean(np.abs(shap_values), axis=1), axis=0)
            if 0 in shap_values:
                omega = torch.from_numpy((1 / (shap_values + 10e-2))) / np.linalg.norm((1 / (shap_values + 10e-2)), 2)
            else:
                omega = torch.from_numpy((1 / shap_values)) / np.linalg.norm((1 / shap_values), 2)

            # clipping values
            clipping_max = torch.max(data_train_tree[:][0], dim=0).values
            clipping_min = torch.min(data_train_tree[:][0], dim=0).values

            for i in range(EPOCHS):
                print("epoch:", i)

                for X, y in trainloader:
                    opt.zero_grad()
                    pred, prob = SoftGeDT.classify(X, return_prob=True)
                    loss = F.nll_loss(torch.log(prob), y.type(torch.int64), reduction='sum')
                    loss.backward()
                    opt.step()

                    training_epoch_num_correct[i] += torch.sum(pred == y)
                    training_epoch_num_total[i] += y.shape[0]

                acc_epoch_train[i] = training_epoch_num_correct[i] / training_epoch_num_total[i]

                with torch.no_grad():
                    for X, y in validloader:
                        pred, prob = SoftGeDT.classify(X, return_prob=True)
                        validation_epoch_num_correct[i] += torch.sum(pred == y)
                        validation_epoch_num_total[i] += y.shape[0]

                acc_epoch_valid[i] = validation_epoch_num_correct[i] / validation_epoch_num_total[i]
                print('train acc:', acc_epoch_train[i], 'valid acc:', acc_epoch_valid[i], "trigger times:",
                      trigger_times)

                torch.save(SoftGeDT.gate_weights,
                           f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{i}_weights')
                torch.save(SoftGeDT.gate_split_values,
                           f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{i}_split_values')
                torch.save(SoftGeDT.sum_weights,
                           f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{i}_sum_weights')

                # Prevent Sudden Drop
                if math.isnan(loss):
                    # restore previous model
                    weights = torch.load(f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{i - 1}_weights')
                    split_values = torch.load(
                        f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{i - 1}_split_values')
                    sum_weights = torch.load(
                        f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{i - 1}_sum_weights')
                    for k in range(len(SoftGeDT.gate_weights)):
                        SoftGeDT.gate_weights[k].data = weights[k]
                        SoftGeDT.gate_split_values[k].data = split_values[k]

                    for k in range(len(SoftGeDT.sum_weights)):
                        SoftGeDT.sum_weights[k].data = sum_weights[k]

                    break

                # Early stopping during epoch
                if i > 0 and acc_epoch_valid[i] <= acc_epoch_valid[i - 1]:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print("Early Stopping")
                        training_fold_num_correct[fold] = training_epoch_num_correct[i - 1]
                        training_fold_num_total[fold] = training_epoch_num_total[i - 1]
                        validation_fold_num_correct[fold] = validation_epoch_num_correct[i - 1]
                        validation_fold_num_total[fold] = validation_epoch_num_total[i - 1]
                        acc_fold_train[fold] = acc_epoch_train[i - 1]
                        acc_fold_valid[fold] = acc_epoch_valid[i - 1]
                        acc_epoch_train[i] = 0
                        acc_epoch_valid[i] = 0
                        # load params from previous epoch
                        weights = torch.load(
                            f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{i - 1}_weights')
                        split_values = torch.load(
                            f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{i - 1}_split_values')
                        sum_weights = torch.load(
                            f'{filename}_params_state_dict/softgedt_normal_cv_{fold}_epoch{i - 1}_sum_weights')

                        for k in range(len(SoftGeDT.gate_weights)):
                            SoftGeDT.gate_weights[k].data = weights[k]
                            SoftGeDT.gate_split_values[k].data = split_values[k]

                        for k in range(len(SoftGeDT.sum_weights)):
                            SoftGeDT.sum_weights[k].data = sum_weights[k]

                        break
                else:
                    trigger_times = 0

                if i == (EPOCHS - 1):
                    training_fold_num_correct[fold] = training_epoch_num_correct[i]
                    training_fold_num_total[fold] = training_epoch_num_total[i]
                    validation_fold_num_correct[fold] = validation_epoch_num_correct[i]
                    validation_fold_num_total[fold] = validation_epoch_num_total[i]
                    acc_fold_train[fold] = acc_epoch_train[i]
                    acc_fold_valid[fold] = acc_epoch_valid[i]

            # test performance
            with torch.no_grad():
                for X, y in testloader:
                    pred, prob = SoftGeDT.classify(X, return_prob=True)
                    test_fold_num_correct[fold] += torch.sum(pred == y)
                    test_fold_num_total[fold] += y.shape[0]

            # test adversarial accuracy
            err_fold_adt[fold], test_total_loss, X_advs, y_advs = epoch_adversarial_training(testloader, SoftGeDT,
                                                                                             pgd_linf,
                                                                                             return_adv=True,
                                                                                             non_editable_vector=non_editable_vector,
                                                                                             clipping_max=clipping_max,
                                                                                             clipping_min=clipping_min,
                                                                                             datatypes=datatypes,
                                                                                             num_iter=10, epsilon=0.1,
                                                                                             L1_REG=L1_REG, omega=omega)

            X_adv = X_advs[0]
            y_adv = y_advs[0]
            for i in range(1, len(X_advs)):
                X_adv = torch.cat((X_adv, X_advs[i]), dim=0)
                y_adv = torch.cat((y_adv, y_advs[i]), dim=0)

            pred, prob = HardGeDT.classify(X_adv, return_prob=True)
            err_fold_hard_adt[fold] = torch.sum(y_adv != pred) / y_adv.shape[0]

            acc_fold_test[fold] = test_fold_num_correct[fold] / test_fold_num_total[fold]

            print("SoftGeDT Accuracy on Training set:", acc_fold_train[fold])
            print("SoftGeDT Accuracy on Validation set:", acc_fold_valid[fold])

            print("SoftGeDT Accuracy on Test set:", acc_fold_test[fold])
            print("Average Accuracy on Test set:", sum(test_fold_num_correct) / sum(test_fold_num_total))

            print("Original Accuracy on Test set:", acc_fold_origin[fold])
            print("Average Original Accuracy on Test set:", sum(acc_fold_origin) / (fold + 1))

            print("Original HardGeDT Accuracy on Test set:", acc_fold_origin_hard[fold])
            print("Average Original HardGeDT Accuracy on Test set:", sum(acc_fold_origin_hard) / (fold + 1))

            print("ADT error, epsilon:0.1:", err_fold_adt[fold])
            print("Average ADT error, epsilon:0.1", sum(err_fold_adt) / (fold + 1))

            print("HardGeDT Adversarial Error:", err_fold_hard_adt[fold])
            print("HardGeDT Average Adversarial Error:", sum(err_fold_hard_adt) / (fold + 1))

        experiments_accs.append(sum(test_fold_num_correct) / sum(test_fold_num_total))
        experiments_adt_errs.append(sum(err_fold_adt) / (fold + 1))
        experiments_hard_accs.append(sum(acc_fold_origin_hard) / (fold + 1))
        experiments_hard_adt_errs.append(sum(err_fold_hard_adt) / (fold + 1))

    acc_mean, acc_radius = mean_confidence_interval(experiments_accs)
    err_mean, err_radius = mean_confidence_interval(experiments_adt_errs)
    acc_hard_mean, acc_hard_radius = mean_confidence_interval(experiments_hard_accs)
    err_hard_mean, err_hard_radius = mean_confidence_interval(experiments_hard_adt_errs)

    print('Experiments mean accuracy:', acc_mean, 'radius:', acc_radius)
    print('Experiments mean adt err', err_mean, 'radius', err_radius)
    print('Experiments mean accuracy(hard gedt):', acc_hard_mean, 'radius:', acc_hard_radius)
    print('Experiments mean adt err(hard gedt)', err_hard_mean, 'radius', err_hard_radius)


def kfold_adversarial_training_softgedt(filename, data, ncat, non_editable_vector, datatypes, include_sum_weight=True):
    """
    This function perform 5 experiments and 5-fold cross validation to test the accuracy and robustness of soft GeDT
    after adversarial training

    :param filename: filename for storing parameters of model
    :param data: input data
    :param ncat: number of categories
    :param non_editable_vector: editability vector
    :param datatypes: data types of features
    :param include_sum_weight: whether include weights of sum nodes in the optimizer
    """

    experiments_accs = []
    experiments_adt_errs = []
    L1_REG = 0.5
    for i in range(5):
        # K-fold parameters
        K_FOLDS = 5
        kfold = KFold(n_splits=K_FOLDS, shuffle=True)

        # counters
        err_fold_train = [0 for i in range(K_FOLDS)]
        err_fold_valid = [0 for i in range(K_FOLDS)]
        err_fold_test = [0 for i in range(K_FOLDS)]

        acc_fold_test = [0 for i in range(K_FOLDS)]
        test_fold_num_correct = [0 for i in range(K_FOLDS)]
        test_fold_num_total = [0 for i in range(K_FOLDS)]

        for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
            # split dataset for each fold
            print("Fold:", fold)
            shuffle = np.random.choice(train_ids, train_ids.shape[0], replace=False)
            valid_ids = shuffle[int(train_ids.shape[0] * 0.75):]
            train_ids = shuffle[:int(train_ids.shape[0] * 0.75)]

            # standardize dataset
            _, maxv, minv, mean, std = get_stats(data[train_ids, :], ncat)
            data_train_tree = standardize_data(data[train_ids, :], mean, std)
            data_valid_tree = standardize_data(data[valid_ids, :], mean, std)
            data_test_tree = standardize_data(data[test_ids, :], mean, std)

            data_train_tree = torch.from_numpy(data_train_tree)
            data_valid_tree = torch.from_numpy(data_valid_tree)
            data_test_tree = torch.from_numpy(data_test_tree)

            data_train_tree = torch.utils.data.TensorDataset(data_train_tree[:, :-1], data_train_tree[:, -1])
            data_valid_tree = torch.utils.data.TensorDataset(data_valid_tree[:, :-1], data_valid_tree[:, -1])
            data_test_tree = torch.utils.data.TensorDataset(data_test_tree[:, :-1], data_test_tree[:, -1])

            trainloader = torch.utils.data.DataLoader(data_train_tree, batch_size=32, shuffle=True)
            validloader = torch.utils.data.DataLoader(data_valid_tree, batch_size=32, shuffle=True)
            testloader = torch.utils.data.DataLoader(data_test_tree, batch_size=32, shuffle=True)

            # create model
            clf = DecisionTreeClassifier()
            clf.fit(data_train_tree[:][0].numpy(), data_train_tree[:][1].numpy())
            SoftGeDTAdv = tree2pc(clf, data_train_tree[:][0], data_train_tree[:][1], ncat, learnspn=30, minstd=.1,
                                  smoothing=0.1)

            # optimization set up
            if include_sum_weight:
                opt = torch.optim.Adam(
                    SoftGeDTAdv.gate_weights + SoftGeDTAdv.gate_split_values + SoftGeDTAdv.sum_weights,
                    lr=0.1)
            else:
                opt = torch.optim.Adam(SoftGeDTAdv.gate_weights + SoftGeDTAdv.gate_split_values,
                                       lr=0.1)
            EPOCHS = 10

            # early stopping parameter
            patience = 4
            trigger_times = 0

            # counters
            err_epoch_train = [0 for i in range(EPOCHS)]
            err_epoch_valid = [0 for i in range(EPOCHS)]

            torch.save(SoftGeDTAdv.gate_weights, f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{-1}_weights')
            torch.save(SoftGeDTAdv.gate_split_values,
                       f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{-1}_split_values')
            torch.save(SoftGeDTAdv.sum_weights,
                       f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{-1}_sum_weights')

            # shap values
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(data_test_tree[:][0].numpy())
            shap_values = np.array(shap_values)
            shap_values = np.mean(np.mean(np.abs(shap_values), axis=1), axis=0)
            if 0 in shap_values:
                omega = torch.from_numpy((1 / (shap_values + 10e-2))) / np.linalg.norm((1 / (shap_values + 10e-2)), 2)
            else:
                omega = torch.from_numpy((1 / shap_values)) / np.linalg.norm((1 / shap_values), 2)

            # clipping values
            clipping_max = torch.max(data_train_tree[:][0], dim=0).values
            clipping_min = torch.min(data_train_tree[:][0], dim=0).values

            for i in range(EPOCHS):
                print("epoch:", i)

                err_epoch_train[i], train_total_loss = epoch_adversarial_training(trainloader, SoftGeDTAdv, pgd_linf,
                                                                                  opt=opt,
                                                                                  non_editable_vector=non_editable_vector,
                                                                                  clipping_max=clipping_max,
                                                                                  clipping_min=clipping_min,
                                                                                  datatypes=datatypes, num_iter=10,
                                                                                  epsilon=0.1, L1_REG=L1_REG,
                                                                                  omega=omega)

                err_epoch_valid[i], valid_total_loss = epoch_adversarial_training(validloader, SoftGeDTAdv, pgd_linf,
                                                                                  non_editable_vector=non_editable_vector,
                                                                                  clipping_max=clipping_max,
                                                                                  clipping_min=clipping_min,
                                                                                  datatypes=datatypes, num_iter=10,
                                                                                  epsilon=0.1, L1_REG=L1_REG,
                                                                                  omega=omega)
                print("train error:", err_epoch_train[i], "train loss:", train_total_loss, "validation error",
                      err_epoch_valid[i])

                torch.save(SoftGeDTAdv.gate_weights,
                           f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{i}_weights')
                torch.save(SoftGeDTAdv.gate_split_values,
                           f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{i}_split_values')
                torch.save(SoftGeDTAdv.sum_weights,
                           f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{i}_sum_weights')

                # Prevent Sudden Drop
                if math.isnan(train_total_loss):
                    # restore previous model
                    weights = torch.load(f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{i - 1}_weights')
                    split_values = torch.load(
                        f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{i - 1}_split_values')
                    sum_weights = torch.load(
                        f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{i - 1}_sum_weights')
                    for k in range(len(SoftGeDTAdv.gate_weights)):
                        SoftGeDTAdv.gate_weights[k].data = weights[k]
                        SoftGeDTAdv.gate_split_values[k].data = split_values[k]

                    for k in range(len(SoftGeDTAdv.sum_weights)):
                        SoftGeDTAdv.sum_weights[k].data = sum_weights[k]

                    break

                # Early stopping during epoch
                if i > 0 and err_epoch_valid[i] >= err_epoch_valid[i - 1]:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print("Early Stopping")
                        err_fold_train[fold] = err_epoch_train[i - 1]
                        err_fold_valid[fold] = err_epoch_valid[i - 1]
                        err_epoch_train[i] = 0
                        err_epoch_valid[i] = 0
                        # load params from previous epoch
                        weights = torch.load(f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{i - 1}_weights')
                        split_values = torch.load(
                            f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{i - 1}_split_values')
                        sum_weights = torch.load(
                            f'{filename}_params_state_dict/softgedt_adt_cv_{fold}_epoch{i - 1}_sum_weights')

                        for k in range(len(SoftGeDTAdv.gate_weights)):
                            SoftGeDTAdv.gate_weights[k].data = weights[k]
                            SoftGeDTAdv.gate_split_values[k].data = split_values[k]

                        for k in range(len(SoftGeDTAdv.sum_weights)):
                            SoftGeDTAdv.sum_weights[k].data = sum_weights[k]

                        break
                else:
                    trigger_times = 0

                if i == (EPOCHS - 1):
                    err_fold_train[fold] = err_epoch_train[i]
                    err_fold_valid[fold] = err_epoch_valid[i]

            # test performance
            with torch.no_grad():
                for X, y in testloader:
                    pred, prob = SoftGeDTAdv.classify(X, return_prob=True)
                    test_fold_num_correct[fold] += torch.sum(pred == y)
                    test_fold_num_total[fold] += y.shape[0]

            # test adversarial accuracy
            err_fold_test[fold], test_total_loss = epoch_adversarial_training(testloader, SoftGeDTAdv, pgd_linf,
                                                                              non_editable_vector=non_editable_vector,
                                                                              clipping_max=clipping_max,
                                                                              clipping_min=clipping_min,
                                                                              datatypes=datatypes, num_iter=10,
                                                                              epsilon=0.1, L1_REG=L1_REG, omega=omega)

            acc_fold_test[fold] = test_fold_num_correct[fold] / test_fold_num_total[fold]

            print("SoftGeDT Adversarial Error on Training set:", err_fold_train[fold])
            print("SoftGeDT Adversarial Error on Validation set:", err_fold_valid[fold])

            print("SoftGeDT Accuracy on Test set:", acc_fold_test[fold])
            print("Average Accuracy on Test set:", sum(test_fold_num_correct) / sum(test_fold_num_total))

            print("ADT error, epsilon:0.1:", err_fold_test[fold])
            print("Average ADT error, epsilon:0.1", sum(err_fold_test) / (fold + 1))

        experiments_accs.append(sum(test_fold_num_correct) / sum(test_fold_num_total))
        experiments_adt_errs.append(sum(err_fold_test) / (fold + 1))

    acc_mean, acc_radius = mean_confidence_interval(experiments_accs)
    err_mean, err_radius = mean_confidence_interval(experiments_adt_errs)

    print('Experiments mean accuracy:', acc_mean, 'radius:', acc_radius)
    print('Experiments mean adt err', err_mean, 'radius', err_radius)


def kfold_normal_training_neural_networks(Net, filename, data, ncat, non_editable_vector, datatypes):
    """
    This function perform 5 experiments and 5-fold cross validation to test the accuracy and robustness of neural network
    before adversarial training

    :param filename: filename for storing parameters of model
    :param data: input data
    :param ncat: number of categories
    :param non_editable_vector: editability vector
    :param datatypes: data types of features
    :param include_sum_weight: whether include weights of sum nodes in the optimizer
    """

    experiments_accs = []
    experiments_adt_errs = []
    for i in range(5):
        # K-fold parameters
        K_FOLDS = 5
        kfold = KFold(n_splits=K_FOLDS, shuffle=True)

        # counters
        acc_fold_train = [0 for i in range(K_FOLDS)]
        acc_fold_valid = [0 for i in range(K_FOLDS)]
        acc_fold_test = [0 for i in range(K_FOLDS)]
        acc_fold_origin = [0 for i in range(K_FOLDS)]
        err_fold_adt = [0 for i in range(K_FOLDS)]

        training_fold_num_correct = [0 for i in range(K_FOLDS)]
        training_fold_num_total = [0 for i in range(K_FOLDS)]
        validation_fold_num_correct = [0 for i in range(K_FOLDS)]
        validation_fold_num_total = [0 for i in range(K_FOLDS)]
        test_fold_num_correct = [0 for i in range(K_FOLDS)]
        test_fold_num_total = [0 for i in range(K_FOLDS)]

        for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
            # split dataset for each fold
            print("Fold:", fold)
            shuffle = np.random.choice(train_ids, train_ids.shape[0], replace=False)
            valid_ids = shuffle[int(train_ids.shape[0] * 0.75):]
            train_ids = shuffle[:int(train_ids.shape[0] * 0.75)]

            # standardize dataset
            _, maxv, minv, mean, std = get_stats(data[train_ids, :], ncat)
            data_train_nn = standardize_data(data[train_ids, :], mean, std)
            data_valid_nn = standardize_data(data[valid_ids, :], mean, std)
            data_test_nn = standardize_data(data[test_ids, :], mean, std)

            data_train_nn = torch.from_numpy(data_train_nn)
            data_valid_nn = torch.from_numpy(data_valid_nn)
            data_test_nn = torch.from_numpy(data_test_nn)

            # create model
            net = Net(data_train_nn[:, :-1].shape[1])

            data_train_nn = torch.utils.data.TensorDataset(data_train_nn[:, :-1], data_train_nn[:, -1])
            data_valid_nn = torch.utils.data.TensorDataset(data_valid_nn[:, :-1], data_valid_nn[:, -1])
            data_test_nn = torch.utils.data.TensorDataset(data_test_nn[:, :-1], data_test_nn[:, -1])

            trainloader = torch.utils.data.DataLoader(data_train_nn, batch_size=32, shuffle=True)
            validloader = torch.utils.data.DataLoader(data_valid_nn, batch_size=32, shuffle=True)
            testloader = torch.utils.data.DataLoader(data_test_nn, batch_size=32, shuffle=True)

            # Xavier Initialization for the model
            for layer in net.children():
                if hasattr(layer, "weight"):
                    nn.init.xavier_uniform_(layer.weight)

                if hasattr(layer, "bias"):
                    nn.init.zeros_(layer.bias)

            # optimization set up
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
            EPOCHS = 100

            # early stopping parameter
            patience = 4
            trigger_times = 0

            # original prediction
            pred = net(data_test_nn[:][0])
            acc_fold_origin[fold] = (
                    torch.sum(torch.argmax(pred, dim=1) == data_test_nn[:][1]) / data_test_nn[:][1].shape[0])

            # counters
            acc_epoch_train = [0 for i in range(EPOCHS)]
            acc_epoch_valid = [0 for i in range(EPOCHS)]
            training_epoch_num_correct = [0 for i in range(EPOCHS)]
            training_epoch_num_total = [0 for i in range(EPOCHS)]
            validation_epoch_num_correct = [0 for i in range(EPOCHS)]
            validation_epoch_num_total = [0 for i in range(EPOCHS)]

            torch.save(net.state_dict(), f'{filename}_params_state_dict/fnn_normal_cv_{fold}_epoch{-1}')

            # shap values
            explainer = shap.DeepExplainer(net, data_train_nn[
                np.random.choice(range(data_train_nn[:][0].shape[0]), 30, replace=False)][0])
            shap_values = explainer.shap_values(data_test_nn[:][0])
            shap_values = np.array(shap_values)
            shap_values = np.mean(np.mean(np.abs(shap_values), axis=1), axis=0)
            if 0 in shap_values:
                omega = torch.from_numpy((1 / (shap_values + 10e-2))) / np.linalg.norm((1 / (shap_values + 10e-2)), 2)
            else:
                omega = torch.from_numpy((1 / shap_values)) / np.linalg.norm((1 / shap_values), 2)
            # clipping values
            clipping_max = torch.max(data_train_nn[:][0], dim=0).values
            clipping_min = torch.min(data_train_nn[:][0], dim=0).values

            for i in range(EPOCHS):
                print("epoch:", i)

                for X, y in trainloader:
                    optimizer.zero_grad()
                    prediction = net(X)
                    loss = F.cross_entropy(prediction, y.long())
                    loss.backward()
                    optimizer.step()

                    training_epoch_num_correct[i] += torch.sum(torch.argmax(prediction, dim=1) == y)
                    training_epoch_num_total[i] += y.shape[0]

                acc_epoch_train[i] = training_epoch_num_correct[i] / training_epoch_num_total[i]

                with torch.no_grad():
                    for X, y in validloader:
                        prediction = net(X)
                        validation_epoch_num_correct[i] += torch.sum(torch.argmax(prediction, dim=1) == y)
                        validation_epoch_num_total[i] += y.shape[0]

                acc_epoch_valid[i] = validation_epoch_num_correct[i] / validation_epoch_num_total[i]
                print('train acc:', acc_epoch_train[i], 'valid acc:', acc_epoch_valid[i], "trigger times:",
                      trigger_times)

                torch.save(net.state_dict(), f'{filename}_params_state_dict/fnn_normal_cv_{fold}_epoch{i}')

                # Early stopping during epoch
                if i > 0 and acc_epoch_valid[i] <= acc_epoch_valid[i - 1]:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print("Early Stopping")
                        training_fold_num_correct[fold] = training_epoch_num_correct[i - 1]
                        training_fold_num_total[fold] = training_epoch_num_total[i - 1]
                        validation_fold_num_correct[fold] = validation_epoch_num_correct[i - 1]
                        validation_fold_num_total[fold] = validation_epoch_num_total[i - 1]
                        acc_fold_train[fold] = acc_epoch_train[i - 1]
                        acc_fold_valid[fold] = acc_epoch_valid[i - 1]
                        acc_epoch_train[i] = 0
                        acc_epoch_valid[i] = 0
                        # load params from previous epoch
                        net.load_state_dict(torch.load(f'{filename}_params_state_dict/fnn_normal_cv_{fold}_epoch{i - 1}'))
                        break
                else:
                    trigger_times = 0

                if i == (EPOCHS - 1):
                    training_fold_num_correct[fold] = training_epoch_num_correct[i]
                    training_fold_num_total[fold] = training_epoch_num_total[i]
                    validation_fold_num_correct[fold] = validation_epoch_num_correct[i]
                    validation_fold_num_total[fold] = validation_epoch_num_total[i]
                    acc_fold_train[fold] = acc_epoch_train[i]
                    acc_fold_valid[fold] = acc_epoch_valid[i]

            # test performance
            with torch.no_grad():
                for X, y in testloader:
                    prediction = net(X)
                    test_fold_num_correct[fold] += torch.sum(torch.argmax(prediction, dim=1) == y)
                    test_fold_num_total[fold] += y.shape[0]

            # test adversarial accuracy
            err_fold_adt[fold], test_total_loss = epoch_adversarial_training_nn(testloader, net, pgd_linf_nn,
                                                                                non_editable_vector=non_editable_vector,
                                                                                clipping_max=clipping_max,
                                                                                clipping_min=clipping_min,
                                                                                datatypes=datatypes, num_iter=10,
                                                                                epsilon=0.1, omega=omega)

            acc_fold_test[fold] = test_fold_num_correct[fold] / test_fold_num_total[fold]

            print("FNN Accuracy on Training set:", acc_fold_train[fold])
            print("FNN Accuracy on Validation set:", acc_fold_valid[fold])

            print("FNN Accuracy on Test set:", acc_fold_test[fold])
            print("Average Accuracy on Test set:", sum(test_fold_num_correct) / sum(test_fold_num_total))

            print("Original Accuracy on Test set:", acc_fold_origin[fold])
            print("Average Original Accuracy on Test set:", sum(acc_fold_origin) / (fold + 1))

            print("ADT error, epsilon:0.1:", err_fold_adt[fold])
            print("Average ADT error, epsilon:0.1", sum(err_fold_adt) / (fold + 1))

        experiments_accs.append(sum(test_fold_num_correct) / sum(test_fold_num_total))
        experiments_adt_errs.append(sum(err_fold_adt) / (fold + 1))

    acc_mean, acc_radius = mean_confidence_interval(experiments_accs)
    err_mean, err_radius = mean_confidence_interval(experiments_adt_errs)

    print('Experiments mean accuracy:', acc_mean, 'radius:', acc_radius)
    print('Experiments mean adt err', err_mean, 'radius', err_radius)


def kfold_adversarial_training_neural_networks(Net, filename, data, ncat, non_editable_vector, datatypes):
    """
    This function perform 5 experiments and 5-fold cross validation to test the accuracy and robustness of neural network
    before adversarial training

    :param filename: filename for storing parameters of model
    :param data: input data
    :param ncat: number of categories
    :param non_editable_vector: editability vector
    :param datatypes: data types of features
    :param include_sum_weight: whether include weights of sum nodes in the optimizer
    """

    experiments_accs = []
    experiments_adt_errs = []

    for i in range(5):
        # K-fold parameters
        K_FOLDS = 5
        kfold = KFold(n_splits=K_FOLDS, shuffle=True)

        # counters
        err_fold_train = [0 for i in range(K_FOLDS)]
        err_fold_valid = [0 for i in range(K_FOLDS)]
        err_fold_test = [0 for i in range(K_FOLDS)]

        acc_fold_test = [0 for i in range(K_FOLDS)]
        test_fold_num_correct = [0 for i in range(K_FOLDS)]
        test_fold_num_total = [0 for i in range(K_FOLDS)]

        for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
            # split dataset for each fold
            print("Fold:", fold)
            shuffle = np.random.choice(train_ids, train_ids.shape[0], replace=False)
            valid_ids = shuffle[int(train_ids.shape[0] * 0.75):]
            train_ids = shuffle[:int(train_ids.shape[0] * 0.75)]

            # standardize dataset
            _, maxv, minv, mean, std = get_stats(data[train_ids, :], ncat)
            data_train_nn = standardize_data(data[train_ids, :], mean, std)
            data_valid_nn = standardize_data(data[valid_ids, :], mean, std)
            data_test_nn = standardize_data(data[test_ids, :], mean, std)

            data_train_nn = torch.from_numpy(data_train_nn)
            data_valid_nn = torch.from_numpy(data_valid_nn)
            data_test_nn = torch.from_numpy(data_test_nn)

            # create model
            netAdv = Net(data_train_nn[:, :-1].shape[1])

            data_train_nn = torch.utils.data.TensorDataset(data_train_nn[:, :-1], data_train_nn[:, -1])
            data_valid_nn = torch.utils.data.TensorDataset(data_valid_nn[:, :-1], data_valid_nn[:, -1])
            data_test_nn = torch.utils.data.TensorDataset(data_test_nn[:, :-1], data_test_nn[:, -1])

            trainloader = torch.utils.data.DataLoader(data_train_nn, batch_size=32, shuffle=True)
            validloader = torch.utils.data.DataLoader(data_valid_nn, batch_size=32, shuffle=True)
            testloader = torch.utils.data.DataLoader(data_test_nn, batch_size=32, shuffle=True)

            # Xavier Initialization for the model
            for layer in netAdv.children():
                if hasattr(layer, "weight"):
                    nn.init.xavier_uniform_(layer.weight)

                if hasattr(layer, "bias"):
                    nn.init.zeros_(layer.bias)

            # optimization set up
            optimizer = torch.optim.Adam(netAdv.parameters(), lr=0.01)
            EPOCHS = 100

            # early stopping parameter
            patience = 4
            trigger_times = 0

            # counters
            err_epoch_train = [0 for i in range(EPOCHS)]
            err_epoch_valid = [0 for i in range(EPOCHS)]

            torch.save(netAdv.state_dict(), f'{filename}_params_state_dict/fnn_adt_cv_{fold}_epoch{-1}')

            # shap values
            explainer = shap.DeepExplainer(netAdv, data_train_nn[
                np.random.choice(range(data_train_nn[:][0].shape[0]), 30, replace=False)][0])
            shap_values = explainer.shap_values(data_test_nn[:][0])
            shap_values = np.array(shap_values)
            shap_values = np.mean(np.mean(np.abs(shap_values), axis=1), axis=0)
            if 0 in shap_values:
                omega = torch.from_numpy((1 / (shap_values + 10e-2))) / np.linalg.norm((1 / (shap_values + 10e-2)), 2)
            else:
                omega = torch.from_numpy((1 / shap_values)) / np.linalg.norm((1 / shap_values), 2)
            # clipping values
            clipping_max = torch.max(data_train_nn[:][0], dim=0).values
            clipping_min = torch.min(data_train_nn[:][0], dim=0).values

            for i in range(EPOCHS):
                print("epoch:", i)
                err_epoch_train[i], train_total_loss = epoch_adversarial_training_nn(trainloader, netAdv, pgd_linf_nn,
                                                                                     optimizer,
                                                                                     non_editable_vector=non_editable_vector,
                                                                                     clipping_max=clipping_max,
                                                                                     clipping_min=clipping_min,
                                                                                     datatypes=datatypes, num_iter=10,
                                                                                     epsilon=0.1, omega=omega)

                err_epoch_valid[i], valid_total_loss = epoch_adversarial_training_nn(validloader, netAdv, pgd_linf_nn,
                                                                                     non_editable_vector=non_editable_vector,
                                                                                     clipping_max=clipping_max,
                                                                                     clipping_min=clipping_min,
                                                                                     datatypes=datatypes, num_iter=10,
                                                                                     epsilon=0.1, omega=omega)
                print("train error:", err_epoch_train[i], "train loss:", train_total_loss, "validation error",
                      err_epoch_valid[i])

                torch.save(netAdv.state_dict(), f'{filename}_params_state_dict/fnn_adt_cv_{fold}_epoch{i}')

                # Early stopping during epoch
                if i > 0 and err_epoch_valid[i] >= err_epoch_valid[i - 1]:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print("Early Stopping")
                        err_fold_train[fold] = err_epoch_train[i - 1]
                        err_fold_valid[fold] = err_epoch_valid[i - 1]
                        err_epoch_train[i] = 0
                        err_epoch_valid[i] = 0
                        # load params from previous epoch
                        netAdv.load_state_dict(torch.load(f'{filename}_params_state_dict/fnn_adt_cv_{fold}_epoch{i - 1}'))

                        break
                else:
                    trigger_times = 0

                if i == (EPOCHS - 1):
                    err_fold_train[fold] = err_epoch_train[i]
                    err_fold_valid[fold] = err_epoch_valid[i]

            # test performance
            with torch.no_grad():
                for X, y in testloader:
                    prediction = netAdv(X)
                    test_fold_num_correct[fold] += torch.sum(torch.argmax(prediction, dim=1) == y)
                    test_fold_num_total[fold] += y.shape[0]

            # test adversarial accuracy
            err_fold_test[fold], test_total_loss = epoch_adversarial_training_nn(testloader, netAdv, pgd_linf_nn,
                                                                                 non_editable_vector=non_editable_vector,
                                                                                 clipping_max=clipping_max,
                                                                                 clipping_min=clipping_min,
                                                                                 datatypes=datatypes, num_iter=10,
                                                                                 epsilon=0.1, omega=omega)

            acc_fold_test[fold] = test_fold_num_correct[fold] / test_fold_num_total[fold]

            print("FNN Adversarial Error on Training set:", err_fold_train[fold])
            print("FNN Adversarial Error on Validation set:", err_fold_valid[fold])

            print("FNN Accuracy on Test set:", acc_fold_test[fold])
            print("Average Accuracy on Test set:", sum(test_fold_num_correct) / sum(test_fold_num_total))

            print("ADT error, epsilon:0.1:", err_fold_test[fold])
            print("Average ADT error, epsilon:0.1", sum(err_fold_test) / (fold + 1))

        experiments_accs.append(sum(test_fold_num_correct) / sum(test_fold_num_total))
        experiments_adt_errs.append(sum(err_fold_test) / (fold + 1))

    acc_mean, acc_radius = mean_confidence_interval(experiments_accs)
    err_mean, err_radius = mean_confidence_interval(experiments_adt_errs)

    print('Experiments mean accuracy:', acc_mean, 'radius:', acc_radius)
    print('Experiments mean adt err', err_mean, 'radius', err_radius)
