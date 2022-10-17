# Attack-Soft-Generative-Decision-Tree
Soft Generative Decision Tree in Python

# Project informaiton
This is a project for my master thesis 'Making soft generative decision tree robust to gradient-based adversarial attacks'. Most lines of code in this project are from AlCorreia(Alvaro Correia)'s git repository [GeFs](https://github.com/AlCorreia/GeFs.git).

Soft Generative Decision Tree is an extension of Generative Decision Tree(GeDT), where the indicator functions in GeDT are replaced by gating functions. Generative Decision Tree is invented by Correia et al. in their thesis [Joints in Random Forests](https://proceedings.neurips.cc/paper/2020/hash/8396b14c5dff55d13eea57487bf8ed26-Abstract.html) and [Towards Robust Classification with Deep Generative Forests](https://arxiv.org/abs/2007.05721).

# Experiments for thesis
The experiments for the thesis is under the folder 'attack_soft_generative_decision_tree/experiments', which study the performance of soft GeDT in accuracy and robustness, and visualize the decision boundary of soft GeDT.

# Usage
For the usage of GeDT, please visit AlCorreia(Alvaro Correia)'s git repository [GeFs](https://github.com/AlCorreia/GeFs.git). We will use a synthetic dataset for demonstration:

```
# Define a synthetic dataset
n_samples = 100
n_features = 20
n_classes = 2

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_redundant=2, n_repeated=0, 
                           n_classes=n_classes, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, 
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
# We need to specify the number of categories of each feature (with 1 for continuous features).
ncat = np.ones(n_features+1)  # Here all features are continuous
ncat[-1] = n_classes  # The class variable is naturally categorical

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
```

## Convert sklearn Decision Tree into soft GeDT
The first step is to construct sklearn decision tree:
```
clf = DecisionTreeClassifier()
clf.fit(data_train_tree[:][0].numpy(), data_train_tree[:][1].numpy())
```

If you want to use soft GeDT, you will have to convert the data into torch tensors, otherwise the tree converted will be GeDT. This conversion is important for calculating the gradients with respect to the parameters.
```
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
```

convert sklearn decision tree into soft GeDT using method `gefs.sklearn_utils.tree2pc`
```
SoftGeDT = tree2pc(tree, X_train, y_train, ncat, learnspn=30, minstd=.1, smoothing=0.1)
pred = SoftGeDT.classify(X_test)
```
## Train soft GeDT

First, pack data into pytorch data loader
```
synthetic_train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
BATCH_SIZE = 32
synthetic_train_loader = torch.utils.data.DataLoader(synthetic_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

synthetic_test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
BATCH_SIZE = 32
synthetic_test_loader = torch.utils.data.DataLoader(synthetic_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
```

Next, train soft GeDT as training neural networks.
The parameters of SoftGeDT are classified into 'gate_weights', 'gate_split_values' and 'sum_weights'.

In this demo, the loss function is cross-entropy function.
```
opt = torch.optim.Adam(SoftGeDT.gate_weights+SoftGeDT.gate_split_values+SoftGeDT.sum_weights, lr=0.1)
EPOCHS =10
for i in range(EPOCHS):
    print("epoch:",i)
    for X,y in synthetic_train_loader:
        opt.zero_grad()
        pred, prob = SoftGeDT.classify(X,return_prob=True)
        loss =F.nll_loss(torch.log(prob),y.type(torch.int64),reduction='sum')
        loss.backward()
        opt.step()

pred = SoftGeDT.classify(X_test)
```

## Classify with soft GeDT
You can use soft GeDT to classify by using method `gefs.pc.classify`:
```
pred = SoftGeDT.classify(X_test)
```

You can also get access to the conditional probability by setting `return_prob=True` in the `gefs.pc.classify` method:
```
pred, prob = SoftGeDT.classify(X_test, return_prob=True)
```

## Adversarial training
You can use method `adversarial.epoch_adversarial_training` to perform the adversarial training on soft GeDT. The method `adversarial.pgd_linf` is a PGD attack

If no optmizer is passed to method `adversarial.epoch_adversarial_training`, then only adversarial attack will be performed but adversarial training will not.
If you set `return_adv=True`, then adversarial data will be returned.
```
err_test, test_total_loss, X_advs,y_advs = epoch_adversarial_training(synthetic_test_loader,SoftGeDT,pgd_linf,return_adv=True,non_editable_vector=None,clipping_max=None,clipping_min=None,datatypes=None,num_iter=10,epsilon=0.1,L1_REG=None,omega=None)
X_adv=X_advs[0]
y_adv = y_advs[0]
for i in range(1,len(X_advs)):
    X_adv=torch.cat((X_adv,X_advs[i]),dim=0)
    y_adv = torch.cat((y_adv,y_advs[i]),dim=0)
```

If optmizer is passed to method `adversarial.epoch_adversarial_training`, then adversarial training will also be performed:
```
err_epoch_train, train_total_loss = epoch_adversarial_training(synthetic_train_loader, SoftGeDT, pgd_linf, opt=opt, non_editable_vector=None, clipping_max=None, clipping_min=None, datatypes=None, num_iter=10, epsilon=0.1, L1_REG=None, omega=None)
```

To perform adversarial training on neural networks, you can use `adversarial.epoch_adversarial_training_nn` and `adversarial.pgd_linf_nn`. `model` is the neural network in Pytorch and `opt` is its corresponding optmizer.
```
err_epoch_train,train_total_loss=epoch_adversarial_training_nn(synthetic_train_loader,model,pgd_linf_nn,opt=opt,non_editable_vector=None,clipping_max=None,clipping_min=None, datatypes=None, num_iter=10,epsilon=0.1, omega=None)
```
