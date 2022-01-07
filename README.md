# Unofficial PyTorch implementation of [Guided Dropout](https://arxiv.org/abs/1812.03965)
This is a simple implementation of Guided Dropout for research.
We try to reproduce the algorithm according to the paper published in AAA-19, but we can't guarantee the performance reported in the paper.
We will list some experiment results soon.

## TODO
- [x]  Release the reproduced code
- [ ]  list experiment results
- [ ]  ...

## Setup
```
pip install -r requirements.txt
```

## Run
1. Run Guided Dropout on CIFAR10 Dataset (mlp 3 hidden layers with 1024 nodes)
```
python mainpro.py --dataset CIFAR10 --arc mlp --mlp-depth 3 --hidden-dim 1024 -e 200 --lr 0.01 --exp-name mlp-1024-3-guided-dropout-cifar10
```

2. Run Original Dropout on Fashionmnist Dataset (mlp 3 hidden layers with 8192 nodes)
```
python mainpro.py --dataset Fashionmnist --arc mlp --mlp-depth 3 --hidden-dim 8192 -e 200 --lr 0.01 --exp-name mlp-8192-3-original-dropout-cifar10 --drop-type Dropout --drop-rate 0.2
```

3. Run Guided Dropout on CIFAR100 Dataset (ResNet-18)
```
python mainpro.py --dataset CIFAR100 --arc ResNet18 -e 200 --lr 0.01 --exp-name resnet18-guided-dropout-cifar100 --drop-type GuidedDropout --drop-rate 0.2
```
## Result
### CIFAR10
|   Method  | AU1 | AU2 | AU4 | AU6 | AU7 | AU10 | AU12 | AU14 | AU15 | AU17 | AU23 | AU24 | Avg. |
|:--------:|:--------:|:--------:|:--------:| :--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|   EAC-Net  | 39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 | | 35.9 | 35.8 | 55.9 |
|   JAA-Net  |  39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 | | 35.9 | 35.8 | 55.9 |
|   LP-Net |  39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 | | 35.9 | 35.8 | 55.9 |
|   ARL | 39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 | | 35.9 | 35.8 | 55.9 |
|   SEV-Net  | 39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 | | 35.9 | 35.8 | 55.9 |
|   FAUDT | 39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 | | 35.9 | 35.8 | 55.9 |




