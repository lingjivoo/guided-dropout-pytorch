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
|   EAC-Net  | 39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 |  35.9 | 35.8 | 55.9 |
|   JAA-Net  |  47.2 | 44.0 |54.9 |77.5 |74.6 |84.0 |86.9 |61.9 |43.6 |60.3 |42.7 |41.9 |60.0|
|   LP-Net |  43.4  | 38.0  | 54.2  | 77.1  | 76.7  | 83.8  | 87.2  |63.3  |45.3  |60.5  |48.1  |54.2  |61.0|
|   ARL | 45.8 |39.8 |55.1 |75.7 |77.2 |82.3 |86.6 |58.8 |47.6 |62.1 |47.4 |55.4 |61.1|
|   SEV-Net | 58.2 |50.4 |58.3 |81.9 |73.9 |87.8 |87.5 |61.6 |52.6 |62.2 |44.6 |47.6 |63.9|
|   FAUDT | 51.7 |49.3 |61.0 |77.8 |79.5 |82.9 |86.3 |67.6 |51.9 |63.0 |43.7 |56.3 |64.2 |




