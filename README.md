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
|   Algorithm  | MLP-1024-3 | MLP-2048-3 | MLP-4096-3 | MLP-8192-3 | ResNet18 |
|:--------:|:--------:|:--------:|:--------:| :--------:|:--------:|
|   Non Dropout  | - | - | - | - | - |
|   Original Dropout  | - | - | - | - | - |
|   Guided Dropout (top-k) * | 58.75 | 59.65 | 59.64 | 59.92 | 94.02 |
|   Guided Dropout (DR) * | 59.84 | 60.12 | 60.89 | 61.32 | 94.12 |
|   Guided Dropout  | - | - | - | - | - |
|   Guided Dropout | - | - | - | - | - |

\* means the result listed in the paper

### CIFAR100
|   Algorithm  | MLP-1024-3 | MLP-2048-3 | MLP-4096-3 | MLP-8192-3 | ResNet18 |
|:--------:|:--------:|:--------:|:--------:| :--------:|:--------:|
|   Non Dropout  | - | - | - | - | - |
|   Original Dropout  | - | - | - | - | - |
|   Guided Dropout (top-k) * | 30.92 | 31.59 | 31.34 | 32.11 | 76.98 |
|   Guided Dropout (DR) * | 31.88 | 32.78 | 33.01 | 33.15 | 77.52 |
|   Guided Dropout  | - | - | - | - | - |
|   Guided Dropout | - | - | - | - | - |



