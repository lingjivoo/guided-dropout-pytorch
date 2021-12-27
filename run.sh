#CUDA_VISIBLE_DEVICES=7 python mainpro.py --dataset CIFAR10 --arc resnet18 -e 200 --lr 0.01 --drop-type Dropout --exp-name resnet18-original-dropout-cifar10
#CUDA_VISIBLE_DEVICES=7 python mainpro.py --dataset CIFAR10 --arc resnet18 -e 200 --lr 0.01 --exp-name resnet18-guided-dropout-cifar10
#CUDA_VISIBLE_DEVICES=7 python mainpro.py --dataset CIFAR10 --arc mlp --mlp-depth 3 --hidden-dim 1024 -e 200 --lr 0.01 --exp-name mlp-1024-3-guided-dropout-cifar10
CUDA_VISIBLE_DEVICES=7 python mainpro.py --dataset CIFAR10 --arc mlp --mlp-depth 3 --hidden-dim 1024 -e 200 --lr 0.01 --drop-type Dropout --drop-rate 0.1 --exp-name mlp-1024-3-original-dropout-cifar10
