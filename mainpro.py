'''Train Neural Network with Dropout.'''
from __future__ import print_function
import sys
import torch.optim as optim
import torchvision
import transforms as transforms
import os
import torch.backends.cudnn as cudnn
sys.path.append("./models")
import torch
import torch.nn as nn
import argparse
import utils
from torch.autograd import Variable
sys.path.append("./")
from models import *
from datetime import datetime
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch Neural Network Training on CIFAR10, CIFAR100, FM ')
#Dataset param
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--dataset-root', type=str, default='/xxx/Datasets', help='Dataset root')
parser.add_argument('--num-classes', type=int, default=10, help='num of datatset classes')
parser.add_argument('--DOWNLOAD-DATASET','-d' ,action='store_true', help='Download datatset')
parser.add_argument('--RGB-Image', action='store_true', help='RGB image data')

#Train param
parser.add_argument('--batch-size','-b', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', '-r', type=str, default='', help='resume from checkpoint')
parser.add_argument('--save-dir', type=str, default='result_model', help='save log and model')
parser.add_argument('--gpu-ids', type=str, default='0', help='gpu id eg. 0 or  0,1')
parser.add_argument('--exp-name', type=str, default='test', help='experiment-name')
parser.add_argument('--epoch','-e', type=int, default=200, help='num of epochs')
parser.add_argument('--start', type=int, default=1, help='start')
parser.add_argument('--times', type=int, default=5, help='times')

#Network param
parser.add_argument('--arc', type=str, default='resnet18', help='model')
parser.add_argument('--hidden-dim', type=int, default=8192, help='num of hidden nodes of mlp')
parser.add_argument('--mlp-depth', type=int, default=3, help='num of depth of mlp')

#Dropout param
parser.add_argument('--drop-type', type=str, default='GuidedDropout', help='GuidedDropout, Dropout')
parser.add_argument('--drop-rate', type=float, default=0.2, help='initial dropout rate')

use_cuda = torch.cuda.is_available()


def get_dataloader(opt):
    # Data
    print('==> Preparing data..')
    data_path = os.path.join(opt.dataset_root,opt.dataset)
    if 'CIFAR10' == opt.dataset:
        opt.num_classes = 10
        opt.image_size = 32
        opt.RGB_Image = True
        transform_train = transforms.Compose([
            transforms.RandomCrop(opt.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # train and test data
        train_data = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=opt.DOWNLOAD_DATASET, transform=transform_train)
        test_data = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=opt.DOWNLOAD_DATASET, transform=transform_test)

    elif 'CIFAR100' == opt.dataset:
        opt.num_classes = 100
        opt.image_size = 32
        opt.RGB_Image = True
        transform_train = transforms.Compose([
            transforms.RandomCrop(opt.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])

        # train and test data
        train_data = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=opt.DOWNLOAD_DATASET, transform=transform_train)
        test_data = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=opt.DOWNLOAD_DATASET, transform=transform_test)
    elif 'Fashionmnist' == opt.dataset:
        opt.num_classes = 10
        opt.image_size = 28
        opt.RGB_Image = False
        transform_data = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = torchvision.datasets.FashionMNIST(
            root=data_path, train=True, transform=transform_data, download=opt.DOWNLOAD_DATASET)
        test_data = torchvision.datasets.FashionMNIST(
            root=data_path, train=False, transform=transform_data, download=opt.DOWNLOAD_DATASET)
    else:
        raise Exception("Unknown Dataset！")

    print("train_data:", len(train_data))
    print("test_data:", len(test_data))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.num_workers)
    return trainloader,testloader


# Training
def train(epoch,trainloader,net,criterion,optimizer,scheduler):
    net.train()
    losses = utils.AverageMeter()
    lr = scheduler.get_last_lr()
    print('Current Learning Rate: %s' % str(lr))
    net.dropout_scheduler(epoch)
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
    scheduler.step()
    return losses.avg


def test(testloader,net,criterion):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(targets.data).cpu().sum()
            acc.update(float(correct)/inputs.size(0), inputs.size(0))
    return losses.avg,100*acc.avg


def main():
    opt = parser.parse_args()
    cudnn.deterministic = True
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    total_epoch = opt.epoch
    results_log_csv_name = opt.exp_name + '_log.csv'
    path = 'result'
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, opt.exp_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    for i_n in range(opt.start, opt.start+opt.times):
        best_test_acc = 0  # best accuracy
        best_test_acc_epoch = 0
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        save_dir = 'result'+str(i_n)
        save_path = os.path.join(path,save_dir)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        trainloader, testloader = get_dataloader(opt)
        print('==> Building model..')
        if 'mlp' in opt.arc:
            net = mlp(image_size=opt.image_size,hidden_dim=opt.hidden_dim,depth=opt.mlp_depth,num_classes=opt.num_classes,RGB_Image=opt.RGB_Image,drop=opt.drop_type,drop_rate=opt.drop_rate)
        else:
            net = ResNet18(num_classes=opt.num_classes,RGB_Image=opt.RGB_Image,drop=opt.drop_type,drop_rate=opt.drop_rate)
        print("model: {} ".format(opt.arc))

        if opt.resume != '':
            print('==> Resuming from checkpoint..')
            assert os.path.exists(opt.resume), 'Error: no checkpoint file found!'
            checkpoint = torch.load(opt.resume)
            net.load_state_dict(checkpoint['state_dict'])
            best_test_acc = checkpoint['acc']
            best_test_acc_epoch = checkpoint['epoch']
            start_epoch = checkpoint['epoch'] + 1

        if use_cuda:
            net = net.cuda()
            # net = nn.DataParallel(net)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        with open(os.path.join(save_path, results_log_csv_name), 'w') as f:
            f.write(' epoch , train_loss, test_loss, test_acc ,time\n')
        #start train
        for epoch in range(start_epoch, total_epoch):
            print('current time:',datetime.now().strftime('%b%d-%H:%M:%S'))
            print("Epoch: [{} | {} ]".format(epoch+1, total_epoch))
            train_loss = train(epoch,trainloader,net,criterion,optimizer,scheduler)
            test_loss, test_acc = test(testloader,net,criterion)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch
                print('Saving..')
                print("best_test_acc: %0.3f" % best_test_acc)
                checkpoint = {
                    'state_dict': net.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                torch.save(checkpoint, os.path.join(save_path,opt.exp_name+'.pth'))
            print("Train loss: {:.3f} | Test loss {:.3f} | Test acc {:.3f}]".format(train_loss,test_loss, test_acc))
            with open(os.path.join(save_path, results_log_csv_name), 'a') as f:
                f.write('%03d,%0.3f,%0.3f,%0.3f,%s,\n' % (
                    epoch, train_loss, test_loss, test_acc, datetime.now().strftime('%b%d-%H:%M:%S')))

        print("best_test_acc: %0.3f" % best_test_acc)
        print("best_test_acc_epoch: %d" % best_test_acc_epoch)
        # best ACC
        with open(os.path.join(save_path, results_log_csv_name), 'a') as f:
            f.write('%s,%03d,%0.3f,\n' % ('best acc (test)',best_test_acc_epoch, best_test_acc))


if __name__=="__main__":
    main()