import os
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from utils import try_gpu
from configs import Config
from model import SiamResNet18
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset import SiameseNetworkDataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

############################################################
#  Seed
############################################################

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

############################################################
#  Train
############################################################

def train(k_fold,fold_id):
    # CUDA Settings
    if Config.using_gpu == True:
        device = try_gpu(Config.gpu_id)
        if device == torch.device('cpu'):
            print("Warning: CPU is used!")
        cudnn.benchmark = False
        cudnn.deterministic = True
        print("GPU is used!")
    else:
        device = torch.device('cpu')
        print("CPU is used!")
    
    # Log Dir
    if not os.path.exists('../logs/%s' % Config.log_dir):
        os.makedirs('../logs/%s' % Config.log_dir)
    output_file = str(
        "../logs/%s/log_fold%02d_foldID%02d.out" % (Config.log_dir,k_fold,fold_id))
    model_file = str(
        "../logs/%s/model_fold%02d_foldID%02d.pth" % (Config.log_dir,k_fold,fold_id))
    tb_dir = "../logs/%s/tb_fold%02d_foldID%02d/" % (Config.log_dir,k_fold,fold_id)
    os.makedirs(tb_dir)
    writer = SummaryWriter(tb_dir)
    f = open(output_file,'w')
    f.close()

    # Train Preparation
    transform = transforms.ToTensor()
    label = pd.read_csv(Config.train_label_dir,header=None)
    label = label.values
    train_dataset = SiameseNetworkDataset(image_dir = Config.train_data_dir,
                                          label = label,
                                          fold = k_fold,
                                          fold_id = fold_id,
                                          transform = transform,
                                          training = True)
    valid_dataset = SiameseNetworkDataset(image_dir = Config.train_data_dir,
                                          label = label,
                                          fold = k_fold,
                                          fold_id = fold_id,
                                          transform = transform,
                                          training = False)
    train_dataLoader = DataLoader(train_dataset,
                                  shuffle = True,
                                  num_workers = Config.num_workers,
                                  batch_size = Config.train_batch_size)
    valid_dataLoader = DataLoader(valid_dataset,
                                  shuffle = False,
                                  num_workers = Config.num_workers,
                                  batch_size = Config.train_batch_size)

    # Initialize
    net = SiamResNet18()
    net.to(device)
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(net.parameters(),lr = Config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,Config.lr_scheduler_gamma)

    # Global Variable
    max_correct = 0

    # Iteration
    for epoch in range(0,Config.train_epochs):
        # one epoch training
        net.train()
        train_loss = 0
        for i, data in enumerate(train_dataLoader,0):
            # one batch training
            image0, image1, labels = data
            image0, image1, labels = image0.to(device), image1.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(image0,image1)
            l = loss(output,labels)
            train_loss += l.sum().item()
            l.mean().backward()
            optimizer.step()

            # print
            if (i+1)%10 == 0:
                print('Train Epoch: {} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1,(i+1)*len(image0),len(train_dataset),100.*(i+1)/len(train_dataLoader),
                        l.mean().item()
                    ))
        train_loss /= len(train_dataset)

        # one epoch validation
        net.eval()
        valid_loss = 0
        valid_corr = 0
        with torch.no_grad():
            for i, data in enumerate(valid_dataLoader,0):
                image0, image1, labels = data
                image0, image1, labels = image0.to(device), image1.to(device), labels.to(device)
                output = net(image0,image1)
                l = loss(output,labels)
                acc = (output.argmax(1)==labels).sum().item()
                valid_corr += acc
                valid_loss += l.sum().item()
        
        if max_correct<valid_corr:
            torch.save(net.state_dict(),model_file)
            max_correct = valid_corr
            print("Best accuracy! correct cls: %5d" % valid_corr)
        
        valid_loss /= len(valid_dataset)
        valid_accuracy = 100*valid_corr/len(valid_dataset)
        best_valid_accuracy = 100*max_correct/len(valid_dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) (best: {:.2f}%)\n'.format(
                valid_loss, valid_corr, len(valid_dataset), valid_accuracy, best_valid_accuracy))

        f = open(output_file,'a')
        f.write(" %3d %12.6f %12.6f %9.3f %9.3f\n" % (
            epoch+1, train_loss, valid_loss, valid_accuracy, best_valid_accuracy))
        f.close()
        writer.add_scalar(tag="Train Loss",scalar_value=train_loss,global_step=epoch+1)
        writer.add_scalar(tag="Valid Loss",scalar_value=valid_loss,global_step=epoch+1)
        writer.add_scalar(tag="Valid Accuracy",scalar_value=valid_accuracy,global_step=epoch+1)

        # update
        lr_scheduler.step()
    writer.close()

############################################################
#  Start!
############################################################

if __name__ == '__main__':
    start_time = time.time()
    # Random seed
    setup_seed(Config.seed)
    # Train
    for i in range(0,Config.k_fold):
        print('\nStart [%1dth/%d] fold validation!\n'% (i+1,Config.k_fold))
        train(k_fold = Config.k_fold, fold_id = i)
    end_time = time.time()
    print("Total time: "+str(end_time - start_time))