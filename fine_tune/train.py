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
from model import FineTuneResNet18
from dataset import FineTuneDataset
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
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

def train(k_fold, fold_id, model_path, model_id):
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
        "../logs/%s/log_ID%03d_fold%02d_foldID%02d.out" % (Config.log_dir,model_id,k_fold,fold_id))
    model_file = str(
        "../logs/%s/model_ID%03d_fold%02d_foldID%02d.pth" % (Config.log_dir,model_id,k_fold,fold_id))
    tb_dir = "../logs/%s/tb_ID%03d_fold%02d_foldID%02d/" % (Config.log_dir,model_id,k_fold,fold_id)
    f = open(output_file,'w')
    f.close()
    os.makedirs(tb_dir)
    writer = SummaryWriter(tb_dir)

    # Train Preparation
    label = pd.read_csv(Config.train_label_dir)
    label = label.values
    train_dataset = FineTuneDataset(image_dir = Config.train_data_dir,
                                    label = label,
                                    fold = k_fold,
                                    fold_id = fold_id,
                                    transform = transforms.ToTensor(),
                                    training = True)
    valid_dataset = FineTuneDataset(image_dir = Config.train_data_dir,
                                    label = label,
                                    fold = k_fold,
                                    fold_id = fold_id,
                                    transform = transforms.ToTensor(),
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
    net = FineTuneResNet18(model_train_mode = model_id,
                           model_path = model_path)
    net.to(device)
    loss = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(net.parameters(),lr = Config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,Config.lr_scheduler_gamma)

    # Global Variable
    best_valid_loss = -1

    # Iteration
    for epoch in range(0,Config.train_epochs):
        # one epoch training
        net.train()
        train_loss = 0
        for i, data in enumerate(train_dataLoader,0):
            # one batch training
            image,labels = data
            image,labels = image.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(image)
            labels = labels.reshape(-1,1)
            l = loss(output,labels)
            train_loss += l.sum().item()
            l.mean().backward()
            optimizer.step()
        
        train_loss /= len(train_dataset)

        # one epoch validation
        net.eval()
        valid_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valid_dataLoader,0):
                image, labels = data
                image, labels = image.to(device), labels.to(device)
                output = net(image)
                labels = labels.reshape(-1,1)
                l = loss(output,labels)
                valid_loss += l.sum().item()
        
        valid_loss /= len(valid_dataset)

        if best_valid_loss == -1:
            best_valid_loss = valid_loss
            torch.save(net.state_dict(),model_file)
        elif best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            torch.save(net.state_dict(),model_file)
        
        print('\nTest set: Average loss: {:.4f}, (best: {:.4f})'.format(
                valid_loss, best_valid_loss))
        
        f = open(output_file,'a')
        f.write(" %3d %12.6f %12.6f %12.6f \n" % (
        epoch+1, train_loss, valid_loss, best_valid_loss))
        f.close()
        writer.add_scalar(tag="Train Loss",scalar_value=train_loss,global_step=epoch+1)
        writer.add_scalar(tag="Valid Loss",scalar_value=valid_loss,global_step=epoch+1)

        # update
        lr_scheduler.step()
    writer.close()

############################################################
#  Start!
############################################################

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str)
    parser.add_argument("--model_id",type=int)
    args = parser.parse_args()
    # Random Seed
    setup_seed(Config.seed)
    # Train
    for i in range(0,Config.k_fold):
        print('\nStart [%1dth/%d] fold validation!\n'% (i+1,Config.k_fold))
        train(k_fold = Config.k_fold, fold_id = i, model_path = args.model_path, model_id = args.model_id)
    end_time = time.time()
    print("Total time: "+ str(end_time-start_time))