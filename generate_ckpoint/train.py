import os
import time
import torch
import random
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

def train():
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
        "../logs/%s/log.out" % (Config.log_dir))
    model_file = str(
        "../logs/%s/model.pth" % (Config.log_dir))
    tb_dir = "../logs/%s/tb/" % (Config.log_dir)
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
                                          transform = transform)
    train_dataLoader = DataLoader(train_dataset,
                                  shuffle = True,
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
        train_corr = 0
        for i, data in enumerate(train_dataLoader,0):
            # one batch training
            image0, image1, labels = data
            image0, image1, labels = image0.to(device), image1.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(image0,image1)
            l = loss(output,labels)
            train_loss += l.sum().item()
            train_corr += (output.argmax(1)==labels).sum().item()
            l.mean().backward()
            optimizer.step()

            # Print
            if (i+1)%10 == 0:
                print('Train Epoch: {} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1,(i+1)*len(image0),len(train_dataset),100.*(i+1)/len(train_dataLoader),
                    l.mean().item()
                ))
        
        train_loss /= len(train_dataset)

        if max_correct < train_corr:
            torch.save(net.state_dict(),model_file)
            max_correct = train_corr
            print("Best accuracy! correct cls: %5d" % train_corr)
        
        train_accuracy = 100*train_corr/len(train_dataset)
        best_train_accuracy = 100*max_correct/len(train_dataset)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) (best: {:.2f}%)\n'.format(
            train_loss, train_corr, len(train_dataset), train_accuracy, best_train_accuracy))

        f = open(output_file,'a')
        f.write(" %3d %12.6f %9.3f %9.3f\n" % (
            epoch+1, train_loss, train_accuracy, best_train_accuracy))
        f.close()
        writer.add_scalar(tag="Train Loss",scalar_value=train_loss,global_step=epoch+1)
        writer.add_scalar(tag="Train Accuracy",scalar_value=train_accuracy,global_step=epoch+1)

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
    train()
    end_time = time.time()
    print("Total time: "+ str(end_time-start_time))