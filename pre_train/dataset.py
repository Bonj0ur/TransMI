import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

############################################################
#  Dataset
############################################################

class SiameseNetworkDataset(Dataset):
    def __init__(self,image_dir,label,fold,fold_id,transform = None,training = True):
        # Initialize
        self.training = training
        self.transform = transform
        self.image_dir = image_dir
        num_per_fold = int(label.shape[0]/fold)
        # Train or Valid
        if self.training:
            self.label = np.vstack((label[:fold_id * num_per_fold,:],label[(fold_id+1)*num_per_fold:,:]))
            self.num_examples = len(self.label)
        else:
            self.label = label[fold_id*num_per_fold:(fold_id+1)*num_per_fold,:]
            self.num_examples = len(self.label)
        # Random Disruption
        self.indices = list(range(self.num_examples))
        random.shuffle(self.indices)
    
    def __getitem__(self,index):
        labels = self.label[self.indices[index]][2]
        image1_dir = self.image_dir + self.label[self.indices[index]][0]
        image2_dir = self.image_dir + self.label[self.indices[index]][1]
        image1 = Image.open(image1_dir)
        image2 = Image.open(image2_dir)
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, labels

    def __len__(self):
        return self.num_examples