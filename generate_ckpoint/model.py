import torch
import torchvision
import torch.nn as nn

############################################################
#  Network
############################################################

class SiamResNet18(nn.Module):
    def __init__(self):
        super(SiamResNet18,self).__init__()
        self.backbone_net = torchvision.models.resnet18(pretrained=False)
        self.backbone_net.fc = nn.Linear(self.backbone_net.fc.in_features,32)
        nn.init.xavier_uniform_(self.backbone_net.fc.weight)
        self.cls1 = nn.Linear(64,32)
        self.cls2 = nn.Linear(32,20)
        nn.init.xavier_uniform_(self.cls1.weight)
        nn.init.xavier_uniform_(self.cls2.weight)
    
    def forward_once(self,x):
        output = self.backbone_net(x)
        return output

    def forward(self,input1,input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        cct = torch.cat([output1,output2],1)
        output = self.cls2(self.cls1(cct))
        return output