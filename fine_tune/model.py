import torch
import torchvision
import torch.nn as nn

############################################################
#  Network
############################################################

class FineTuneResNet18(nn.Module):
    def __init__(self,model_train_mode,model_path):
        super(FineTuneResNet18,self).__init__()
        if model_train_mode == 1:
            ckpt_path = model_path
            pretrained_dict = torch.load(ckpt_path,map_location='cpu')
            self.backbone_net = torchvision.models.resnet18(pretrained=False)
            self.backbone_net.fc = nn.Linear(self.backbone_net.fc.in_features,32)
            model_dict = self.backbone_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
            model_dict.update(pretrained_dict)
            self.backbone_net.load_state_dict(model_dict)
            self.regressor  = nn.Linear(32,1)
            nn.init.xavier_uniform_(self.regressor.weight)
        elif model_train_mode == 2:
            self.backbone_net = torchvision.models.resnet18(pretrained=False)
            self.backbone_net.fc = nn.Linear(self.backbone_net.fc.in_features,32)
            nn.init.xavier_uniform_(self.backbone_net.fc.weight)
            self.regressor  = nn.Linear(32,1)
            nn.init.xavier_uniform_(self.regressor.weight)

    def forward(self,x):
        output = self.regressor(self.backbone_net(x))
        return output