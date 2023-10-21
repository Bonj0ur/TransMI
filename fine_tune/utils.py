import torch

############################################################
#  Utils
############################################################

# try to use the i-th GPU
def try_gpu(i = 0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# try to use all the gpus
def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]