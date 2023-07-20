import torch
import os.path as osp

def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Current device: {device}')
    return device

def relative(relative_path):    
    return osp.join(osp.dirname(osp.abspath(__file__)), relative_path)

# TODO: write saving utility
'''
should save final model, performance, and specifications (e.g,. which network
activations were used, which transform, which distance, what seed)
'''