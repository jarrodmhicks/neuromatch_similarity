import torch
import os.path as osp
import random
import numpy as np

def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Current device: {device}')
    return device

def relative(relative_path):    
    return osp.join(osp.dirname(osp.abspath(__file__)), relative_path)

def set_seed(seed=None, seed_torch=True):  
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  print(f'Random seed {seed} has been set.')

def seed_worker(worker_id):  
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

# TODO: write saving utility
'''
should save final model, performance, and specifications (e.g,. which network
activations were used, which transform, which distance, what seed)
'''