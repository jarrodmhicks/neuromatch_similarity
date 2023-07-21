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
class util_log:
    def __init__(self):
        self.device = ""
        self.seed = 1
        self.model = {
            "data_transform": "",
            "model": ""
        }
        self.data = {
            "human_decision": "",
            "feature_map": ""
        }
        self.utils = {
            "loss_function": "",
            "optimizer": "",
            "learning_rate": 0.01,
            "n_trials": 100000,
            "hold_out": 0.1,
            "num_epochs": 30,
            "batch_size": 10000,
            "summary_every": 10,
            "momentum": "",
            "regularizer": ""
        }
        self.results = {
            "train": {
                "zero_shot": 0,
                "transformed": 0
            },
            "test": {
                "zero_shot": 0,
                "transformed": 0
            }
        }
        self.find("device", device)
        self.find("seed", seed)
        self.find("[model]", model)
        self.find("[data][human_decision]", human_decision)
        self.find("[data][feature_map]", feature_map)
        self.find("[utils][loss_function]", loss_function)
        self.find("[utils][optimizer]", optimizer)
        self.find("[utils][learning_rate]", learning_rate)
        self.find("[utils][n_trials]", n_trials)
        self.find("[utils][hold_out]", hold_out)
        self.find("[utils][num_epochs]", num_epochs)
        self.find("[utils][batch_size]", batch_size)
        self.find("[utils][summary_every]", summary_every)
        # self.find("[utils][momentum]", momentum)
        # self.find("[utils][regularizer]", regularizer)
        self.find("[results][train][zero_shot]", train_zero_shot)
        self.find("[results][train][transformed]", train_transformed)
        # self.find("[results][test][zero_shot]", test_zero_shot)
        # self.find("[results][test][transformed]", test_transformed)

    def __str__(self):
        return str(self.__dict__)

    def find(self, key, value):
        keys = key.split('][')
        current_dict = self.__dict__
        for k in keys[:-1]:
            k = k.strip('[]')
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        last_key = keys[-1].strip('[]')
        if last_key in current_dict:
            current_dict[last_key] = value
