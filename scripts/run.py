from argparse import ArgumentParser
import os
import torch
from torch import nn
import sys
sys.path.append('/om2/user/jmhicks/projects/NeuroMatchSimilarity/code/')
import neuromatch_similarity as nms

def load_params(params_name):
    params = torch.load(params_name) # path to file saved as 'model#_params.pt'
    _, file_name = os.path.split(params_name)
    prefix, _ = os.path.splitext(file_name)
    model_name = prefix[:-7] # removes '_params' from file name
    params['model_name'] = model_name
    return params

def main():
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, help='path to experiment parameters file')    
    args = parser.parse_args()
    params = load_params(args.params)
    train_losses, optimized_model, test_performance = nms.fit(params)
    filename = nms.utils.helpers.relative(f'../results/{params["model_name"]}.pt')
    nms.save(filename, params, train_losses, optimized_model, test_performance)

if __name__ == "__main__":
    main()