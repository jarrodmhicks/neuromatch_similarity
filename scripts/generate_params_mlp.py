import sys
sys.path.append('/om2/user/jmhicks/projects/NeuroMatchSimilarity/code/')
import neuromatch_similarity as nms
import os
import torch
from glob import glob
import numpy as np

# parameters to loop over
learning_rates = [0.1, 0.01, 0.001, 0.0001]

networks = [{'model_name': 'alexnet',
             'layer_name': 'classifier.5'},
            {'model_name': 'clip',
             'layer_name': 'visual'},
            {'model_name': 'efficientnet_b0_None',
             'layer_name': 'classifier.0'},
            {'model_name': 'resnet50_None',
             'layer_name': 'avgpool'},
            {'model_name': 'vgg16_None',
             'layer_name': 'classifier.4'},
            {'model_name': 'Harmonization_EfficientNetB0',
             'layer_name': 'avg_pool'},
            {'model_name': 'Harmonization_ResNet50',
             'layer_name': 'avg_pool'},
            {'model_name': 'Harmonization_VGG16',
             'layer_name': 'fc2'}]

# fixed parameters
seed = 0
summary_every = 5
hold_out = 0.2
batch_size = 100000
num_epochs = 100
optimizer = torch.optim.Adam
loss_function = torch.nn.CrossEntropyLoss
distance = nms.utils.distances.Cosine
transform = nms.utils.transforms.MLP
transform_kwargs = {'out_features': 16,
                    'hidden_features': [512, 256, 128, 64, 32]}

def make_and_save_params():
    files = glob('../params/model*_params.pt')
    file_nums = []
    for file in files:
        file_nums.append(file[15:-10])
    counter = np.max(np.array(file_nums, dtype=int))
    
    for lr in learning_rates:
        for network in networks:
            counter += 1
            params = {'device': nms.utils.helpers.set_device,
                        'seed': seed,
                        'network_features': network,
                        'model': {'transform': transform,
                                'transform_kwargs': transform_kwargs,
                                'distance': distance},
                        'training': {'learning_rate': lr,
                                    'loss_function': loss_function,
                                    'optimizer': optimizer,
                                    'num_epochs': num_epochs,
                                    'summary_every': summary_every,
                                    'batch_size': batch_size,
                                    'hold_out': hold_out}}
            filename = nms.utils.helpers.relative(f'../params/model{counter}_params.pt')
            torch.save(params, filename)        

if __name__ == '__main__':
    make_and_save_params()