import sys
sys.path.append('/om2/user/jmhicks/projects/NeuroMatchSimilarity/code/')
import neuromatch_similarity as nms

import torch

# parameters to loop over
learning_rates = [0.1, 0.01, 0.001, 0.0001]
num_epochs = [100, 1000]
transforms = [nms.utils.transforms.Identity, nms.utils.transforms.LinearFull]
distances = [nms.utils.distances.Cosine, nms.utils.distances.Euclidean]
networks = [{'model_name': 'alexnet',
             'layer_name': 'classifier.5'}, 
             {'model_name': 'vgg16_bn',
             'layer_name': 'classifier.4'},
             {'model_name': 'clip',
             'layer_name': 'visual'}]

# fixed parameters
seed = 0
summary_every = 10
hold_out = 0.2
batch_size = 10000
optimizer = torch.optim.Adam
loss_function = torch.nn.CrossEntropyLoss

def make_and_save_params():
    counter = 0
    for lr in learning_rates:
        for n_epochs in num_epochs:
            for transform in transforms:
                for distance in distances:
                    for network in networks:
                        counter += 1
                        params = {'device': nms.utils.helpers.set_device,
                                'seed': seed,
                                'network_features': network,
                                    'model': {'transform': transform,
                                                'transform_kwargs': None,
                                                'distance': distance},
                                    'training': {'learning_rate': lr,
                                                'loss_function': loss_function,
                                                'optimizer': optimizer,
                                                'num_epochs': n_epochs,
                                                'summary_every': summary_every,
                                                'batch_size': batch_size,
                                                'hold_out': hold_out}}
                        filename = nms.utils.helpers.relative(f'../params/model{counter}_params.pt')
                        torch.save(params, filename)

if __name__ == '__main__':
    make_and_save_params()