import sys
sys.path.append('/om2/user/jmhicks/projects/NeuroMatchSimilarity/code/')
import neuromatch_similarity as nms
import os
import torch

# parameters to loop over
learning_rates = [0.1, 0.01, 0.001, 0.0001]
num_epochs = [100, 1000]
networks = [{'model_name': 'alexnet',
             'layer_name': 'classifier.5'}, 
             {'model_name': 'vgg16_bn',
             'layer_name': 'classifier.4'},
             {'model_name': 'clip',
             'layer_name': 'visual'}]
projection_dims = (2 ** torch.linspace(1, 9, 9)).int().tolist() # 2 to 512

# fixed parameters
seed = 0
summary_every = 10
hold_out = 0.2
batch_size = 10000
optimizer = torch.optim.Adam
loss_function = torch.nn.CrossEntropyLoss
distance = nms.utils.distances.Cosine
transform = nms.utils.transforms.LinearProjection

def make_and_save_params():
    param_dir = nms.utils.helpers.relative('../params/')
    if not os.path.exists(param_dir):
         os.mkdir(param_dir)

    counter = 0
    for lr in learning_rates:
        for n_epochs in num_epochs:
                for network in networks:
                    for projection_dim in projection_dims:
                        counter += 1
                        params = {'device': nms.utils.helpers.set_device,
                                  'seed': seed,
                                  'network_features': network,
                                  'model': {'transform': transform,
                                            'transform_kwargs': {'out_features': projection_dim},
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
    for network in networks: # run again but only once per network with identity transform
         counter += 1
         params = {'device': nms.utils.helpers.set_device,
                   'seed': seed,
                   'network_features': network,
                   'model': {'transform': nms.utils.transforms.Identity,
                             'transform_kwargs': None,
                             'distance': distance},
                             'training': {'learning_rate': lr,
                                          'loss_function': loss_function,
                                          'optimizer': optimizer,
                                          'num_epochs': n_epochs,
                                          'summary_every': summary_every,
                                          'batch_size': batch_size,
                                          'hold_out': hold_out}}

if __name__ == '__main__':
    make_and_save_params()