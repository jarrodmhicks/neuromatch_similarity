from neuromatch_similarity import utils
import torch
import os

# example usage

# import neuromatch_similarity as nms
# params = {'device': nms.utils.helpers.set_device,
#           'seed': 2021,
#           'network_features': {'model_name': 'alexnet',
#                                'layer_name': 'classifier.5'},
#           'model': {'transform': nms.utils.transforms.LinearProjection,
#                     'transform_kwargs': {'out_features': 10},
#                     'distance': nms.utils.distances.Cosine},
#           'training': {'learning_rate': 0.01,
#                        'loss_function': nn.CrossEntropyLoss,
#                        'optimizer': torch.optim.Adam,
#                        'num_epochs': 1,
#                        'summary_every': 1,
#                        'batch_size': 1000,
#                        'hold_out': 0.2}}
# train_losses, optimized_model, test_performance = nms.fit(params)
# filename = os.path.relpath('../results/model_info.pt')
# nms.save(filename, params, train_losses, optimized_model, test_performance)

def fit(params):
    train_loader, test_loader = utils.datasets.LoadSimilarityDataset(batch_size=params['training']['batch_size'],
                                                                   seed=params['seed'],
                                                                   device=params['device'](),
                                                                   hold_out=params['training']['hold_out'])
    network_features = utils.datasets.NetworkFeatures(model_name=params['network_features']['model_name'],
                                                    layer_name=params['network_features']['layer_name'],
                                                    device=params['device']())
    in_features = network_features.shape[1]
    if not params['model']['transform_kwargs']:
            transform = params['model']['transform'](in_features)
    else:
        transform = params['model']['transform'](in_features, **params['model']['transform_kwargs'])    
    distance = params['model']['distance']()
    model = utils.model.DistanceModel(transform, distance).to(params['device']())

    if not (list(model.parameters())): # if model has no trainable parameters (e.g., in zero-shot case using nms.utils.transforms.Identity)
        train_losses = []        
        optimized_model = model
    else:         
        loss_function = params['training']['loss_function']()
        optimizer = params['training']['optimizer'](model.parameters(), 
                                                    lr=params['training']['learning_rate'])
        train_losses, optimized_model = utils.model.train(network_features, model, optimizer, 
                                                        loss_function, train_loader, 
                                                        params['training']['num_epochs'], 
                                                        params['training']['summary_every'])    
    test_performance = utils.model.test(network_features, optimized_model, test_loader)

    return train_losses, optimized_model, test_performance

def save(filename, params, train_losses, optimized_model, test_performance):
     model_info = {'params': params,
                   'train_losses': train_losses,
                   'model_state_dict': optimized_model.state_dict(),
                   'test_performance': test_performance}
     
     path, _ = os.path.split(filename)
     if not os.path.exists(path):
          os.mkdir(path)

     torch.save(model_info, filename)