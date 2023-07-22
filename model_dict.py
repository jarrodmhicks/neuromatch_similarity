import os

# Configure your models here:
    #Keys 'model_name_shorthand'
    #Values {'model_name': '$model_name', 'module name': '$model_layer', '
configs = {
    'vgg16': {'model_name': 'vgg16', 
            'module_name': 'features.23',
            'source':'torchvision'},
    'resnet50': {'model_name': 'resnet50', 
            'module_name': 'layer4',
            'source':'torchvision'},
    'clip': {'model_name': 'clip', 
            'module_name': 'visual',
            'source':'custom'} # Download @ https://github.com/openai/CLIP
}