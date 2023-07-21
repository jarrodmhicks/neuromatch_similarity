import torch
from torch import nn

class Identity(nn.Module):
  def __init__(self, in_features):
    super(Identity, self).__init__()

  def forward(self, features):
    return features 

class LinearDiagonal(nn.Module):
  def __init__(self, n_features, sigma=0.01):
    super(LinearDiagonal, self).__init__()
    self.diagonal = nn.Parameter(torch.randn(n_features) * sigma)

  def forward(self, features):
    return features * self.diagonal # equivalent to features @ torch.diag(self.diagonal)

class LinearFull(nn.Module):
  def __init__(self, n_features, sigma=0.01):
    super(LinearFull, self).__init__()
    self.transformation_matrix = nn.Parameter(torch.randn(n_features, n_features) * sigma)

  def forward(self, features):
    return features @ self.transformation_matrix

class LinearProjection(nn.Module):
  def __init__(self, in_features, out_features, sigma=0.01):
    super(LinearProjection, self).__init__()
    self.projection_matrix = nn.Parameter(torch.randn(in_features, out_features) * sigma)

  def forward(self, features):
    return features @ self.projection_matrix

class MLP(nn.Module):
  def __init__(self, in_features, out_features, hidden_features=[], use_bias=True):    
    super(MLP, self).__init__()

    self.in_features = in_features
    self.out_features = out_features

    # If we have no hidden layer, just initialize a linear model
    if len(hidden_features) == 0:
      layers = [nn.Linear(in_features, out_features, bias=use_bias)]
    else:
      # MLP with dimensions in_dim - num_hidden_layers*[hidden_dim] - out_dim
      layers = [nn.Linear(in_features, hidden_features[0], bias=use_bias), nn.ReLU()]

      # Loop until before the last layer
      for i, hidden_dim in enumerate(hidden_features[:-1]):
        layers += [nn.Linear(hidden_dim, hidden_features[i + 1], bias=use_bias),
                   nn.ReLU()]

      # Add final layer to the number of classes
      layers += [nn.Linear(hidden_features[-1], out_features, bias=use_bias)]

    self.mlp = nn.Sequential(*layers)

  def forward(self, features):   
    return self.mlp(features)

#TODO: Make sure this works as expected
class NeuralNet(nn.Module):
  def __init__(self, transform_layers):
    super(NeuralNet, self).__init__()
    self.network = self.build_network(transform_layers)

  def build_network(self, transform_layers):
    layers = []
    for layer in transform_layers:
      layers.append(layer)
      return nn.Sequential(*layers)

  def forward(self, features):
    return self.network(features)