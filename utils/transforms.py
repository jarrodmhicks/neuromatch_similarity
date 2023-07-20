import torch
from torch import nn

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, features):
    return features # return features

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

#TODO: Make sure this works as expected
class NNtransform(nn.Module):
  def __init__(self, transform_layers):
    super(NNtransform, self).__init__()
    self.network = self.build_network(transform_layers)

  def build_network(self, transform_layers):
    layers = []
    for layer in transform_layers:
      layers.append(layer)
      return nn.Sequential(*layers)

  def forward(self, features):
    transformed_features = self.network(features)
    return transformed_features
