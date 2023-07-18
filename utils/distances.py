import torch
from torch import nn

class CosineDistance(nn.Module):
  def __init__(self):
    super(CosineDistance, self).__init__()
    self.cos_sim = nn.CosineSimilarity(dim=1)

  def forward(self, input1, input2):
    return 1 - self.cos_sim(input1, input2)

# TODO: other distance metrics can go here