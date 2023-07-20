import torch
from torch import nn

class Cosine(nn.Module):
  def __init__(self):
    super(Cosine, self).__init__()
    self.cos_sim = nn.CosineSimilarity(dim=1)

  def forward(self, input1, input2):
    return 1 - self.cos_sim(input1, input2)

def Euclidean():
  return nn.PairwiseDistance(p=2)

# TODO: other distance metrics can go here