import pandas as pd
import numpy as np
import torch

class SimilarityDataset(Dataset):
  def __init__(self, hold_out=0.1, device=device):

    human_data_link = 'https://osf.io/download/h2smy/'
    df = pd.read_csv(human_data_link, delimiter='\t')
    image_indices = torch.tensor(np.array(df[['image1', 'image2', 'image3']]), device=device)-1
    human_decisions = torch.squeeze(torch.tensor(np.array(df[['choice']]), device=device)-1)

    self.total_samples = len(human_decisions)
    self.n_train = int(round(self.total_samples * (1 - hold_out)))
    self.n_test = int(self.total_samples - self.n_train)
    self.decisions = human_decisions
    self.triplets = image_indices

  def __len__(self):
    return self.total_samples

  def __getitem__(self, index):
    triplet = self.triplets[index,:]
    decision = self.decisions[index]
    return triplet, decision

# TODO: handle loading pre-computed network activations