import os
import requests
import random
from io import BytesIO
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from neuromatch_similarity.utils.helpers import set_seed
from neuromatch_similarity.utils.helpers import seed_worker

class SimilarityDataset(Dataset):
  def __init__(self, device='cpu'):

    human_data_link = 'https://osf.io/download/h2smy/'
    df = pd.read_csv(human_data_link, delimiter='\t')
    image_indices = torch.tensor(np.array(df[['image1', 'image2', 'image3']]), device=device)-1
    human_decisions = torch.squeeze(torch.tensor(np.array(df[['choice']]), device=device)-1)

    self.total_samples = len(human_decisions)
    self.decisions = human_decisions
    self.triplets = image_indices

  def __len__(self):
    return self.total_samples

  def __getitem__(self, index):
    triplet = self.triplets[index,:]
    decision = self.decisions[index]
    return triplet, decision

def UnbiasedTrainTestSplit(dataset, seed=2021, device='cpu', hold_out=0.1):
  set_seed(seed=seed)
  total_images = len(torch.unique(dataset.triplets)) #count total number of unique images
  n_test_images = int(round(total_images * (hold_out))) #determine number of hold out images
  n_train_images = int(total_images - n_test_images)

  test_images=torch.squeeze(torch.tensor(random.sample(range(total_images-1), n_test_images)))
  total_samples=dataset.triplets.shape[0]
  total_sample_ind=torch.tensor(range(0, total_samples), device=device)
  n_stimuli=dataset.triplets.shape[1]
  ind=torch.zeros(total_samples, device=device)
  for stim in range(n_stimuli):
    for im in test_images:
      ind=ind+(dataset.triplets[:, stim] == im)

  #testing data must have three testing images (ind==3)
  #training data must have no testing images (ind==0)
  #all else must be discarded (e.g., ind!= & ind!=3)
  test_indices = total_sample_ind[(ind==3)]
  train_indices = total_sample_ind[(ind==0)]

  return train_indices, test_indices

def LoadSimilarityDataset(batch_size, seed=2021, device='cpu', hold_out=0.01):
  dataset = SimilarityDataset(device=device)
  train_indices, test_indices = UnbiasedTrainTestSplit(dataset, seed=seed, device=device, hold_out=hold_out)
  
  set_seed(seed)
  g_seed = torch.Generator()
  g_seed.manual_seed(seed)
  train_dataset = Subset(dataset, train_indices)
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            worker_init_fn=seed_worker,
                            generator=g_seed)
  
  set_seed(seed)
  g_seed = torch.Generator()
  g_seed.manual_seed(seed)
  test_dataset = Subset(dataset, test_indices)
  test_loader = DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           worker_init_fn=seed_worker,
                           generator=g_seed)
  
  return train_loader, test_loader

def NetworkFeatures(model_name, layer_name, device='cpu'):
  url_prefix = 'https://mcdermottlab.mit.edu/jmhicks/neuromatch/network_features'
  feature_dir = f'features_{model_name}_{layer_name}'
  url = os.path.join(url_prefix, feature_dir, 'features.npy')
  response = requests.get(url)
  network_features = torch.from_numpy(np.load(BytesIO(response.content))).to(dtype=torch.float32, device=device)
  return network_features