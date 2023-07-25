import torch
from torch import nn
import copy
from neuromatch_similarity.utils.helpers import relative, save_checkpoint

class DistanceModel(nn.Module):
  def __init__(self, transform, distance):
    super(DistanceModel, self).__init__()
    self.transform = transform
    self.distance = distance

  def forward(self, features): # input features: batch dim by 3 by n features
    transformed_features = self.transform(features) # batch dim by 3 by k features
    distances = torch.hstack((torch.unsqueeze(self.distance(transformed_features[:, 1, :], transformed_features[:, 2, :]), 1),
                              torch.unsqueeze(self.distance(transformed_features[:, 0, :], transformed_features[:, 2, :]), 1),
                              torch.unsqueeze(self.distance(transformed_features[:, 0, :], transformed_features[:, 1, :]), 1)))
    return -distances # batch dim by 3; (-) to convert distance to "similarity" to work with softmax in CrossEntropy loss

def model_performance(outputs, targets):
  model_decisions = torch.argmax(outputs, dim=1)
  performance = torch.sum(model_decisions == targets) / len(targets)
  return performance.item()

def train_epoch(features, model, optimizer, loss_function, train_loader, summary=True):
  n_steps = len(train_loader)
  for i, (triplets, targets) in enumerate(train_loader):
    inputs = features[triplets,:]
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print summary on last step of epoch
    if ((i+1) == n_steps) & summary:
      print(f'\tStep [{i+1}/{n_steps}], Loss: {loss.item():.6f}')
  return loss.item()

def train(model_name, features, model, optimizer, loss_function, train_loader, 
          num_epochs, summary_every, patience=10, tol=0.001):
  train_losses=[]
  curr_model = copy.deepcopy(model)
  counter = 0
  for epoch in range(num_epochs):

    #print training progress
    if (epoch+1) % summary_every == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}]')
      summary = True
    else:
      summary = False
   
  # compute loss for current training epoch
    train_loss = train_epoch(features, model, optimizer, loss_function, train_loader, summary=summary)
    train_losses.append(train_loss)
    if epoch > 0:
      #check loss tolerance
      if (-torch.diff(torch.tensor(train_losses[-2:]))>tol): #if model improves greater than tolerance
        curr_model = copy.deepcopy(model) #update current best model
        counter = 0 #reset counter to 0
        filename = relative(f'../checkpoints/{model_name}.pt')
        save_checkpoint(filename, train_losses, curr_model)
      else:
        counter = counter + 1
    
    if counter == patience: #stop training if model no longer improves
      break
  return train_losses, curr_model

def test(network_features, model, test_loader):
  performances = []
  weights = []
  for i, (triplets, targets) in enumerate(test_loader):
      inputs = network_features[triplets,:]
      outputs = model(inputs)
      performances.append(model_performance(outputs, targets))
      weights.append(len(inputs))
  performances = torch.tensor(performances)
  weights = torch.tensor(weights)
  weights = weights / weights.sum()
  final_performance = torch.sum(weights * performances).item()
  return final_performance
