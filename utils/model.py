import torch
from torch import nn

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

def train_model(features, model, optimizer, loss_function, train_loader, summary=True):
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

def train(features, model, optimizer, loss_function, train_loader, num_epochs, summary_every):
  train_losses = []
  for epoch in range(num_epochs):
    if (epoch+1) % summary_every == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}]')
      summary = True
    else:
      summary = False
    train_loss = train_model(features, model, optimizer, loss_function, train_loader, summary=summary)
    train_losses.append(train_loss)
  return train_losses