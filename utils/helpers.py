import torch

def train_model(features, model, optimizer, loss_function, train_loader, summary=True):
  n_steps = len(train_loader)
  for i, (triplets, targets) in enumerate(train_loader):
    inputs = features[triplets,:]

    # forward pass
    outputs = model(inputs)
    loss = loss_function(outputs, targets)

    # backward and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print summary on last step of epoch
    if ((i+1) == n_steps) & summary:
      print(f'\tStep [{i+1}/{n_steps}], Loss: {loss.item():.6f}')

  return loss.item()

def model_performance(outputs, targets):
  model_decisions = torch.argmax(outputs, dim=1)
  performance = torch.sum(model_decisions == targets) / len(targets)
  return performance.item()

def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Current device: {device}')
    return device


# TODO: write saving utility
'''
should save final model, performance, and specifications (e.g,. which network
activations were used, which transform, which distance, what seed)
'''