import torch
from torch import nn, optim
from torchvision import models

from functions import (get_train_args, make_classifier, save_checkpoint,
                       set_device, test_model, train_model)
from utility_functions import load_data


# get our command line arguments and store as 'arguments'
arguments = get_train_args()

# separate arguments into their needed variables
data_dir = arguments.data_dir
save_dir = arguments.save_dir
arch = arguments.arch
learn_rate = arguments.learn_rate
hidden_units = arguments.hidden_units
epochs = arguments.epochs
gpu = arguments.gpu

# set torch.device cuda/cpu
device = set_device(gpu)

# set all of our data with transforms and set our data loaders
print('Loading data...')
train_data, trainloader, valid_data, validloader, test_data, testloader = load_data(data_dir)

# get the specified model
model = getattr(models, arch)(pretrained=True)

# get input layer node number and sub in the classifier
num_inputs = model.classifier[0].in_features
print('\nMaking classifier with ' + str(num_inputs) + ' inputs and ' +
      str(hidden_units) + ' hidden nodes...')
make_classifier(model, num_inputs, hidden_units)

# set our criterion and optimizer for training
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

# train and validate the model, setting the new model and optimizer from results
print('\nStarting training: \n')
model, optimizer = train_model(model, device, criterion, optimizer, epochs, trainloader, validloader, gpu)

# save a checkpoint of the newly-trained model
print('\nSaving checkpoint as ' + '"' + save_dir + '"')
save_checkpoint(model, epochs, optimizer, train_data, save_dir)