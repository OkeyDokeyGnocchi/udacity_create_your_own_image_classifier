# functions for train and predict
import argparse
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


def get_train_args():
    """
    get and parse command line arguments for train.py when run from terminal.
    First argument data_dir is required.
    
    Optional Arguments (flags):
    1. set the directory for saving checkpoints
    2. choose an architecture (pretrained from torchvision models)
    3. set parameters [learning_rate, hidden units, epochs)
    4. use gpu for training
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', action='store',
                        help='Data directory containing images')
    
    parser.add_argument('--save_dir', action='store', dest='save_dir',
                        type=str, default='checkpoint.pth',
                        help='Path to directory in which to save checkpoints. Default: checkpoint.pth')
    
    parser.add_argument('--arch', action='store', dest='arch',
                        type=str, default='vgg16',
                        help='Pretrained model to use for training. Default: VGG16')
    
    parser.add_argument('--learn_rate', action='store', dest='learn_rate',
                        type=float, default=0.001,
                        help='The learning rate of the model. Default: 0.001')
    
    parser.add_argument('--hidden_units', action='store', dest='hidden_units',
                        type=int, default=512,
                        help='The number of hidden units used. Default: 512')
    
    parser.add_argument('--epochs', action='store', dest='epochs',
                        type=int, default=2,
                        help='The number of epochs to conduct training over. Default: 2')
    
    parser.add_argument('--gpu', action='store_true',
                        help='Turns on gpu for training if applicable')
    
    arguments = parser.parse_args()
    
    return arguments

def get_predict_args():
    """
    get and parse command line arguments for predict.py when run from terminal.
    1. topk to get the number of top predictions (e.g. top_k=3 gives 3)
    2. map categories to the real names
    3. use gpu for prediction
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_path', action='store',
                        help='Path to the image to predict')
    
    parser.add_argument('checkpoint', action='store',
                        help='Path to the checkpoint for the system to use')
    
    parser.add_argument('--top_k', action='store', dest='top_k',
                        type=int, default=3,
                       help='The number of most likely classes to return')
    
    parser.add_argument('--category_names', action='store', dest='cat_file',
                        type=str, default='cat_to_name.json',
                        help='Path to the mapping for categories to real names')
    
    parser.add_argument('--gpu', action='store_true',
                        help='Turns on gpu for training if applicable')
    
    parser.add_argument('--arch', action='store', dest='arch',
                        type=str, default='vgg16',
                        help='Pretrained model to use for training. Default: VGG16')
    
    arguments = parser.parse_args()
    
    return arguments

def make_classifier(model, num_inputs, hidden_units):
    # freeze the pretrained parameters
    for param in model.parameters():
        torch.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(num_inputs, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    
    return model

def train_model(model, device, criterion, optimizer, epochs, trainloader, validloader, gpu):
    steps = 0
    train_loss = 0
    print_every = 10
    
    model.to(device) 

    # get it running with keep_awake so it doesn't shut off
    for e in range(epochs):
        # get images, labels from trainloader to train
        for images, labels in trainloader:
            # increment our steps taken
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # every 10 iterations we'll validate
            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0
                model.eval()

                # with the model set to eval mode, and no_grad set get validloader
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        log_ps = model.forward(images)
                        valid_loss += criterion(log_ps, labels).item()

                        # get probabilities by exp(the log_probs)
                        ps = torch.exp(log_ps)

                        # get the top probabilities
                        top_p, top_class = ps.topk(1, dim=1)

                        # get number of correct answers, add the mean to accuracy
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor))

                # print our results so far
                print(f"Epoch {e+1}/{epochs}... "
                      f"Training Loss: {train_loss/print_every:.3f}... "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}... "
                      f"Validation Accuracy: {valid_accuracy/len(validloader):.3f}")

                # reset our running training loss, re-enable training mode
                train_loss = 0
                model.train()
            
    print("***Finished Training***")
    
    return model, optimizer

def test_model(model, device, testloader, criterion, gpu):
    model.eval()
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            log_ps = model.forward(images)
            test_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))

        return print(f"Test Loss: {test_loss/len(testloader):.3f}... "
                     f"Test Accuracy: {test_accuracy/len(testloader):.3f}")

def save_checkpoint(model, epochs, optimizer, train_data, save_dir):
    checkpoint = {'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': train_data.class_to_idx}

    torch.save(checkpoint, save_dir)
    
    return print('\nCheckpoint saved!')

def set_device(gpu):
    if gpu == True:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")

def load_checkpoint(model, device, filepath):
    checkpoint = torch.load(filepath)
    epochs = checkpoint['epochs']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    
    # set the vgg parameters to remain unchanged
    for param in model.parameters():
        param.requires_grad = False
    
    model.to(device)
    
    return model

def predict(proc_image, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # load model and put it in eval mode
    model.eval()
    
    # get the image and make it a torch tensor
    image = proc_image
    image = torch.from_numpy(image).to(device)
    image = image.unsqueeze(0)
    
    with torch.no_grad():     
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        probs, classes = ps.topk(topk, dim=1)
        
    # move to cpu in case of cuda so we can use numpy
    probs, classes = probs.cpu(), classes.cpu()
    
    # convert to numpy
    probs_np = np.array(probs)[0]
    classes_np = np.array(classes)[0]
    
    # use class_to_idx
    class_to_idx = model.class_to_idx
    
    # invert as mentioned above
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    
    # indexes to classes
    classes_list = [idx_to_class[i] for i in classes_np]
    
    return probs_np, classes_list