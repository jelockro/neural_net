# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import helper
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import os
import matplotlib.pyplot as plt
import argparse
import load_data.Dataset as DS
testloader = DS.testloader
trainloader = DS.trainloader
validloader = DS.validloader



# pass arguments of model to be trained,
# future interaction to choose classifier options

# python train.py data_directory
# print out training loss, validation loss, and validation accuracy



parser = argparse.ArgumentParser(
    description='This is AI'
)


class Model:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Caclulate x to the power of Y")
    parser.add_argument('datadir', help="path to image dataset")
    parser.add_argument("-s", "--save_dir", help="path to saved model")
    parser.add_argument("-a", "--arch", default="vgg13", help="architecture of model")
    parser.add_argument("-r", "--learning_rate", type=float, default=0.01, help="learning rate as float. Default: 0.01.")
    parser.add_argument("--hidden_units", type=int, default="512", help="Number of hidden units. Default: 512.")
    parser.add_argument("-e", "--epochs", type=int, default="20", help="Number of epochs. Default: 20")
    parser.add_argument("--gpu", action='store_true', help="Turn on Cuda Usage")


    def __init__(self, datadir, save_dir=current_dir, **kwargs ):
        self.datadir = datadir
        self.save_dir = save_dir
        self.gpu = False
        for k, v in kwargs.items():
            setattr(self, k, v)
            ''' Default are:
                    arch=vgg13,
                    learning_rate=0.01
                    hidden_units=512
                    epochs=20
                    gpu=False
                    '''

    # Use GPU if it's available
    def setDevice(self):
        if self.gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg11(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4000),
                                     nn.ReLU(),
                                     # nn.Dropout(0.2),
                                     nn.Linear(4000, 1280),
                                     nn.ReLU(),
                                     nn.Linear(1280, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    images, labels = next(iter(testloader))
    # print('labels', labels)
    # print (images.size())
    # images = images[1:3]
    # print('images', images.size())
    # images = images.view(1, -1)

    # print(images.size())
    # images = images.view(images.size(0), -1)
    # print(images.size())
    ps = torch.exp(model(images))
    print("shape should be [64, 102]", ps.shape)
    top_p, top_class = ps.topk(1, dim=1)
    print(top_class[:10, :])
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item() * 100}%')


    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device);
        epochs = 30
        steps = 0

        train_losses, test_losses = [], []
        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in trainloader:

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            else:
                ## TODO: Implement the validation pass and print out the validation accuracy
                test_loss = 0
                accuracy = 0

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model(images)
                        batch_loss = criterion(log_ps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(validloader))

                print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
                      "Test Loss: {:.3f}.. ".format(test_loss / len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy / len(validloader)))


    def plot_test(self, train_losses, test_losses):
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)

    def validate_model(self, testloader, model):
        test_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                batch_loss = criterion(log_ps, labels)

                test_loss += batch_loss.item()

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        test_loss.append(test_loss / len(validloader))

        "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
        "Test Accuracy: {:.3f}".format(accuracy / len(testloader))

    def save_model_checkpoint(self, model):
        print("Our model: \n\n", model, '\n')
        print("The state dict keys: \n\n", model.state_dict().keys())

        torch.save(model.state_dict(), 'checkpoint.pth')

    def load_model_checkpoint(self, checkpoint):
        # TODO: Write a function that loads a checkpoint and rebuilds the model
        from torchvision import datasets, transforms, models
        def rebuild(checkpoint):
            model = models.vgg11(pretrained=True)
            model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4000),
                                             nn.ReLU(),
                                             # nn.Dropout(0.2),
                                             nn.Linear(4000, 1280),
                                             nn.ReLU(),
                                             nn.Linear(1280, 102),
                                             nn.LogSoftmax(dim=1))
            state_dict = torch.load(checkpoint)
            model.class_to_idx = train_data.class_to_idx
            print(model.class_to_idx)
            model.load_state_dict(state_dict)
            print(model)
            return model

        loadedModel = rebuild('checkpoint.pth')


####################
# Examples of usage
####################
#
# Set directory to save checkpoints:
# python train.py data_dir --save_dir save_directory
#
# Choose Architecture of model:
# python train.py data_dir --arch "vgg13"
#
# Set Hyperparameters:
# python train.py data_dir --learning_rate 0.01 --hidden_units 512\
# --epochs 20
#
# Use GPU for Training:
# python train.py data_dir --gpu
def main():
    parser = argparse.ArgumentParser(description="Caclulate x to the power of Y")
    parser.add_argument('data-dir', help="path to image dataset")
    parser.add_argument("-s", "--save_dir", default=".", help="path to saved model")
    parser.add_argument("-a", "--arch", default="vgg13", help="architecture of model")
    parser.add_argument("-r", "--learning_rate", type=float, default="0.01", help="learning rate as float. Default: 0.01.")
    parser.add_argument("--hidden_units", type=int, default="512", help="Number of hidden units. Default: 512.")
    parser.add_argument("-e", "--epochs", type=int, default="20", help="Number of epochs. Default: 20")
    parser.add_argument("--gpu", action='store_true', help="Turn on Cuda Usage")

    args = parser.parse_args()



if __name__ == "__main__":
    main()