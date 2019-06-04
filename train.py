# Imports here
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torch
import os
import argparse
from load_data import Dataset as DS
import helper



# pass arguments of model to be trained,
# future interaction to choose classifier options

# python train.py data_directory
# print out training loss, validation loss, and validation accuracy


class Model:
    current_dir = os.path.dirname(os.path.realpath(__file__))

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
        self.device=self.setDevice()
        self.dataset=DS(self.datadir)
        self.dataset.transform()
        self.trainloader, self.validloader, self.testloader = self.dataset.init_loaders()

    def __str__(self):
        return '{0.__class__.__name__}:(\n\tarch={0.arch}\n ' \
               '\tlearning_rate={0.learning_rate}\n' \
               '\thidden_units={0.hidden_units}\n' \
               '\tepochs={0.epochs}\n' \
               '\tsave_Dir={0.save_dir}\n' \
               '\tgpu={0.gpu})\n' \
               '\tdevice={0.device}\n'.format(self)

    # Use GPU if it's available
    def setDevice(self):
        if self.gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        else:
            self.device = torch.device("cpu")
        print('device set to {}'.format(self.device))
        
        return self.device
    
    def setModel(self, arch):
        print("\nsetting up model...\n")
#         alexnet = models.alexnet(pretrained=True)
#         squeezenet = models.squeezenet1_0(pretrained=True)
#         vgg11 = models.vgg11(pretrained=True)
#         vgg13 = models.vgg13(pretrained=True)
#         vgg16 = models.vgg16(pretrained=True)
#         vgg19 = models.vgg19(pretrained=True)
#         densenet = models.densenet161(pretrained=True)
#         inception = models.inception_v3(pretrained=True)
#         googlenet = models.googlenet(pretrained=False)
#         shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
#         mobilenet = models.mobilenet_v2(pretrained=True)
#         resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
        switcher = {
            'alexnet' : models.alexnet(pretrained=True),
            'squeezenet': models.squeezenet1_0(pretrained=True),
            'vgg16': models.vgg16(pretrained=True),
            'inception' : models.inception_v3(pretrained=True),
            #'googlenet': models.googlenet(pretrained=False),
            #'mobilenet': models.mobilenet_v2(pretrained=True),
            #'resnext50_32x4d' : models.resnext50_32x4d(pretrained=True),
            'vgg11': models.vgg11(pretrained=True),
            'vgg13': models.vgg13(pretrained=True),
            'vgg16': models.vgg16(pretrained=True),

        }
        error = "\nThat model is not supported yet. The supported models are : 'alexnet', squeezenet',\
        'vgg11', 'vgg13', 'vgg16', 'vgg19', 'inception', 'googlenet', 'mobilenet', 'resnext50_32x4d' "
        self.model = switcher.get(arch, error)
        if self.model == error:
            print(error)
        else: 
            print('\nmodel successfully set to {}'.format(arch))
        
        

    # Freeze parameters so we don't backprop through them

    def create_classifier(self):
        print("\ncreating classifier...")
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4000),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(4000, 1280),
                                         nn.Linear(1280, self.hidden_units),
                                         nn.Linear(self.hidden_units, 102),
                                         nn.LogSoftmax(dim=1))

        self.criterion = nn.NLLLoss()
        
        # Only train the classifier parameters, feature parameters are frozen
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.003)
        #optimizer.zero_grad()
        images, labels = next(iter(self.validloader))
        ps = torch.exp(self.model(images))
        #print("shape should be [64, 102]", ps.shape)
        top_p, top_class = ps.topk(1, dim=1)
        print(top_class[:10, :])
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        print(f'Accuracy: {accuracy.item() * 100}%')
        print('\ncheck results to see if classifer is configured correctly.')
    
    def train_model(self):
        
        device = self.setDevice()
        self.model.to(self.device);
        print('\ntraining {} on {}, for {} epochs. Optimizer learning rate set to {}...'.format(self.arch, self.device, self.epochs, self.learning_rate))
        epochs = self.epochs
        steps = 0
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        train_losses, test_losses = [], []
        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in self.trainloader:
    
                inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                optimizer.zero_grad()
    
                logps = self.model.forward(inputs)
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
                    for images, labels in self.validloader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        log_ps = self.model(images)
                        batch_loss = criterion(log_ps, labels)
    
                        test_loss += batch_loss.item()
    
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
                train_losses.append(running_loss / len(self.trainloader))
                test_losses.append(test_loss / len(self.validloader))
    
                print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / len(self.trainloader)),
                      "Test Loss: {:.3f}.. ".format(test_loss / len(self.validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy / len(self.validloader)))
    

    def validate_model(self):
        test_loss = 0
        accuracy = 0
        test_losses = []
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                log_ps = self.model(images)
                batch_loss = self.criterion(log_ps, labels)
    
                test_loss += batch_loss.item()
    
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
        test_losses.append(test_loss / len(self.testloader))
        print("Test Loss: {:.3f}.. ".format(test_loss / len(self.testloader)),
        "Test Accuracy: {:.3f}".format(accuracy / len(self.testloader))
             )
    
    def save_model_checkpoint(self):
        print("\nOur model: \n\n", self.model, '\n')
        self.model.epochs = self.epochs
        self.model.class_to_idx = self.trainloader.dataset.class_to_idx
        print('model.epochs: ', self.model.epochs)
        print("The state dict keys: \n\n", self.model.state_dict().keys())
        checkpoint_out = self.save_dir + '/' + 'checkpoint2.pth'
        checkpoint = { 'input_size': [3, 224, 224],
                       'output_size': 102,
                       'arch': self.arch,
                       'state_dict': self.model.state_dict(),
                       'epoch': self.model.epochs,
                       'class_to_idx': self.model.class_to_idx
        }
        print('\n\nsaving to {}.'.format(checkpoint_out))
        try:
            torch.save(checkpoint, checkpoint_out)
        except:
            print('Checkpoint did not save.')
        print('Checkpoint successful.')
    

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
    parser.add_argument('data_dir', help="path to image dataset")
    parser.add_argument("-s", "--save_dir", default=".", help="path to saved model")
    parser.add_argument("-a", "--arch", default="vgg13", help="architecture of model")
    parser.add_argument("-r", "--learning_rate", type=float, default=0.03, help="learning rate as float. Default: 0.01.")
    parser.add_argument("--hidden_units", type=int, default="512", help="Number of hidden units. Default: 512.")
    parser.add_argument("-e", "--epochs", type=int, default="20", help="Number of epochs. Default: 20")
    parser.add_argument("--gpu", action='store_true', help="Turn on Cuda Usage")

    args = parser.parse_args()
    print('data directory set to "{}".'.format(args.data_dir))
    myModel = Model(args.data_dir,
                    save_dir=args.save_dir,
                    arch=args.arch,
                    learning_rate=args.learning_rate,
                    hidden_units=args.hidden_units,
                    epochs=args.epochs,
                    gpu=args.gpu,)

    print(myModel)
    myModel.setModel(myModel.arch)
    myModel.create_classifier()
    myModel.train_model()
    myModel.validate_model()
    myModel.save_model_checkpoint()


if __name__ == "__main__":
    main()