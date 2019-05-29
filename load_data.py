# Imports here
import argparse
import torch
from torchvision import datasets, transforms
import json
import os

# create mapping of flower names to loaded_data_feautre index


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# take arguments 'data_dir',


class Dataset():


    def __init__(self, data_dir):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = data_dir
        # make folders if they don't exist
        s = "/"
        self.train_dir = dir_path + '/' + data_dir + '/' + 'train' + '/'
        self.valid_dir = dir_path + '/' + data_dir + '/' + 'valid' + '/'
        self.test_dir = dir_path + '/' + data_dir + '/' + 'test' + '/'
        try:
            os.mkdir(self.train_dir)
        except FileExistsError:
            print("Directory already Exists")
        try:
            os.mkdir(self.valid_dir)
        except FileExistsError:
            print("Directory already Exists")
        try:
            os.mkdir(self.test_dir)
        except FileExistsError:
            print("Directory already Exists")

        #print(self.train_dir)
        #print(self.valid_dir)
        #print(self.test_dir)
        directory_index = 0
        directory_list = [self.train_dir, self.valid_dir, self.test_dir]
        for filename in os.listdir(data_dir):
            #print(filename)
            if filename == "test" or filename == "train" or filename or "valid":
                #print(filename)
                continue
            else:
                #print(filename)
                put_directory = directory_list[int(directory_index) % 3]
                print(put_directory)
                old_path = self.data_dir + filename
                new_path = put_directory + filename
                os.rename( old_path, new_path)
                directory_index += 1


    #  Define your transforms for the training, validation, and testing sets
    def transform(self):
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])

        test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])

        validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                     (0.229, 0.224, 0.225))])

        #  Load the datasets with ImageFolder
        self.train_data = datasets.ImageFolder(self.train_dir, transform=train_transforms)
        self.valid_data = datasets.ImageFolder(self.valid_dir, transform=validation_transforms)
        self.test_data = datasets.ImageFolder(self.test_dir, transform=test_transforms)


    #  Using the image datasets and the trainforms, define the dataloaders
    def trainloader(self):
        torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)

    def validloader(self):
        torch.utils.data.DataLoader(self.valid_data, batch_size=64)

    def testloader(self):
        torch.utils.data.DataLoader(self.test_data, batch_size=64)


def main():
    parser = argparse.ArgumentParser(description="Caclulate x to the power of Y")
    parser.add_argument("-p", "--path", default="flowers", help="path to image dataset")
    args = parser.parse_args()
    c1 = Dataset(args.path)
    #c1.transform()


if __name__ == "__main__":
    main()