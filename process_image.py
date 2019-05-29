from PIL import Image
import numpy as np
import torch
import argparse

#### Only accepts .flowers's at the moment, no transform for alpha scales
parser = argparse.ArgumentParser(description="Turns a flowers into pytorch tensor")
parser.add_argument("image_path", action="store", dest="image_path" )
args = parser.parse_args()


def process_image(image_path):
    pil_image = Image.open(image_path)
    im_copy = pil_image.copy()

    ###################
    # Make Thumbnail
    # define the size for PIL.thumbnail. Make sure the the smallest side is 256, the 512 will be ignored
    # and scaled to the ratio of the original image with PIL.thumbnail
    ###################
    if im_copy.size[0] > im_copy.size[1]:
        size = 512, 256
    else:
        size = 256, 512

    # creates thumbnail in place
    im_copy.thumbnail(size)

    ###################
    # Crop Image
    ###################
    # define cropping start and endpoints relative to dimensions of re-sized thumbnail
    start_x_crop = round(((im_copy.size[0] - 224) / 2)) + 1
    start_y_crop = round(((im_copy.size[1] - 224) / 2)) + 1
    end_x_crop = start_x_crop + 224
    end_y_crop = start_y_crop + 224
    area = (start_x_crop, start_y_crop, end_x_crop, end_y_crop)
    # return cropped image
    cropped = im_copy.crop(area)

    ####################
    # Turn image into an nd.array
    ####################
    imageArray = np.array(cropped)

    ###################
    # Normalize RGB
    ###################
    # static definition of Standard Deviation and Mean
    # in future think of way to dynamically determine these
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])

    ###################################
    # Convert RGB Channels into floats
    ###################################
    imageArray = imageArray / 256
    # print('imageArray', imageArray)
    ###################################
    # Subtract mean, than divide my STD
    ###################################
    imageArray = (imageArray - mean) / std
    # print('imageArray after normalization: ', imageArray)
    ######################################
    # Transpose so 3rd dimension in front
    #####################################
    transposed = imageArray.T
    # torch_img = transposed
    ######################################
    # Convert to torch image
    #####################################
    torch_img = torch.from_numpy(transposed).float()
    # print(torch_img)
    return torch_img

def main(argv):
    process_image(args.image_path)

if __name__ == '__main__':
    import sys
    main()
