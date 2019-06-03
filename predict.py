import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from process_image import process_image
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# Pass in a single image /path/to/image and return flower name
# and class probability
# python predict.py /path/to/image checkpoint

####################
# Examples of usage
####################
#
# Return top K most likely classes
# python predict.py /path/to/image checkpoint --top_k 3
#
# Use a mapping of categories to real names
# python predict.py /path/to/image checkpoint --category_names cat_to_name.json
#
# Use GPU for inference:
# python predict.py /path/to/image checkpoint --gpu
def load_model_checkpoint(checkpoint):
    # TODO: Write a function that loads a checkpoint and rebuilds the model
    from torchvision import datasets, transforms, models
    def rebuild(checkpoint):
        #model = models.vgg11(pretrained=True)
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

    loadedModel = rebuild('checkpoint2.pth')
    return loadedModel


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, (ax, ax2) = plt.subplots(2)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    numpy_image = image.numpy()
    image = numpy_image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.show()
    return fig, (ax, ax2)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img.unsqueeze_(0)
    logps = model(img)
    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(topk, dim=1)
    #print(top_ps[0].tolist(), top_class.shape)
    return top_ps[0].tolist(), top_class[0].tolist()
    # TODO: Implement the code to predict the class from an image file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)


        # # instantiate the stacked plots
        #fig, (ax1, ax2) = plt.subplots(2)

        #path = 'flowers/test/1/image_06743.flowers'
        path = 'flowers/test/1/image_06764.flowers'
        loadedmodel = load_model_checkpoint(args.checkpoint)
        top5_probs, top5_class_names = predict(path, loadedModel,5)
        #print(top5_probs)
        #print(top5_class_names)

        flower_tensor_image = process_image(path)
        # = torch.from_numpy(flower_np_image).type(torch.cuda.FloatTensor)
        #flower_tensor_image = flower_tensor_image.unsqueeze_(0)

        # make the first plot the image ax from imshow
        fig, (axs, ax2) = imshow(flower_tensor_image)
        axs.axis('off')
        index = str(top5_class_names[0])
        #print(index)
        #print(cat_to_name[index].upper())
        axs.set_title(cat_to_name[index].upper())
        list_of_names =[]
        for i in range(len(top5_class_names)):
            list_of_names.append(cat_to_name[str(top5_class_names[i])])

            #
            #plt.figure(figsize=(1,1))
            y_pos = np.arange(len(top5_class_names))
            ax2.barh(y_pos, list(reversed(top5_probs)))
            plt.yticks(y_pos, list(reversed(list_of_names)))
            print(type(ax2))
            print(type(axs))
            ax2.axis('auto')
            #plt.ylabel('Flower Type')
            #plt.xlabel('Class Probability')
            plt.show()
#predict('flowers/test/1/image_06743.flowers', model , topk=5)

# TODO: Display an image along with the top 5 classes


####################
# Examples of usage
####################
#
# Return top K most likely classes
# python predict.py /path/to/image checkpoint --top_k 3
#
# Use a mapping of categories to real names
# python predict.py /path/to/image checkpoint --category_names cat_to_name.json
#
# Use GPU for inference:
# python predict.py /path/to/image checkpoint --gpu

# ex python predict.py flowers/test/1/image_06743.jpg checkpoint.pth
def main():
    parser = argparse.ArgumentParser(description="Predicts image based on loaded model")
    parser.add_argument("image_path", help="path to image")
    parser.add_argument("checkpoint", help="path to checkpoint")
    parser.add_argument("--category_names", help="file that contains category name map")
    parser.add_argument("--gpu", action='store_true', help="Turn on Cuda Usage")
    parser.add_argument("--topk", type=int , help="Choose how many predictions")
    args = parser.parse_args()
    tensor_image = process_image(args.image_path)
    imshow(tensor_image)
    import json

    

if __name__ == "__main__":
    main()