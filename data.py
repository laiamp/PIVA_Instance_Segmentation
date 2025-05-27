from PIL import Image

import utils
import transforms as T


def load_data():
    Image.open('PennFudanPed/PNGImages/FudanPed00012.png')

    mask = Image.open('PennFudanPed/PedMasks/FudanPed00012_mask.png').convert('P')
    # each mask instance has a different color, from zero to N, where
    # N is the number of instances. In order to make visualization easier,
    # let's adda color palette to the mask.
    mask.putpalette([
        0, 0, 0, # black background
        255, 0, 0, # index 1 is red
        255, 255, 0, # index 2 is yellow
        255, 153, 0, # index 3 is orange
    ])
    mask


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)