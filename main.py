import yaml
import torch
import shutil
import numpy as np
import os
from PIL import Image

from models.mask_r import MaskRCNN
from dataset import get_dataloaders
from plots import plot_img_pred_target


TEST_FOLDER = "dataset_person/test"
ZIPFILE = 'G13_E02'


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_model(config):
    if config['model'] == 'mask_r':
        return MaskRCNN(config['hidden_layer'], config['pretrained'])
    else:
        raise ValueError(f"Model {config['model']} is not supported.")


def save_predictions(model):

    for f in open(os.path.join(TEST_FOLDER,'test_names.txt'),'r'):

        b = f.strip()

        # read test image
        image = Image.open(os.path.join(TEST_FOLDER,'images', b)+'.jpg')


        model.model.eval()
        with torch.no_grad():
            pred = model([image])  # image should already be a PIL Image, and model converts it

        # Access the first prediction
        if 'masks' in pred[0] and len(pred[0]['masks']) > 0:
            # Get the first mask, squeeze channel dimension, convert to [H, W]
            mask_tensor = pred[0]['masks'][0, 0]

            # Move to CPU, scale to 0–255, convert to uint8 numpy array
            mask_np = mask_tensor.mul(255).byte().cpu().numpy()

            # Convert to image and save
            pred_image = Image.fromarray(mask_np)
            predictions_dir = os.path.join(TEST_FOLDER, 'predictions')
            os.makedirs(predictions_dir, exist_ok=True)
            pred_image.save(os.path.join(predictions_dir, b + '.png'))
        else:
            print(f"No masks found for image {b}")



def create_zip():
    shutil.make_archive(ZIPFILE, 'zip', os.path.join(TEST_FOLDER,'predictions'))



def main():
    config = read_yaml('config.yaml')

    # Initialize model
    model = get_model(config)

    
    DATASET_DIR = 'dataset_person'
    dataloader, dataloader_validation = get_dataloaders(DATASET_DIR)
    dataloader_test = get_dataloaders(DATASET_DIR, test_only=True)

    #model.evaluate(data_loader_test=data_loader_test)

    save_predictions(model)

    return

    model.model.eval()
    with torch.no_grad():
        images, mask = next(iter(dataloader))
    
        print('image', len(images))
        print('mask', mask)
        outs = model(images.to(model.device))

        plot_img_pred_target(img, outs, mask)


if __name__ == "__main__":
    main()
    