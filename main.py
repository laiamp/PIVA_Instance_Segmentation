import yaml
import torch
import shutil
import numpy as np
import os
from PIL import Image

from models.mask_r import CustomMaskRCNN
from models.yolo import CustomYOLO
from models.unet import CustomUnet
from dataset import get_dataloaders
from plots import plot_img_pred_target


TEST_FOLDER = "dataset_person/test"
ZIPFILE = 'G13_E02'


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_model(config):
    if config['model'] == 'mask_r':
        return CustomMaskRCNN(config['mask_rcnn_backbone'], config['hidden_layer'], config['finetune'], config['optimizer'], config['save_path'])
    elif config['model'] == 'yolo':
        return CustomYOLO(config['yolo_model_name'], config['finetune'])
    elif config['model'] == 'unet':
        return CustomUnet()
    else:
        raise ValueError(f"Model {config['model']} is not supported.")


def save_predictions(model):

    for f in open(os.path.join(TEST_FOLDER,'test_names.txt'),'r'):

        b = f.strip()

        # read test image
        image = Image.open(os.path.join(TEST_FOLDER,'images', b)+'.jpg')

        pred = model.predict([image])

        
        # Access the first prediction
        if 'masks' in pred[0] and len(pred[0]['masks']) > 0:
            # Get the first mask, squeeze channel dimension, convert to [H, W]
            mask_tensor = pred[0]['masks'][0, 0]

            mask_bin = (mask_tensor >= 0.5).to(torch.uint8)  # values will be 0 or 1

            # Move to CPU, scale to 0–255, convert to uint8 numpy array (optional)
            mask_np = mask_bin.mul(255).byte().cpu().numpy()

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

    if config['model'] == 'mask_r':
        config['save_path'] = f'{config['model']}_{config['mask_rcnn_backbone']}_{config['hidden_layer']}'
    elif config['model'] == 'yolo':
        config['save_path'] = f'{config['model']}_{config['yolo_model_name']}'

    # Initialize model
    model = get_model(config)

    if config['load_path'] != '':
        print('Loading model from', config['load_path'])
        model.model.load_state_dict(torch.load(config['load_path']))

    
    DATASET_DIR = 'dataset_person'
    dataloader, dataloader_validation = get_dataloaders(DATASET_DIR)
    dataloader_test = get_dataloaders(DATASET_DIR, test_only=True)

    if config['finetune'] and not config['predict']:
        if config['model'] == 'mask_r' or config['model'] == 'unet':
            model.train(num_epochs=config['epochs'], dataloader=dataloader)
        elif config['model'] == 'yolo':
            model.train(config['epochs'])

    # mIoU = model.evaluate(dataloader_validation)
    # print('config', config, 'mIoU', mIoU)

    save_predictions(model)
    create_zip()
    return

   

if __name__ == "__main__":
    main()
    