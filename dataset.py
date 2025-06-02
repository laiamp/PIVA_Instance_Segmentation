import os
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
from torchvision import utils as tutils
from PIL import Image
import skimage.transform as sktf
import skimage.io as skio
import cv2
import random
import plotly.express as px

import utils
import transforms as T

from plots import draw_segmentation_map


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
  

COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COLORS = np.random.uniform(0, 255, size=(len(COCO_NAMES), 3)).astype(int)



class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    
class PersonDataset(torch.utils.data.Dataset):
  # TODO: fix assumptions about labels
    def __init__(self, root, transforms=None, has_masks=True):
        self.root = root
        self.transforms = transforms
        self.has_masks = has_masks
        if has_masks:
          self.imgs = list(sorted(os.listdir(os.path.join(root, "data/images"))))
          self.masks = list(sorted(os.listdir(os.path.join(root, "data/masks"))))
        else:
          self.imgs = list(sorted(os.listdir(os.path.join(root, "test/images"))))
          self.masks = None

    def __getitem__(self, idx):
        if self.has_masks:
          img_path = os.path.join(self.root, "data/images", self.imgs[idx])
        else:
          img_path = os.path.join(self.root, "test/images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if self.has_masks:
            mask_path = os.path.join(self.root, "data/masks", self.masks[idx])
            mask = Image.open(mask_path)
            mask = np.array(mask)
            obj_ids = np.array([0, 1]) # només té classes 0 i 1 (background i person)
            
            # Visualize the mask using plotly express
            # fig = px.imshow(mask, color_continuous_scale='gray', title="Mask")
            # fig.show()
            obj_ids = obj_ids[1:]

            masks = mask == obj_ids[:, None, None]
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": image_id,
                "area": area,
            }

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        else:
            if self.transforms is not None:
                img = self.transforms(img)

            return img

    def __len__(self):
        return len(self.imgs)


def get_dataloaders(dataset_dir, test_only=False):
    """dataset_dir: 'PennFudanPedor' or 'dataset_person' """
    # use our dataset and defined transformations
    if dataset_dir == 'PennFudanPed':
        # PennFudanPed dataset
        dataset = PennFudanDataset(dataset_dir, get_transform(train=True))
        dataset_test = PennFudanDataset(dataset_dir, get_transform(train=False))
    elif dataset_dir == 'dataset_person':
        # Person dataset
        dataset = PersonDataset(dataset_dir, get_transform(train=True))
        dataset_test = PersonDataset(dataset_dir, get_transform(train=False))
    else:
        raise ValueError("Unsupported dataset name. Use 'PennFudanPed' or 'person'.")


    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])


    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    if test_only:
        return data_loader_test
    
    return data_loader, data_loader_test




if __name__ == "__main__":
    dataset = PennFudanDataset('PennFudanPed/',get_transform(train=False))
    (img,target) = dataset[11]

    # Let's see the shape of the tensors
    print(img.shape)
    print(target["masks"].shape)
    print(target["labels"].shape)
    print(target["boxes"].shape)

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Input", "Ground Truth"))
    fig.add_trace(go.Image(z=img.numpy().transpose(1,2,0)*255), 1, 1)
    fig.add_trace(go.Image(z=draw_segmentation_map(img, target)*255), 1, 2)
    fig.show()
    
