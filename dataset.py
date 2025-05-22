import os
import numpy as np

import torch
import torch.utils.data

from torchvision import transforms
from torchvision import utils as tutils


import skimage.transform as sktf
import skimage.io as skio

from PIL import Image



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
import cv2
import random

def draw_segmentation_map(image, target, score_thres=0.8):

    # Convert back to numpy arrays
    _image = np.copy(image.cpu().detach().numpy().transpose(1,2,0)*255)
    _masks = np.copy(target['masks'].cpu().detach().numpy().astype(np.float32))
    _boxes = np.copy(target['boxes'].cpu().detach().numpy().astype(int))
    _labels = np.copy(target['labels'].cpu().detach().numpy().astype(int))
    if "scores" in target:
      _scores = np.copy(target["scores"].cpu().detach().numpy())
    else:
      _scores = np.ones(len(_masks),dtype=np.float32)

    alpha = 0.3

    label_names = [COCO_NAMES[i] for i in _labels]

    # Add mask if _scores
    m = np.zeros_like(_masks[0].squeeze())
    for i in range(len(_masks)):
      if _scores[i] > score_thres:
        m = m + _masks[i]

    # Make sure m is the right shape
    m = m.squeeze()

    # dark pixel outside masks
    _image[m<0.5] = 0.3*_image[m<0.5]

    # convert from RGB to OpenCV BGR and back (cv2.rectangle is just too picky)
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

    for i in range(len(_masks)):
      if _scores[i] > score_thres:
        # apply a randon color to each object
        color = COLORS[random.randrange(0, len(COLORS))].tolist()

        # draw the bounding boxes around the objects
        cv2.rectangle(_image, _boxes[i][0:2], _boxes[i][2:4], color=color, thickness=2)
        # put the label text above the objects
        cv2.putText(_image , label_names[i], (_boxes[i][0], _boxes[i][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=1, lineType=cv2.LINE_AA)

    return _image/255


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


if __name__ == "__main__":
    dataset = PennFudanDataset('PennFudanPed/',get_transform(train=False))
    (img,target) = dataset[11]

    # Let's see the shape of the tensors
    print(img.shape)
    print(target["masks"].shape)
    print(target["labels"].shape)
    print(target["boxes"].shape)
