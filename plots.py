import plotly.express as px
import numpy as np


from plotly.subplots import make_subplots
import plotly.graph_objects as go

# fig = make_subplots(rows=1, cols=2, subplot_titles=("Input", "Ground Truth"))
# fig.add_trace(go.Image(z=img.numpy().transpose(1,2,0)*255), 1, 1)
# fig.add_trace(go.Image(z=draw_segmentation_map(img, target)*255), 1, 2)



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


def plot_prediction():
    (img,target) = dataset_test[6]
    imgs = img.unsqueeze(0).to(device) #torch.stack((img,img))
    outs = model2(imgs)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Prediction (all scores)", "Prediction (scores>0.8)"))
    fig.add_trace(go.Image(z=draw_segmentation_map(img, outs[0], score_thres=0.0)*255), 1, 1)
    fig.add_trace(go.Image(z=draw_segmentation_map(img, outs[0], score_thres=0.8)*255), 1, 2)



def plot_img_pred_target(img, pred, target):
  """
  Plots the input image, prediction, and target side by side.

  Args:
    img (Tensor or np.ndarray): Input image (C,H,W) or (H,W,C).
    pred (dict): Prediction dict (with 'masks', 'boxes', 'labels', 'scores').
    target (dict): Target dict (with 'masks', 'boxes', 'labels').
  """
  # Convert img to numpy if it's a tensor
  if hasattr(img, "cpu"):
    img_np = img.cpu().detach().numpy().transpose(1,2,0)
    img_np = np.clip(img_np, 0, 1)
  else:
    img_np = img

  pred_img = draw_segmentation_map(img, pred, score_thres=0.8)
  target_img = draw_segmentation_map(img, target, score_thres=0.8)

  # Stack images for px.imshow
  stacked = np.stack([img_np, pred_img, target_img])
  titles = ["Image", "Prediction", "Target"]

  fig = px.imshow(stacked, facet_col=0, facet_col_wrap=3, labels={"facet_col": "Type"})
  for i, t in enumerate(titles):
    fig.layout.annotations[i]['text'] = t
  fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
  fig.show()