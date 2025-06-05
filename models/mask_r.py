from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torchvision
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.ops import masks_to_boxes

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from models.base_model import BaseModel

class CustomMaskRCNN(BaseModel):
    def __init__(self, backbone='resnet50', hidden_layer=256, finetune=False, optimizer_name = 'sgd', save_path='mask_rcnn_model.pt'):
        super().__init__(save_path=save_path)
        self.model = self.get_model(backbone, hidden_layer, finetune)
        self.optimizer = self.get_optimizer(optimizer_name)
        print('optimizer', self.optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    def __call__(self, images):
        if isinstance(images[0], Image.Image):
            transform = torchvision.transforms.ToTensor()
            images = [transform(image).to(self.device) for image in images]

        return self.model(images)

    def predict(self, images):
        if isinstance(images[0], Image.Image):
            transform = torchvision.transforms.ToTensor()
            images = [transform(image).to(self.device) for image in images]

        self.model.eval()
        with torch.no_grad():
            return self.model(images)


    def compute_iou(self, pred_masks, true_masks):
        """
        Compute IoU between a single predicted and ground truth mask.

        Args:
            pred_masks: [1, H, W] binary tensor.
            true_masks: [1, H, W] binary tensor.

        Returns:
            List with a single IoU score.
        """
        if pred_masks.shape[0] > 1:
            print('pred_masks.shape', pred_masks.shape)
            raise ValueError("pred_masks should be a single mask tensor, not a batch.")

        if true_masks.shape[0] > 1:
            print('true_masks.shape', true_masks.shape)
            raise ValueError("true_masks should be a single mask tensor, not a batch.")
        
        ious = []
        for true_mask in true_masks:
            best_iou = 0.0
            for pred_mask in pred_masks:
                intersection = (true_mask & pred_mask).float().sum()
                union = (true_mask | pred_mask).float().sum()
                if union == 0:
                    iou = 1.0 if intersection == 0 else 0.0
                    
                if union > 0:
                    iou = (intersection / union).item()
                    best_iou = max(best_iou, iou)
            ious.append(best_iou)
        return ious


    def evaluate(self, dataloader, iou_threshold=0.5):
        self.model.eval()
        total_ious = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)

                for output, target in zip(outputs, targets):
                    if 'masks' not in output or output['masks'].shape[0] == 0:
                        total_ious.append(0.0)
                        continue

                    # Filtrar solo máscaras de la clase 'person' (1 para COCO en torchvision)
                    labels = output['labels'].cpu()
                    person_indices = (labels == 1).nonzero(as_tuple=True)[0]  # 1 = 'person' en torchvision Mask R-CNN

                    if len(person_indices) == 0 and target['labels'].numel() != 0:
                        total_ious.append(0.0)
                        continue

                    # Filtrar y umbralizar máscaras predichas
                    pred_masks = output['masks'][person_indices, 0] > iou_threshold  # [P, H, W]
                    pred_masks = pred_masks.to(torch.bool).cpu()

                    # Combinar en una sola máscara binaria
                    pred_mask_combined = torch.any(pred_masks, dim=0, keepdim=True)  # [1, H, W]

                    # GT masks ya deberían ser solo de personas
                    true_masks = target['masks'].to(torch.bool).cpu()
                    true_mask_combined = torch.any(true_masks, dim=0, keepdim=True)  # [1, H, W]

                    # Redimensionar si tamaños no coinciden
                    # if pred_mask_combined.shape[-2:] != true_mask_combined.shape[-2:]:
                    #     pred_mask_combined = F.interpolate(
                    #         pred_mask_combined.unsqueeze(0).float(),
                    #         size=true_mask_combined.shape[-2:],
                    #         mode='nearest'
                    #     ).squeeze(0).to(torch.bool)

                    # Calcular IoU con máscara combinada
                    ious = self.compute_iou(pred_mask_combined, true_mask_combined)
                    total_ious.extend(ious)

        mean_iou = np.mean(total_ious) if total_ious else 0.0
        print(f"Mean IoU (combined person masks): {mean_iou:.4f}")
        return mean_iou

    
    def get_model(self, backbone='resnet50', hidden_layer=256, finetune=False):
        num_classes = 2
        model = None

        if backbone == 'resnet50':
            model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

        # elif backbone == 'resnet101':
        #     backbone_model = resnet_fpn_backbone('resnet101', pretrained=True)
        #     # model = MaskRCNN(backbone_model, num_classes=num_classes)
        #     model = MaskRCNN(backbone_model, num_classes=num_classes)
        else:
            raise ValueError(f"Backbone '{backbone}' no soportado")

        if finetune:
            # Personaliza la cabeza de clasificación
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # Personaliza el predictor de máscaras
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                            hidden_layer,
                                                            num_classes)

        model.to(self.device)
        return model

