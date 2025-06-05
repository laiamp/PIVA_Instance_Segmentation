import torch
import torchvision
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.ops import masks_to_boxes
from ultralytics import YOLO
import torch.nn.functional as F

from models.base_model import BaseModel




class CustomYOLO(BaseModel): 
    def __init__(self, model_name='yolov8n-seg.pt', finetune=False, save_path='yolo_model.pt'):
        super().__init__(save_path=save_path)
        self.model = self.get_model(model_name, finetune)
        self.optimizer = None  # YOLO models from ultralytics handle optimization internally
        self.lr_scheduler = None

    def __call__(self, images):
        if isinstance(images[0], Image.Image):
            images = [image for image in images]
        results = self.model(images)
        return results

    def predict(self, images):
        # This will return segmentation masks if using a segmentation model
        results = self.model.predict(images)
        # Extract masks from results
        masks = []
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                masks.append(result.masks.data.cpu())
            else:
                masks.append(None)
        return masks


    def get_model(self, model_name, finetune=False):
        # Use a segmentation model checkpoint, e.g., 'yolov8n-seg.pt'
        model = YOLO(model_name)
        model.to(self.device)
        return model



    def compute_iou(self, pred_masks, true_masks, threshold=0.5):
        """
        Compute mean IoU between predicted and true masks.

        Args:
            pred_masks: [1, H, W] binary tensor (after thresholding).
            true_masks: [1, H, W] binary tensor.
            threshold: score threshold to binarize predicted masks.

        Returns:
            iou_scores: list of IoU values for matched masks.
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
                processed_images = []
                for img in images:
                    if isinstance(img, torch.Tensor):
                        img = img.permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                    elif isinstance(img, Image.Image):
                        img = np.array(img)
                    elif isinstance(img, np.ndarray):
                        pass
                    else:
                        raise TypeError(f"Formato de imagen no soportado: {type(img)}")
                    processed_images.append(img)

                results = self.model(processed_images)

                for result, target in zip(results, targets):
                    if result.masks is None or result.masks.data.shape[0] == 0:
                        total_ious.append(0.0)
                        continue

                    # Filtrar solo máscaras de clase 'person' (clase 0 COCO)
                    if result.boxes is None or result.boxes.cls is None:
                        total_ious.append(0.0)
                        continue

                    cls = result.boxes.cls.cpu()
                    person_indices = (cls == 0).nonzero(as_tuple=True)[0]

                    if len(person_indices) == 0:
                        total_ious.append(0.0)
                        continue

                    # Obtener máscaras predichas de personas y umbralizar
                    pred_masks = result.masks.data[person_indices]
                    pred_masks = (pred_masks > iou_threshold).to(torch.bool).cpu()

                    # Combinar todas las máscaras de persona en una sola máscara binaria (canal único)
                    pred_mask_combined = torch.any(pred_masks, dim=0)  # [H, W]

                    # Máscara GT (suponiendo que target['masks'] ya contiene solo personas)
                    true_masks = target['masks'].to(torch.bool).cpu()
                    true_mask_combined = torch.any(true_masks, dim=0)  # [H, W]

                    # Redimensionar si es necesario
                    if pred_mask_combined.shape != true_mask_combined.shape:
                        pred_mask_combined = F.interpolate(
                            pred_mask_combined.unsqueeze(0).unsqueeze(0).float(),
                            size=true_mask_combined.shape,
                            mode='nearest'
                        ).squeeze(0).squeeze(0).to(torch.bool)

                    # Calcular IoU con las máscaras combinadas
                    ious = self.compute_iou(pred_mask_combined.unsqueeze(0), true_mask_combined.unsqueeze(0))
                    total_ious.extend(ious)

        mean_iou = np.mean(total_ious) if total_ious else 0.0
        print(f"Mean IoU (person mask combined): {mean_iou:.4f}")
        return mean_iou
            
    def train(self, num_epochs=5):
        self.model.train(data=data, epochs=num_epochs, device=self.device)
