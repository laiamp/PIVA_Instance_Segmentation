import numpy as np
import torch
import segmentation_models_pytorch as smp

from models.base_model import BaseModel 

class CustomUnet(BaseModel):
    def __init__(self, encoder_name='resnet34', lr=1e-3):
        super().__init__(save_path='unet_model.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        ).to(self.device)
        self.loss_fn = smp.losses.DiceLoss(mode='binary')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for images, masks in dataloader:
            # images y masks son listas con 1 elemento cada una
            image = images[0].to(self.device).unsqueeze(0)  # (1,3,H,W)
            mask = masks[0]['masks'].to(self.device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.loss_fn(outputs, mask)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, dataloader, num_epochs=5):
        for epoch in range(num_epochs):
            avg_loss = self.train_one_epoch(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            self.save_model(f'unet_model_epoch_{epoch+1}.pth')
        

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            img = image.unsqueeze(0).to(self.device)
            output = self.model(img)
            mask = torch.sigmoid(output)[0, 0]
            return mask.cpu()

    def evaluate(self, dataloader, threshold=0.5):
        self.model.eval()
        ious = []
        with torch.no_grad():
            for images, masks in dataloader:
                image = images[0].to(self.device).unsqueeze(0)  # (1,3,H,W)
                mask = masks[0]['masks'].to(self.device).unsqueeze(0)  # (1,1,H,W)

                outputs = self.model(image)
                print(f"outputs.shape: {outputs.shape}, mask.shape: {mask.shape}")
                # pred = torch.sigmoid(outputs) > threshold
                pred = outputs > threshold

                pred = pred.bool().squeeze(0)
                true = mask.bool().squeeze(0)

                intersection = (pred & true).sum().item()
                union = (pred | true).sum().item()
                iou = intersection / union if union != 0 else 1.0
                ious.append(iou)

        mean_iou = np.mean(ious) if ious else 0.0
        print(f"Mean IoU: {mean_iou:.4f}")
        return mean_iou
