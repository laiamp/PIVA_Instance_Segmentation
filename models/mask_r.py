
import torchvision
import torch

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT', progress=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}.")
model.to(device).eval()


# add a batch dimension
imgs = img.unsqueeze(0).to(device) #torch.stack((img,img))
outs = model(imgs.to(device))


print(f"Number of predictions = {len(outs[0]['labels'])}")
print(f"  labels = {outs[0]['labels'].cpu().numpy()}")
print(f"  scores = {outs[0]['scores'].detach().cpu().numpy()}")