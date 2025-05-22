# PIVA_Instance_Segmentation


%%shell
# download the Penn-Fudan dataset
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
# extract it in the current folder
unzip -q PennFudanPed.zip
# This library is not needed (already installed)
#pip install opencv-contrib-python==4.11.0.86

## Models
- Mask R CNN fine-tuned last layer
- YOLO