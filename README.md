# CRCSD
A novel Movie Scene Detection method based on Clue Relationship and Constrained Shot Description
## Environment
The project requires about 220G storage and 30G memory to run  

  -python 3.9.18  

  -PyTorch 2.0.1  

  -torchvision 0.15.2  
  
## Prepare Dataset
1.Extracting visual features.  
Backbone is ResNet-50 Pretrained on ImageNet.  
Extract board features.  
Using Gemma-3-4B to build a constrained shot description, we will provide a Shotboard.  
2.Download MovieNet dataset label: https://drive.google.com/drive/folders/1F-uqCKnhtSdQKcDUiL3dRcLOrAxHargz  
3.Generate dataset using CRCSD/dataloader/supervised_get_movienet.py

## Train and Test
run CRCSD/main.py
