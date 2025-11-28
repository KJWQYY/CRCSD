# CRCSD
A novel Movie Scene Detection method based on Clue Relationship and Constrained Shot Description
## Environment
The project requires about 220G storage and 30G memory to run  

   -python 3.9.18  

   -PyTorch 2.0.1  

   -torchvision 0.15.2  
  
## Prepare Dataset
1. Extract visual features.Backbone is ResNet-50 Pretrained on ImageNet.
2. Extract board features.Using Gemma-3-4B to build a constrained shot description. Our generated shot description:  https://pan.baidu.com/s/12ODIDMGF0t3jhVMTEPoJuw Code: yyck
3. Download MovieNet dataset label: https://drive.google.com/drive/folders/1F-uqCKnhtSdQKcDUiL3dRcLOrAxHargz  
4. Generate dataset using CRCSD/dataloader/supervised_get_movienet.py
## Model
https://pan.baidu.com/s/1aaYTDD7I6lbIWCovUGt92Q Code: ed33
## Graph
https://pan.baidu.com/s/1KrInxlAyeLalKwsgx9pnOQ Code: n6q4
## Train and Test
run CRCSD/main.py
