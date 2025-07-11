# HA-DETR
### **HA-DETR: Hybrid Transformer and Convolutional Attention for Real-time Object Detection**

*Shen Wang\*, Ssww Ll\*, Yyjjjj Yyyy, Yyyhh Wwww (\*Equal Contribution)*

## Updates
* **2025/07/11**: Pre-trained models and codes of HA-DETR are released in [Pytorch](https://github.com/xinghaochen/DECO)

## Overview

## Main Results

Here we provide the pretrained `DECO` weights.

| Detector | Backbone | Epochs | AP (%) | Download                                                                                          |
| -------- | -------- | ------ | ------ | ------------------------------------------------------------------------------------------------- |
| HA-DETR  | R-50     | 60     | 53.5   | [hadetr_r50_60e.pth](https://github.com/xinghaochen/DECO/releases/download/1.0/deco_r50_150e.pth) |


## HA-DETR

### Installation
```shell
conda create -n ha_detr python=3.11.9
conda activate ha_detr
pip install -r requirements.txt
```

### Training
```shell
python train.py -c configs/hybrid/hybrid_${model}_coco.yml --seed=0
```
By default, we use 4 GPUs with total batch size as 24 for training HA-DETR with ResNet-50 backbone.
```shell
torchrun --nproc_per_node=4 train.py -c configs/hybrid/hybrid_${model}.yml --seed=0
```

### Evaluation
Model evaluation can be done as follows:
```shell
python train.py -c configs/hybrid/hybrid_${model}_coco.yml --test-only -r model.pth
```
```shell
torchrun --nproc_per_node=4 train.py -c configs/hybrid/hybrid_${model}_coco.yml --test-only -r model.pth
```

```bash
Results of HA-DETR with ResNet-50 backbone:
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.535
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.707
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.581
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.579
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.704
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.393
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.663
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.737
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.580
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.778
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.898
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.925
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.801
```


## 6. Acknowledgement
Our work is built upon [RT-DETR](https://github.com/lyuwenyu/RT-DETR) and [DECO](https://github.com/xinghaochen/DECO).

✨ Feel free to contribute and reach out if you have any questions! ✨