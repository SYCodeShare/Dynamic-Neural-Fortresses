# Dynamic_Neural_Fortresses

The code for "Dynamic Neural Fortresses: An Adaptive Shield for Model Extraction Defense"
This repository contains the code accompanying the ICLR 2025 paper "Dynamic Neural Fortresses: An Adaptive Shield for Model Extraction Defense" Paper [link](https://openreview.net/pdf?id=029hDSVoXK): 

I am organizing the code. :)
#### Requirements to run the code:
---

1. Python 3.11
2. PyTorch 2.2.1
3. numpy 1.24.3
4. torchvision 0.17.1




#### Experiments on Resnet architecture (data-free):
---
Usage for training and testing Resnet network with the proposed method on dataset CIFAR100

Download the model and place it in the corresponding folder.

```python

cd data-free/

python dfme_cifar100.py
```
#### Experiments on Resnet architecture (data-based):
---
Usage for training and testing Resnet network with the proposed method on dataset CUB200.

Download the model and place it in the corresponding folder.

```python

cd data-based/Resnet/

python scripts/run_cub200.py
```

#### Experiments on ViT architecture (data-based):
---
Usage for training and testing ViT with the proposed method on dataset TinyImageNet200.

Download the model and place it in the corresponding folder. Edit the **path** in defenses/victim/eval_VIT.py

```python

cd data-based/ViT/

python scripts/run_tinyimagenet_VIT.py

```




### Reference
---

```
@inproceedings{
luan2025dynamic,
title={Dynamic Neural Fortresses: An Adaptive Shield for Model Extraction Defense},
author={Siyu Luan and Zhenyi Wang and Li Shen and Zonghua Gu and Chao Wu and Dacheng Tao},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=029hDSVoXK}
}
```

### Acknowledgment
---

Some codes are from [datafree-model-extraction](https://github.com/cake-lab/datafree-model-extraction.git) and [ModelGuard](https://github.com/Yoruko-Tang/ModelGuard.git) Thanks.
