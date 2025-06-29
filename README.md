# ADT-Net: Adaptive Transformation-Driven Text-BasedPerson Search Network for Enhancing Cross-Modal Retrieval Robustness

## Abstract

Text-based person search aims to retrieve person images matching a given textual description. The challenge lies in mapping images and textual descriptions into a unified semantic space. This paper introduces ADT-Net, a novel framework designed to address the issue of excessive intra-class variance and insufficient inter-class variance caused by lighting variations. ADT-Net comprises two key modules: Invariant Representation Learning (IRL), which employs style transfer strategies and multi-scale alignment techniques to learn visually invariant features, and Dynamic Matching Alignment (DMA), which introduces nonlinear transformations and learnable dynamic temperature parameters to optimize the prediction distribution. Experimental results on multiple benchmark datasets demonstrate that ADT-Net outperforms current mainstream baseline methods, achieving superior retrieval accuracy and generalization ability. Here, we show that our proposed method significantly enhances the robustness of cross-modal person retrieval, particularly under varying lighting conditions and shooting angles.

![](/img/framework.png "Magic Gardens")

## Usage

### Requirements
* torch: 1.3.1
* torchvision: 0.14.1
* transformers: 4.46.3

### Prepare Datasets
1. Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset).

2. Organize them in your dataset root dir folder as follows:
```bash
|-- data/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|           |-- cam_a
|           |-- cam_b
|           |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|           |-- test
|           |-- train
|       |-- ICFG-PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```


3. Installation Environment

```bash
conda create -n adtnet python=3.8

conda activate adtnet

pip install torch==1.3.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117

pip install -r requirements.txt
```

### Training
```bash
python train.py
```
