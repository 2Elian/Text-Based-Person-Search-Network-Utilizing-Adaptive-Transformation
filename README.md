# ADT-Net: Text-Based Person Search Network Utilizing Adaptive Transformation

This is the official PyTorch implementation of the paper ADT-Net: Text-Based Person Search Network Utilizing Adaptive Transformation. This repository supports training and evaluation on three text-based person search benchmarks: CUHK-PEDES, ICFG-PEDES and RSTPReid.

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

### Training
```bash
python train.py
```
