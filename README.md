# Feature Refinement from Multiple Perspectives for High Performance Salient Object Detection

PaperID-210 for PRCV 2023

## Code Environment
```bash
conda env create -f environment.yaml
```

## Data Preparation

You can refer to the work  'Salient Object Detection via Integrity Learning' [IEEE TPAMI 2022] https://github.com/mczhuge/ICON#get-start for the DUT-OMRON, DUTS-TE, ECSSD, HKU-IS and PASCAL-S datasets.

## Inference

First, unzip the compressed files in the folder 'ckpt' into the checkpoint file 'model_best.pth'

```bash
zip -s 0 ckpt/model_best.zip --out ckpt/ckpt.zip
unzip ckpt/ckpt.zip
```
Then, test the model

```python
python test.py -p /path/of/the/root/of/datasets -c model_best.pth
```

## Predicted Saliency Maps
The predicted saliency maps is in the folder 'saliency_maps'.

## Evaluation Code
We use the same [evaluation code](https://github.com/lartpang/PySODEvalToolkit) as the MSFNet [1] to evaluate the predicted saliency maps.

[1] Auto-MSFNet: Search Multi-scale Fusion Network for Salient Object Detection. ACMMM, 2021

We ensembles the aboved evaluation code into our code.


