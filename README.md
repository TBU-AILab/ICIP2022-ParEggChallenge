# ICIP2022-ParEggChallenge
ICIP2020 - Parasitic Egg Detection and Classification in Microscopic Images

This repository contains codes and configs needed to reproduce results handed in by our team in [ICIP2020 Challenge](https://2022.ieeeicip.org/) - [Parasitic Egg Detection and Classification in Microscopic Images](https://icip2022challenge.piclab.ai/). The leaderboard of the challenge is available [here](https://icip2022challenge.piclab.ai/leaderboard/).

Our final solution achieves a Mean Intersection-over-Union (mIoU) of 0.885721 and a Mean F1-score (mF1scr) of 0.964605. Please refer to the original paper _'will be added'_ for more information about our solution. The following sections walk you through each step in a general way. Keep in mind that you need to accommodate the relative paths to your file system.

## Installation
The project was implemented using [MMDetection](https://github.com/open-mmlab/mmdetection) and [SAHI](https://github.com/obss/sahi) projects. Please refer to their installation guides to set up your system first.
Other required packages are in requirements.txt.

## Data preparation
To cut images into patches, use script `data_preparation/n_fold_sahi_slice.py`. It creates a folder with sliced images (sliced) and an accompanying JSON file (cut_960x1280__coco.json).
Script `data_preparation/n_fold_coco_slit_trainval.py` then splits all available data (original full image size in labels.json and patches in cut_960x1280__coco.json) into 5-fold for cross-validation.
Script `data_preparation/coco_create_empty_test_json.py` creates empty JSON in COCO style needed for inferencing the test dataset.

## Training
We use the `train.py` script available in MMDetection tools. For training, a GPU must be available. The command is as follows:
```
python mmdetection/tools/train.py configs/TOOD_r50-parasiteEgg.py
```

## Inference
We use the `test.py` script available in MMDetection tools for full-size image inference. The command is as follows:
```
python mmdetection/tools/test.py models/TOOD_r50/fold-0/TOOD_r50-parasiteEgg.py  models/TOOD_r50/fold-0/epoch_24.pth
```
For slicing aided hyper inference, please use the 'inference/sahi_predict_n_fold.py' script. You need to adapt the paths accordingly.

## Results merging
First, we need to merge results from all five folds. Script `results_merging/coco_merge_results.py` does the job.
Second, the script `results_merging/coco_results_nms.py` removes duplicate detections (for more details about the process, please refer to the original paper).

## Submmision
Script `results2submision.py` serves for converting the final results into the required format for the challenge leaderboard. The script can also be applied to raw output from both inference variants.

