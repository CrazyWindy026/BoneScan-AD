# BoneScanAD

Nine-region anomaly detection on whole-body bone scintigraphy using a Qwen3.5 ViT and [VisualAD](https://github.com/7HHHHH/VisualAD)-style normal/anomaly tokens, with SCA,AAEA and SARE.

## Setup

```bash
pip install -r requirements.txt
```

Place the Qwen3.5 backbone in `./backbone`, or set `--backbone-path`.

## Data

```text
data/ant/<patient_id>.png
data/post/<patient_id>.png
data/crop_boxes.json
dataset/limbs/{train,val,test}.csv
dataset/spine/{train,val,test}.csv
dataset/chest/{train,val,test}.csv
```

`crop_boxes.json` defines the nine anatomical ROIs (see `utils/crop.py`). Labels: `0` normal, `>0` abnormal; missing/negative labels are ignored.

## Train

```bash
python train.py \
  --backbone-path ./backbone \
  --data-dir ./data \
  --label-dir ./dataset \
  --crop-boxes ./data/crop_boxes.json
```

## Acknowledgement

Built upon [VisualAD](https://github.com/7HHHHH/VisualAD); independent implementation for bone scintigraphy.
