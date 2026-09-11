# BoneScanAD

Anomaly detection for whole-body bone SPECT scans, built on the
[VisualAD](https://github.com/7HHHHH/VisualAD) formulation: a learnable
anomaly/normal token pair is inserted into the ViT patch sequence and scored by
`cos(patch, t_a) - cos(patch, t_n)`, with one token pair per anatomical region.
Nine regions (knees, ankles, shoulders, spine, chest) are scored across the
anterior and posterior projections.

## Install

```bash
pip install -r requirements.txt
```

A CUDA GPU with at least 24 GB of memory is recommended.

## Backbone

Download a Qwen3.5 vision-language model and pass it with `--backbone-path`
(a local directory or a HuggingFace model id). The default is `./backbone`.
Only the vision tower is used for scoring.

## Data

Whole-body images named by patient id, plus region labels in three CSV sources:

```
data/ant/<patient_id>.png        # anterior projection
data/post/<patient_id>.png       # posterior projection
dataset/limbs/{train,val,test}.csv    # patient_id, loc_0 ... loc_5
dataset/spine/{train,val,test}.csv    # patient_id, label_spine
dataset/chest/{train,val,test}.csv    # patient_id, loc_7, loc_8
```

`0` is normal, any positive value is abnormal, and a blank or negative value
marks the region as unlabelled (the sample is skipped). Use `--data-dir`,
`--label-dir` and `--label-subdirs` if your layout differs.

## Crop template

**No ROI coordinates ship with this repository** — they depend on your scan
protocol and camera, so you must measure them on your own data. Write a JSON
template mapping each projection and region id to `[x1, y1, x2, y2]`:

```json
{
  "ant": {
    "0": [140, 480, 190, 560],
    "7": {"box": [132, 135, 190, 254], "polygon": [[140,150],[188,150],[188,240]]}
  },
  "post": {
    "6": [102, 140, 150, 320]
  }
}
```

Region order: `0 left knee | 1 right knee | 2 left ankle | 3 right ankle |
4 left shoulder | 5 right shoulder | 6 spine | 7 left chest | 8 right chest`.
The spine is cropped from `post`, every other region from `ant`. An optional
`polygon` masks out neighbouring anatomy inside the box. The full format is
documented at the top of `utils/crop.py`.

`--crop-boxes` defaults to `<data-dir>/crop_boxes.json`. All nine regions must
be present; training checks this before loading the backbone and reports which
entries are missing.

## Train

```bash
python train.py \
    --data-dir data \
    --label-dir dataset \
    --crop-boxes data/crop_boxes.json \
    --backbone-path /path/to/Qwen3.5-VL \
    --output-dir outputs/bonescanad
```

Defaults: batch size 8, 50 epochs, AdamW, threshold selected on the validation
split at a specificity floor of 0.90. Run `python train.py --help` for the rest,
including the ablation switches (`--no-aaea`, `--no-acdf`, `--no-sare`,
`--no-sca`) and `--evaluate-test`.

Outputs go to `--output-dir`: `best.pt`, `run_config.json`, `train.log`,
prediction CSVs and metric JSONs.

## Acknowledgements

The backbone formulation and the SCA module follow
[VisualAD](https://github.com/7HHHHH/VisualAD). This is an independent
implementation for bone SPECT, not affiliated with the VisualAD authors.
