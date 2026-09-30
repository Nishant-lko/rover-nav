# Rover navigation — YOLO26n fine-tuning

Stage 1 obstacle detection fine-tune for the autonomous rover. This repo keeps the training code, the base `yolo26n` weights, the fine-tuned checkpoint, and the run metrics. Image datasets and other pretrained weights are not included.

## Layout

- `training/train.py` — 85/15 split and YOLO26n fine-tune (AdamW, cosine LR, 100 epochs, patience 20)
- `training/preprocess.py` — YOLO label and box checks
- `training/dataset/data.yaml` — 12 remapped obstacle classes
- `models/pretrained/yolo26n.pt` — base weights
- `models/fine_tuned/best.pt` — fine-tuned checkpoint
- `training/runs/yolo26n_finetuned/` — args, `results.csv`, and metric plots

## Run

```bash
pip install -r requirements.txt
python3 training/train.py
```

`train.py` expects a compiled dataset at `annotation/compiled_dataset` (images + YOLO labels) and writes the train/val split under `training/dataset`.

Last logged epoch: precision 0.74, recall 0.24, mAP50 0.78, mAP50-95 0.27.
