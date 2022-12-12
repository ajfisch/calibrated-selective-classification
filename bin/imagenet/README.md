# ImageNet

## Pre-training
For ImageNet, we use a pre-trained ResNet for our base $f(X)$.

## Data preparation
Download ImageNet-C:
 ```
mkdir -p ./data/processed/imagenet/imagenet-c
curl -O https://zenodo.org/record/2235448/files/blur.tar
curl -O https://zenodo.org/record/2235448/files/digital.tar
curl -O https://zenodo.org/record/2235448/files/noise.tar
curl -O https://zenodo.org/record/2235448/files/weather.tar
tar -xvf blur.tar -C data/processed/imagenet/imagenet-c
tar -xvf digital.tar -C data/processed/imagenet/imagenet-c
tar -xvf noise.tar -C data/processed/imagenet/imagenet-c
tar -xvf weather.tar -C data/processed/imagenet/imagenet-c
```

Make predictions and generate last-layer features for all CIFAR-10 image splits (including CIFAR-10-C):
```
python bin/imagenet/process_splits.py --temperature-scale
```

Derive meta features:
```
python bin/tools/generate_meta_features.py \
  --train-dataset data/processed/imagenet/train.pt \
  --cal-dataset data/processed/imagenet/pcal.pt \
  --val-dataset data/processed/imagenet/pval.pt \
  --test-datasets data/processed/imagenet/test_*.pt \
  --skip-class-features
```

## Training and evaluation
Train the selector:
```
python bin/tools/train_selective_model.py \
  --cal-dataset data/processed/imagenet/pcal.meta.pt \
  --val-dataset data/processed/imagenet/pval.meta.pt
```

Evaluate the selector in terms of AUC, or at a desired coverage level (coverage > 0):
```
python bin/tools/evaluate_selective_model.py \
  --datasets data/processed/imagenet/test*.meta.pt \
  --coverage -1
```
