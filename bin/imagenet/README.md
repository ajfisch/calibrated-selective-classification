# ImageNet

## Pre-training
For ImageNet, we use a pre-trained ResNet for our base $f(X)$.

## Data preparation
Download ImageNet-C:
 ```
mkdir -p ./data/datasets/imagenet/imagenet-c
curl -O https://zenodo.org/record/2235448/files/blur.tar
curl -O https://zenodo.org/record/2235448/files/digital.tar
curl -O https://zenodo.org/record/2235448/files/noise.tar
curl -O https://zenodo.org/record/2235448/files/weather.tar
tar -xvf blur.tar -C data/datasets/imagenet/imagenet-c
tar -xvf digital.tar -C data/datasets/imagenet/imagenet-c
tar -xvf noise.tar -C data/datasets/imagenet/imagenet-c
tar -xvf weather.tar -C data/datasets/imagenet/imagenet-c
```
These files are separated by corruption severity---for out experiments we combined them all. To collapse the directory structure, run:
```
python bin/imagenet/collapse_imagenet_c.py --input-dir data/datasets/imagenet/imagenet-c --output-dir data/datasets/imagenet/imagenet-c-collapsed
```

You will also need to download the `train` and `val` splits for ImageNet, and put them in `data/datasets/imagenet/train` and `data/datasets/imagenet/val`, respectively.

Make predictions and generate last-layer features (projected with SVD) for chosen ImageNet image splits (including ImageNet-C):
```
python bin/imagenet/process_splits.py --temperature-scale --apply-svd
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
  --val-dataset data/processed/imagenet/pval.meta.pt \
  --model-dir data/models/imagenet/selector
```

Evaluate the selector in terms of AUC, or at a desired coverage level (coverage > 0):
```
python bin/tools/evaluate_selective_model.py \
  --datasets data/processed/imagenet/test*.meta.pt \
  --model-files data/models/imagenet/selector/model.pt \
  --coverage -1
```
