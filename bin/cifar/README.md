# CIFAR-10
The following commands outline how to process data and train a selector model for CIFAR-10.

## Data and model preparation

> **Note**
> To skip all subsequent steps in this section, download our precomputed features by running
> ```
> ./bin/cifar/download_data.sh
> ```


### Create splits
To split CIFAR-10 into training, calibration, and validation sets (testing is done on CIFAR-10-C), run:
```
python bin/cifar/create_splits.py
````

### Pre-training
For CIFAR-10, we trained our base model $f(X)$ using 
```
python bin/cifar/train_base_model.py --no-aug
```
This is based on the training in [AugMix](https://github.com/google-research/augmix). Augmentations can be turned on by removing the `--no-aug` flag.

Alternatively, download our pre-trained models using the [download script](../download_models.sh).

### Preprocessing
Download CIFAR-10-C:
```
mkdir -p ./data/processed/cifar
curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar -C data/processed/cifar/
```

Make predictions and generate last-layer features for all CIFAR-10 image splits (including CIFAR-10-C):
```
python bin/cifar/process_splits.py --temperature-scale
```

Derive meta features:
```
python bin/tools/generate_meta_features.py \
  --train-dataset data/processed/cifar/train.pt \
  --cal-dataset data/processed/cifar/pcal.pt \
  --val-dataset data/processed/cifar/pval.pt \
  --test-datasets data/processed/cifar/test_*.pt
```

## Training and evaluation
Train the selector:
```
python bin/tools/train_selective_model.py \
  --cal-dataset data/processed/cifar/pcal.meta.pt \
  --val-dataset data/processed/cifar/pval.meta.pt \
  --model-dir data/models/cifar/selector
```

Evaluate the selector in terms of AUC, or at a desired coverage level (if coverage > 0):
```
python bin/tools/evaluate_selective_model.py \
  --model-files data/models/cifar/selector/model.pt \
  --datasets data/processed/cifar/test*.meta.pt \
  --coverage -1
```
