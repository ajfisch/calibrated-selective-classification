# CIFAR-10

Download CIFAR-10-C:
```
mkdir -p ./data/processed/cifar
curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar -C data/processed/cifar/
```

Train a model on CIFAR-10 using
```
python bin/cifar/train_base_model.py
```
Alternatively, download our pre-trained models using the [download script](../download_models.sh).

Make predictions and generate last-layer features for all CIFAR-10 image splits (including CIFAR-10-C).
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

Train the selector:
```
python bin/tools/train_selective_model.py \
  --cal-dataset data/processed/cifar/pcal.meta.pt \
  --val-dataset data/processed/cifar/pval.meta.pt
```

Evaluate the selector in terms of AUC, or at a desired coverage level (coverage > 0):
```
python bin/tools/evaluate_selective_model.py \
  --datasets data/processed/cifar/test*.meta.pt \
  --coverage -1
```
