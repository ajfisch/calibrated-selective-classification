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
```

Train the selector:
```
```

Evaluate the selector:
```
```
