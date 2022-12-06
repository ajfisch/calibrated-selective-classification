# CIFAR-10

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
