# Calibrated Selective Classification

This code accompanies the TMLR paper [Calibrated Selective Classification](https://openreview.net/forum?id=zFhNBs8GaV).

## Summary

Selective classification allows models to abstain from making predictions (e.g., say "I don't know") when in doubt in order to obtain better effective accuracy. While typical selective models can succeed at producing more accurate predictions on average, they may still allow for wrong predictions that have high confidence, or skip correct predictions that have low confidence. Providing *calibrated* uncertainty estimates alongside predictions---probabilities that correspond to true frequencies---can be as important as having predictions that are simply accurate on average. Uncertainty estimates, however, can sometimes be unreliable. This repository implements a new approach to calibrated selective classification (CSC), where a selector is learned to reject examples with "uncertain" uncertainties. The goal is to make predictions with well-calibrated uncertainty estimates over the distribution of accepted examples, a property called selective calibration.

## Setup

We provide instructions to install CSC within a [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment running Python 3.8. 

```
conda create -n csc python=3.8 && conda activate csc
```

Install PyTorch following the instructions from pytorch.org.

Install all other requirements via `pip install -r requirements.txt`.

All commands should now be run from the root directory of this repository.


## Pre-trained models

For CIFAR-10, we trained our base model `f(X)` using 
```
python bin/cifar/train_base_model.py
```
This is based on the training in [AugMix](https://github.com/google-research/augmix). Augmentations can be turned on/off using the `--no-aug` flag.

For ImageNet, our base model `f(X)` is automatically downloaded from the pre-trained model zoo. Our lung cancer risk prediction model was trained using [Sybil](https://github.com/reginabarzilaygroup/Sybil).


All pre-trained `f(X)` and `g(X)` models used in our experiments (for CIFAR-10, ImageNet, and Lung) can be downloaded by running:
```
./download_models.sh
```

## Training a selective model

