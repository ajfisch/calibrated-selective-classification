# ImageNet

2. Download ImageNet-C:
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
