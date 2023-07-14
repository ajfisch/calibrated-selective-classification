#! /bin/bash

OPTION=$1
OUTPUT_DIR="data-tmp"

if [[ $OPTION != "cifar" && $OPTION != "imagenet" ]]; then
    echo "ERROR: dataset must be either 'cifar' or 'imagenet'. Given '${OPTION}'."
    exit 1
fi

mkdir download
pushd download
wget "https://selective-calibration.s3.us-east-2.amazonaws.com/${OPTION}.tar.gz"
tar -zxvf "${OPTION}.tar.gz"
popd

mkdir -p ${OUTPUT_DIR}/processed/${OPTION}
mv download/${OPTION}/data/* ${OUTPUT_DIR}/processed/${OPTION}

if [[ $OPTION == "imagenet" ]]; then
    mkdir -p ${OUTPUT_DIR}/models/${OPTION}/selector
    mv download/${OPTION}/models/* ${OUTPUT_DIR}/models/${OPTION}/selector
else
    mkdir -p ${OUTPUT_DIR}/models/${OPTION}
    mv download/${OPTION}/models/* ${OUTPUT_DIR}/models/${OPTION}
fi

rm -r download

echo "Done."
