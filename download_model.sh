#!/bin/sh
set -e

model_name=72061bc # fist compatible commit @ 2021-10-25T20:03
model_dir=""
# model_name=P384 # model name
# model_dir="lpcnet_efficiency/"

model_archive="lpcnet_data-${model_name}.tar.gz"

# Data download
if [ ! -f $model_archive ]; then
        echo "Downloading latest model"
        wget "https://media.xiph.org/lpcnet/data/${model_dir}${model_archive}"
fi

# Data extraction
tar xvf $model_archive
# `src/nnet_data.c` & `src/nnet_data.h`
touch src/nnet_data.[ch]
