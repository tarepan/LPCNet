#!/bin/sh
set -e

# `$1`: commit hash value
model=lpcnet_data-$1.tar.gz

if [ ! -f $model ]; then
        echo "Downloading latest model"
        wget https://media.xiph.org/lpcnet/data/$model
fi
tar xvf $model
touch src/nnet_data.[ch]
