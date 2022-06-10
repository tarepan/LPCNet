#!/bin/sh
set -e

# `$1`: model specifier
# `$2`: 1 if efficiency mode else 0

model=lpcnet_data-$1.tar.gz

# Mode
if [ $2 -eq 1 ]; then
        echo "LPCNet-efficiency mode"
        eff=lpcnet_efficiency/
fi

# Data download
if [ ! -f $model ]; then
        echo "Downloading latest model"
        wget "https://media.xiph.org/lpcnet/data/${eff}${model}"
fi
tar xvf $model
touch src/nnet_data.[ch]
