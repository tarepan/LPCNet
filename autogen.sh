#!/bin/sh
# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

#SHA1 of the first commit compatible with the current model (`72061bc` @ 2021-10-25T20:03)
commit=72061bc
./download_model.sh $commit

echo "Updating build configuration files for lpcnet, please wait...."

autoreconf -isf
