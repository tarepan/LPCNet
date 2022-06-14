#!/bin/sh
# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

# Construct env generation script `./configure`
echo "Updating build configuration files for lpcnet, please wait...."
autoreconf -isf
