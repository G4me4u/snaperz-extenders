#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
mkdir -p "$BASEDIR/build" &&
cd "$BASEDIR/build" &&
cmake -DCMAKE_BUILD_TYPE=Release .. &&
make all
