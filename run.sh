#!/bin/bash
set -e

source="$1"
python -m msgwam config/${source}.toml data/${source}.nc
python plot.py data/${source}.nc plots/${source}.png
