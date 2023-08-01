#!/bin/bash
set -e

source="desaubies"
python run.py config-${source}.toml data/${source}.nc
python plot.py data/${source}.nc plots/${source}.png
