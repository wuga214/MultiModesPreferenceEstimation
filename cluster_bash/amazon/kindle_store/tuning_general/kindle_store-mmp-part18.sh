#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir data/amazon/kindle_store/ --save-path amazon/kindle_store/tuning_general/mmp-part18.csv --parameters config/amazon/kindle_store/mmp-part18.yml
