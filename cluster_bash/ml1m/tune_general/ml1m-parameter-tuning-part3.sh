#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir datax/ --save-path ml1m/tuning_general/mmp-part3.csv --parameters config/ml1m/mmp-part3.yml --gpu
