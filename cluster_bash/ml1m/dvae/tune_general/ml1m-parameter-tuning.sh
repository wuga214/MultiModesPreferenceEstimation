#!/usr/bin/env bash
cd /home/maizheda/projects/def-ssanner/maizheda/MultiModesPreferenceEstimation
python tune_parameters_new.py --parameters config/ml1m/dvae.yml --default_params config/default_dvae.yml --data-dir datax/ --save-path ml1m/dvae/tuning_general/dvae.csv

