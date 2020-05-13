#!/usr/bin/env bash
cd /home/maizheda/projects/def-ssanner/maizheda/MultiModesPreferenceEstimation
python tune_parameters_new.py --parameters config/amazon/digital_music/dvae_3.yml --default_params config/default_dvae.yml --data-dir data/amazon/digital_music/ --save-path amazon/digital_music/dvae/tuning_general/dvae_3.csv

