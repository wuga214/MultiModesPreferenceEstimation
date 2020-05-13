#!/usr/bin/env bash
cd /home/maizheda/projects/def-ssanner/maizheda/MultiModesPreferenceEstimation
python tune_parameters_new.py --parameters config/amazon/video_games/dvae_10.yml --default_params config/default_dvae.yml --data-dir data/amazon/video_games/ --save-path amazon/video_games/dvae/tuning_general/dvae_10.csv

