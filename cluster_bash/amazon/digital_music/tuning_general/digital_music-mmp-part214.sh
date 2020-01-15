#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir data/amazon/digital_music/ --save-path amazon/digital_music/tuning_general/mmp-part214.csv --parameters config/amazon/digital_music/mmp-part214.yml
