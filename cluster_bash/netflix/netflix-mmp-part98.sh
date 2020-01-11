#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir data/netflix/ --save-path netflix/tuning_general/mmp-part98.csv --parameters config/netflix/mmp-part98.yml
