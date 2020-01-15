#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir data/netflix/ --save-path netflix/tuning_general/mmp-part131.csv --parameters config/netflix/mmp-part131.yml
