#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir data/netflix/ --save-path netflix/mmp-part5.csv --parameters config/netflix/mmp-part5.yml
