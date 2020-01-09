#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir data/netflix/ --save-path netflix/mmp-part2.csv --parameters config/netflix/mmp-part2.yml
