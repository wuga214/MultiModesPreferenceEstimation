#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir data/amazon/video_games/ --save-path amazon/video_games/tuning_general/mmp-part244.csv --parameters config/amazon/video_games/mmp-part244.yml
