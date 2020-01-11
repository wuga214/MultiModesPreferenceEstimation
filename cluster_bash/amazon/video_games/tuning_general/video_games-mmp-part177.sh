#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir data/amazon/video_games/ --save-path amazon/video_games/tuning_general/mmp-part177.csv --parameters config/amazon/video_games/mmp-part177.yml
