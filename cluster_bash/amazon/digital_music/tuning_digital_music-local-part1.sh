#!/usr/bin/env bash
cd ~/git/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir data/amazon/digital_music/ --save-path amazon/digital_music/tuning_general/mmp-part1.csv --parameters config/amazon/digital_music/mmp-part1.yml
