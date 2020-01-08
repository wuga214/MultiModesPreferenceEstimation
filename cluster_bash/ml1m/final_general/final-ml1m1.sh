#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python reproduce_general_results.py --data-dir datax/ --tuning-result-path ml1m/tuning_general --save-path ml1m/final_general/mmp_final1.csv
