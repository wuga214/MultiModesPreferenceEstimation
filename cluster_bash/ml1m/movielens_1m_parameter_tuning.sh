#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/MultiModesPreferenceEstimation
python tune_parameters.py --data-dir datax/ --save-path movielens1m/mmp.csv --parameters config/mmp.yml --gpu

#python reproduce_paper_results.py -p tables/movielens1m -d datax/ -v Rvalid.npz -n movielens1m_test_result_gpu.csv -gpu
#python reproduce_paper_results.py -p tables/movielens1m -d datax/ -v Rvalid.npz -n movielens1m_test_result_cpu.csv
