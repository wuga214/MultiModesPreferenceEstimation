#!/usr/bin/env bash
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner ml1m-parameter-tuning.sh
