#!/usr/bin/env bash
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 ml1m-parameter-tuning-part1.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 ml1m-parameter-tuning-part2.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 ml1m-parameter-tuning-part3.sh
