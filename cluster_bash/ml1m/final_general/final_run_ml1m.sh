#!/usr/bin/env bash
sbatch --nodes=1 --time=1:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-ml1m1.sh
sbatch --nodes=1 --time=1:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-ml1m2.sh
sbatch --nodes=1 --time=1:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-ml1m3.sh
sbatch --nodes=1 --time=1:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-ml1m4.sh
sbatch --nodes=1 --time=1:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-ml1m5.sh
