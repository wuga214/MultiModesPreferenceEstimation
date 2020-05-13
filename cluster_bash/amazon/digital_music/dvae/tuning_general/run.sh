#!/usr/bin/env bash

sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner 1.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner 2.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner 3.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner 4.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner 5.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner 6.sh
