from experiment.tune_new import hyper_parameter_tuning
from utils.io import load_numpy, save_dataframe_csv, load_yaml
from utils.modelnames import models

import argparse
import numpy as np


def main(args):
    params = load_yaml(args.parameters)
    defaul_params = load_yaml(args.default_params)
    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    R_valid = load_numpy(path=args.data_dir, name=args.valid_set)
    data_name = args.data_dir.replace('/', '_')
    hyper_parameter_tuning(data_name, R_train, R_valid, params, defaul_params, save_path=args.save_path)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")

    parser.add_argument('--data-dir', dest='data_dir', default="datax/")
    parser.add_argument('--parameters', dest='parameters', default='config/default.yml')
    parser.add_argument('--default_params', dest='default_params', default='config/default_dvae.yml')
    parser.add_argument('--save-path', dest='save_path', default="mmp_tuning.csv")
    parser.add_argument('--train', dest='train_set', default='Rtrain.npz')
    parser.add_argument('--valid', dest='valid_set', default='Rvalid.npz')
    args = parser.parse_args()

    main(args)
