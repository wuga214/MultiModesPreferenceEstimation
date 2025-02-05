from experiment.general import general
from plots.rec_plots import precision_recall_curve
from utils.io import load_numpy, save_dataframe_csv, find_best_hyperparameters, load_yaml
from utils.modelnames import models

import argparse
import pandas as pd
import timeit


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    df = find_best_hyperparameters(table_path+args.tuning_result_path, 'NDCG')

    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    R_valid = load_numpy(path=args.data_dir, name=args.valid_set)
    R_test = load_numpy(path=args.data_dir, name=args.test_set)

    R_train = R_train + R_valid

    topK = [5, 10, 15, 20, 50]

    frame = []
    for idx, row in df.iterrows():
        start = timeit.default_timer()
        row = row.to_dict()
        row['metric'] = ['R-Precision', 'NDCG', 'Precision', 'Recall', "MAP"]
        row['topK'] = topK
        result = general(R_train,
                         R_test,
                         row,
                         models[row['model']],
                         measure=row['similarity'],
                         gpu_on=args.gpu,
                         model_folder=args.model_folder)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        frame.append(result)

    results = pd.concat(frame)
    save_dataframe_csv(results, table_path, args.save_path)
#    precision_recall_curve(results, topK, save=True, folder='analysis/'+args.tuning_result_path)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce Final General Recommendation Performance")

    parser.add_argument('--data-dir', dest='data_dir', default="datax/")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--model-model', dest='model_folder', default='latent') # Model saving folder
    parser.add_argument('--save-path', dest='save_path', default="ml1m/final_general/mmp_final.csv")
    parser.add_argument('--test', dest='test_set', default='Rtest.npz')
    parser.add_argument('--train', dest='train_set', default='Rtrain.npz')
    parser.add_argument('--tuning-result-path', dest='tuning_result_path', default="ml1m/tuning_general")
    parser.add_argument('--valid', dest='valid_set', default='Rvalid.npz')

    args = parser.parse_args()

    main(args)
