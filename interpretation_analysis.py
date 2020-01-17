import numpy as np
from utils.io import load_numpy, load_dataframe_csv
from experiment.attention import attention

import argparse


def main(args):

    settings_df = load_dataframe_csv(args.tab_path+"/ml1m/final_general/mmp_final.csv")

    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    R_valid = load_numpy(path=args.data_dir, name=args.valid_set)
    R_test = load_numpy(path=args.data_dir, name=args.test_set)

    index_map = np.load(args.data_dir+args.index)

    item_names = None

    try:
        item_names = load_dataframe_csv(args.data_dir+args.names, delimiter="::", names=['ItemID', 'Name', 'Category'])
    except:
        print("Meta-data does not exist")

    attention(R_train, R_valid, R_test, index_map, item_names, args.tex_path, args.fig_path, settings_df,
              args.template_path,
              case_study=args.case_study,
              gpu_on=True)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Interpretation Analysis")

    parser.add_argument('--data-dir', dest='data_dir', default="datax/")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--tex-path', dest='tex_path', default="texs")
    parser.add_argument('--index', dest='index', default="Index.npy")
    parser.add_argument('--item-names', dest='names', default="ml-1m/movies.dat")
    parser.add_argument('--fig-path', dest='fig_path', default="figs/attention_demos")
    parser.add_argument('--table-path', dest='tab_path', default="tables")
    parser.add_argument('--template-path', dest='template_path', default="templates/attention.tex")
    parser.add_argument('--test', dest='test_set', default='Rtest.npz')
    parser.add_argument('--train', dest='train_set', default='Rtrain.npz')
    parser.add_argument('--valid', dest='valid_set', default='Rvalid.npz')
    parser.add_argument('--case-study', dest='case_study', action='store_true')

    args = parser.parse_args()

    main(args)
