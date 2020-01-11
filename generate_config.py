from utils.io import write_file

import argparse


def main(args):

    key_dimensions = ['3', '4', '5']
    lambs = ['0.001', '0.0001', '0.00001', '0.000001']
    learning_rates = ['0.00005', '0.0001', '0.0005']
    mode_dimensions = ['1', '3']
    ranks = ['60', '80', '100']

    pattern = "parameters:\n" \
              "    models: MMP\n" \
              "    similarity: Cosine\n" \
              "    alpha: [0, 0.1, 1]\n" \
              "    batch_size: [32]\n" \
              "    corruption: [0.2, 0.3, 0.4]\n" \
              "    epoch: [300]\n" \
              "    iteration: [10]\n" \
              "    key_dimension: [{0}]\n" \
              "    lambda: [{1}]\n" \
              "    learning_rate: [{2}]\n" \
              "    mode_dimension: [{3}]\n" \
              "    rank: [{4}]\n" \
              "    root: [1.0]\n" \
              "    topK: [5, 10, 15, 20, 50]\n" \
              "    metric: [R-Precision, NDCG, Precision, Recall]"

    i = 1

    for key_dimension in key_dimensions:
        for lamb in lambs:
            for learning_rate in learning_rates:
                for mode_dimension in mode_dimensions:
                    for rank in ranks:
                        content = pattern.format(key_dimension, lamb, learning_rate, mode_dimension, rank)
                        write_file('config/'+args.path+'/', 'mmp-part'+str(i)+'.yml', content, exe=False)
                        i += 1


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Create Config")

    parser.add_argument('--path', dest='path', default="amazon/digital_music")

    args = parser.parse_args()

    main(args)
