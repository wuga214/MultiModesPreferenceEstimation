from evaluation.metrics import evaluate
from prediction.predictor import predict
from utils.argcheck import check_float_positive, check_int_positive, shape
from utils.io import load_numpy, load_pandas, load_csv
from utils.modelnames import models
from utils.progress import WorkSplitter, inhour

import argparse
import numpy as np
import time


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.data_dir))
    print("Train File Name: {0}".format(args.train_set))
    if args.validation:
        print("Valid File Name: {0}".format(args.valid_set))
    print("Algorithm: {0}".format(args.model))
    if args.item == True:
        mode = "Item-based"
    else:
        mode = "User-based"
    print("Mode: {0}".format(mode))
    print("Alpha: {0}".format(args.alpha))
    print("Rank: {0}".format(args.rank))
    print("Mode Dimension: {0}".format(args.mode_dim))
    print("Key Dimension: {0}".format(args.key_dim))
    print("Batch Size: {0}".format(args.batch_size))
    print("Optimizer: {0}".format(args.optimizer))
    print("Learning Rate: {0}".format(args.learning_rate))
    print("Lambda: {0}".format(args.lamb))
    print("SVD/Alter Iteration: {0}".format(args.iteration))
    print("Epoch: {0}".format(args.epoch))
    print("Corruption: {0}".format(args.corruption))
    print("Root: {0}".format(args.root))
    print("Evaluation Ranking Topk: {0}".format(args.topk))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    if args.shape is None:
        R_train = load_numpy(path=args.data_dir, name=args.train_set)
    else:
        # R_train = load_pandas(path=args.data_dir, name=args.train_set, shape=args.shape)
        R_train = load_csv(path=args.data_dir, name=args.train_set, shape=args.shape)

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    print("Train U-I Dimensions: {0}".format(R_train.shape))

    # Item-Item or User-User
    if args.item == True:
        RQ, Yt, Bias = models[args.model](R_train, embedded_matrix=np.empty((0)), mode_dim=args.mode_dim,
                                          key_dim=args.key_dim, batch_size=args.batch_size, optimizer=args.optimizer,
                                          learning_rate=args.learning_rate,
                                          iteration=args.iteration, epoch=args.epoch, rank=args.rank,
                                          corruption=args.corruption, gpu_on=args.gpu,
                                          lamb=args.lamb, alpha=args.alpha, seed=args.seed, root=args.root)
        Y = Yt.T
    else:
        Y, RQt, Bias = models[args.model](R_train.T, embedded_matrix=np.empty((0)), mode_dim=args.mode_dim,
                                          key_dim=args.key_dim, batch_size=args.batch_size, optimizer=args.optimizer,
                                          learning_rate=args.learning_rate,
                                          iteration=args.iteration, rank=args.rank,
                                          corruption=args.corruption, gpu_on=args.gpu,
                                          lamb=args.lamb, alpha=args.alpha, seed=args.seed, root=args.root)
        RQ = RQt.T

    # np.save('latent/U_{0}_{1}'.format(args.model, args.rank), RQ)
    # np.save('latent/V_{0}_{1}'.format(args.model, args.rank), Y)
    # if Bias is not None:
    #     np.save('latent/B_{0}_{1}'.format(args.model, args.rank), Bias)

    progress.section("Predict")
    prediction = predict(matrix_U=RQ,
                         matrix_V=Y,
                         bias=Bias,
                         topK=args.topk,
                         matrix_Train=R_train,
                         measure=args.sim_measure,
                         gpu=args.gpu)
    if args.validation:
        progress.section("Create Metrics")
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision']
        R_valid = load_numpy(path=args.data_dir, name=args.valid_set)
        result = evaluate(prediction, R_valid, metric_names, [args.topk])
        print("-")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="MMP")

    parser.add_argument('--alpha', dest='alpha', type=check_float_positive, default=100.0)
    parser.add_argument('--batch-size', dest='batch_size', type=check_int_positive, default=32)
    parser.add_argument('--corruption', dest='corruption', type=check_float_positive, default=0.5)
    parser.add_argument('--data-dir', dest='data_dir', default="datax/")
    parser.add_argument('--disable-item-item', dest='item', action='store_false')
    parser.add_argument('--disable-validation', dest='validation', action='store_false')
    parser.add_argument('--epoch', dest='epoch', type=check_int_positive, default=1)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--iteration', dest='iteration', type=check_int_positive, default=1)
    parser.add_argument('--key-dimension', dest='key_dim', type=check_int_positive, default=3)
    parser.add_argument('--lambda', dest='lamb', type=check_float_positive, default=100)
    parser.add_argument('--learning-rate', dest='learning_rate', type=check_float_positive, default=0.001)
    parser.add_argument('--model', dest='model', default="WRMF")
    parser.add_argument('--mode-dimension', dest='mode_dim', type=check_int_positive, default=5)
    parser.add_argument('--optimizer', dest='optimizer', default="Adam")
    parser.add_argument('--rank', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('--root', dest='root', type=check_float_positive, default=1)
    parser.add_argument('--seed', dest='seed', type=check_int_positive, default=1)
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    parser.add_argument('--similarity', dest='sim_measure', default='Cosine')
    parser.add_argument('--topk', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('--train', dest='train_set', default='Rtrain.npz')
    parser.add_argument('--valid', dest='valid_set', default='Rvalid.npz')
    args = parser.parse_args()

    main(args)
