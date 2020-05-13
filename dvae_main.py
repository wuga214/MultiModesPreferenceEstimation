import argparse
import os
import time
from evaluation.metrics import evaluate
from models.dvae import dvae, predict
from utils.io import load_numpy
from utils.progress import WorkSplitter, inhour
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args):
    log_dir = '%s-%dT-%dB-%glr-%grg-%gkp-%gb-%gt-%gs-%dk-%dd-%d' % (
        parser.data_dir.replace('/', '_'), parser.epoch, parser.batch_size, parser.lr, parser.lam, parser.keep,
        parser.beta, parser.tau, parser.std, parser.kfac, parser.dfac, parser.seed)
    if parser.nogb:
        log_dir += '-nogb'
    log_dir = os.path.join(parser.logdir, log_dir)
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.data_dir))
    print("Train File Name: {0}".format(args.train_set))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    print("Train U-I Dimensions: {0}".format(R_train.shape))
    dvae(log_dir, R_train, R_train.shape[1], args.kfac, args.dfac, args.lam, args.lr, args.seed, args.tau, args.std,
         args.nogb, args.beta,
         args.keep, args.topk, args.epoch, args.batch_size)
    progress.section("Predict")
    prediction = predict(R_train, args.kfac, args.dfac, args.lam, args.lr, args.seed, args.tau, args.std, args.nogb,
                         args.topk, args.batch_size, log_dir)
    progress.section("Create Metrics")
    start_time = time.time()

    metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision']
    R_valid = load_numpy(path=args.data_dir, name=args.valid_set)
    result = evaluate(prediction, R_valid, metric_names, [5, 10, 20, 50 ])
    print("-")
    for metric in result.keys():
        print("{0}:{1}".format(metric, result[metric]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, required=True,
    #                  help='./data/ml-latest-small, ./data/ml-1m, '
    #                       './data/ml-20m, or ./data/alishop-7c')
    # parser.add_argument('--mode', type=str, default='trn',
    #                  help='trn/tst/vis, for training/testing/visualizing.')
    parser.add_argument('--data-dir', dest='data_dir', default="datax/")
    parser.add_argument('--train', dest='train_set', default='Rtrain.npz')
    parser.add_argument('--logdir', type=str, default='runs/')
    parser.add_argument('--seed', type=int, default=98765,
                        help='Random seed. Ignored if < 0.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--lam', type=float, default=0.0,
                        help='L2 regularization.')
    parser.add_argument('--keep', type=float, default=0.5,
                        help='Keep probability for dropout, in (0,1].')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Strength of disentanglement, in (0,oo).')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='Temperature of sigmoid/softmax, in (0,oo).')
    parser.add_argument('--std', type=float, default=0.075,
                        help='Standard deviation of the Gaussian prior.')
    parser.add_argument('--kfac', type=int, default=7,
                        help='Number of facets (macro concepts).')
    parser.add_argument('--dfac', type=int, default=100,
                        help='Dimension of each facet.')
    parser.add_argument('--nogb', action='store_true', default=False,
                        help='Disable Gumbel-Softmax sampling.')
    parser.add_argument('--topk', dest='topk', default=20)
    parser.add_argument('--valid', dest='valid_set', default='Rvalid.npz')
    parser = parser.parse_args()
    main(parser)
