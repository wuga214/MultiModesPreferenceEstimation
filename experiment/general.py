from evaluation.metrics import evaluate
from prediction.predictor import predict
from tqdm import tqdm
from utils.progress import WorkSplitter
from types import SimpleNamespace
import inspect
import numpy as np
import os
import pandas as pd
from models.dvae import dvae

def general(train, test, params, model, measure='Cosine', gpu_on=True, analytical=False, model_folder='latent'):
    progress = WorkSplitter()

    columns = ['model', 'similarity', 'alpha', 'batch_size', 'corruption',
               'epoch', 'iteration', 'key_dimension', 'lambda', 'learning_rate',
               'mode_dimension', 'normalize', 'rank', 'root', 'topK']

    progress.section("\n".join([":".join((str(k), str(params[k]))) for k in columns]))

    df = pd.DataFrame(columns=columns)

    if os.path.isfile('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], model_folder)):

        RQ = np.load('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], model_folder))
        Y = np.load('{2}/V_{0}_{1}.npy'.format(params['model'], params['rank'], model_folder))

        if os.path.isfile('{2}/B_{0}_{1}.npy'.format(params['model'], params['rank'], model_folder)):
            Bias = np.load('{2}/B_{0}_{1}.npy'.format(params['model'], params['rank'], model_folder))
        else:
            Bias = None

    else:

        RQ, Yt, Bias = model(train,
                             embedded_matrix=np.empty((0)),
                             mode_dim=params['mode_dimension'],
                             key_dim=params['key_dimension'],
                             batch_size=params['batch_size'],
                             learning_rate=params['learning_rate'],
                             iteration=params['iteration'],
                             epoch=params['epoch'],
                             rank=params['rank'],
                             corruption=params['corruption'],
                             gpu_on=gpu_on,
                             lamb=params['lambda'],
                             alpha=params['alpha'],
                             root=params['root'])

        Y = Yt.T

        """
        np.save('{2}/U_{0}_{1}'.format(params['model'], params['rank'], model_folder), RQ)
        np.save('{2}/V_{0}_{1}'.format(params['model'], params['rank'], model_folder), Y)
        if Bias is not None:
            np.save('{2}/B_{0}_{1}'.format(params['model'], params['rank'], model_folder), Bias)
        """

    progress.subsection("Prediction")

    prediction = predict(matrix_U=RQ,
                         matrix_V=Y,
                         measure=measure,
                         bias=Bias,
                         topK=params['topK'][-1],
                         matrix_Train=train,
                         gpu=gpu_on)

    progress.subsection("Evaluation")

    result = evaluate(prediction, test, params['metric'], params['topK'], analytical=analytical)

    if analytical:
        return result
    else:
        result_dict = params

        for name in result.keys():
            result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
        df = df.append(result_dict, ignore_index=True)

        return df

def general_dvae(train_R, test, params, defaul_params, metrics, topK, data, analytical=False):
    progress = WorkSplitter()

    columns = params.keys()

    progress.section("\n".join([":".join((str(k), str(params[k]))) for k in columns]))

    df = pd.DataFrame(columns=columns)
    defaul_params.update(params)
    final_params = SimpleNamespace(**defaul_params)

    progress.subsection("Prediction")
    log_dir = 'final_%s-%dT-%dB-%glr-%grg-%gkp-%gb-%gt-%gs-%dk-%dd-%d' % (
        data, final_params.epoch, final_params.batch_size, final_params.lr, final_params.lam, final_params.keep,
        final_params.beta, final_params.tau, final_params.std, final_params.kfac, final_params.dfac, final_params.seed)
    if final_params.nogb:
        log_dir += '-nogb'
    log_dir = os.path.join(final_params.logdir, log_dir)
    prediction = dvae(log_dir, train_R, train_R.shape[1], final_params.kfac, final_params.dfac, final_params.lam,
                      final_params.lr, final_params.seed, final_params.tau, final_params.std,
                      final_params.nogb, final_params.beta,
                      final_params.keep, final_params.topK[-1], final_params.epoch, final_params.batch_size)

    progress.subsection("Evaluation")

    result = evaluate(prediction, test, metrics, topK, analytical=analytical)

    if analytical:
        return result
    else:
        result_dict = params

        for name in result.keys():
            result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
        df = df.append(result_dict, ignore_index=True)

        return df
