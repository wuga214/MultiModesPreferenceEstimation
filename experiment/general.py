from evaluation.metrics import evaluate
from prediction.predictor import predict
from tqdm import tqdm
from utils.progress import WorkSplitter

import inspect
import numpy as np
import os
import pandas as pd


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
