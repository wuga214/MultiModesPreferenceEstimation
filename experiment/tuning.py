from evaluation.metrics import evaluate
from prediction.predictor import predict
from tqdm import tqdm
from utils.progress import WorkSplitter
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml

import inspect
import numpy as np
import pandas as pd


def hyper_parameter_tuning(train, validation, params, save_path, measure='Cosine', gpu_on=True):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']

    try:
        df = load_dataframe_csv(table_path, save_path)
    except:
        df = pd.DataFrame(columns=['model', 'similarity', 'alpha', 'batch_size',
                                   'corruption', 'epoch', 'iteration', 'key_dimension',
                                   'lambda', 'learning_rate', 'mode_dimension',
                                   'normalize', 'rank', 'root', 'topK'])

    num_user = train.shape[0]

    for algorithm in params['models']:

        for alpha in params['alpha']:

            for batch_size in params['batch_size']:

                for corruption in params['corruption']:

                    for epoch in params['epoch']:

                        for iteration in params['iteration']:

                            for key_dim in params['key_dimension']:

                                for lamb in params['lambda']:

                                    for learning_rate in params['learning_rate']:

                                        for mode_dim in params['mode_dimension']:

                                            for rank in params['rank']:

                                                for root in params['root']:

                                                    if ((df['model'] == algorithm) &
                                                        (df['alpha'] == alpha) &
                                                        (df['batch_size'] == batch_size) &
                                                        (df['corruption'] == corruption) &
                                                        (df['epoch'] == epoch) &
                                                        (df['iteration'] == iteration) &
                                                        (df['key_dimension'] == key_dim) &
                                                        (df['lambda'] == lamb) &
                                                        (df['learning_rate'] == learning_rate) &
                                                        (df['mode_dimension'] == mode_dim) &
                                                        (df['rank'] == rank) &
                                                        (df['root'] == root)).any():
                                                        continue

                                                    format = "model: {}, alpha: {}, batch_size: {}, corruption: {}, epoch: {}, iteration: {}, \
                                                        key_dimension: {}, lambda: {}, learning_rate: {}, mode_dimension: {}, rank: {}, root: {}"
                                                    progress.section(format.format(algorithm, alpha, batch_size, corruption, epoch, iteration,
                                                                                   key_dim, lamb, learning_rate, mode_dim, rank, root))
                                                    RQ, Yt, Bias = params['models'][algorithm](train,
                                                                                               embedded_matrix=np.empty((0)),
                                                                                               mode_dim=mode_dim,
                                                                                               key_dim=key_dim,
                                                                                               batch_size=batch_size,
                                                                                               learning_rate=learning_rate,
                                                                                               iteration=iteration,
                                                                                               epoch=epoch,
                                                                                               rank=rank,
                                                                                               corruption=corruption,
                                                                                               gpu_on=gpu_on,
                                                                                               lamb=lamb,
                                                                                               alpha=alpha,
                                                                                               root=root)
                                                    Y = Yt.T

                                                    progress.subsection("Prediction")

                                                    prediction = predict(matrix_U=RQ,
                                                                         matrix_V=Y,
                                                                         bias=Bias,
                                                                         topK=params['topK'][-1],
                                                                         matrix_Train=train,
                                                                         measure=measure,
                                                                         gpu=gpu_on)

                                                    progress.subsection("Evaluation")

                                                    result = evaluate(prediction,
                                                                      validation,
                                                                      params['metric'],
                                                                      params['topK'])

                                                    result_dict = {'model': algorithm,
                                                                   'alpha': alpha,
                                                                   'batch_size': batch_size,
                                                                   'corruption': corruption,
                                                                   'epoch': epoch,
                                                                   'iteration': iteration,
                                                                   'key_dimension': key_dim,
                                                                   'lambda': lamb,
                                                                   'learning_rate': learning_rate,
                                                                   'mode_dimension': mode_dim,
                                                                   'rank': rank,
                                                                   'similarity': params['similarity'],
                                                                   'root': root}

                                                    for name in result.keys():
                                                        result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]

                                                    df = df.append(result_dict, ignore_index=True)

                                                    save_dataframe_csv(df, table_path, save_path)
