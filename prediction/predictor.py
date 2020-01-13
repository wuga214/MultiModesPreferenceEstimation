from functools import partial
from multiprocessing import Process
from tqdm import tqdm

import multiprocessing
import numpy as np


def predict(matrix_U, matrix_V, topK, matrix_Train, bias=None, measure="Cosine", gpu=False):
    # Disable gpu option since CC doesn't like Cupy
    gpu = False

    if gpu:
        import cupy as cp
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)

    prediction = []

    for user_index in tqdm(range(matrix_U.shape[0])):
        vector_u = matrix_U[user_index]
        vector_train = matrix_Train[user_index]
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(vector_u, matrix_V, vector_train, bias, measure, topK=topK, gpu=gpu)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        prediction.append(vector_predict)

    return np.vstack(prediction)


def sub_routine(vector_u, matrix_V, vector_train, bias, measure, topK=500, gpu=False):

    train_index = vector_train.nonzero()[1]
    if measure == "Cosine":
        if len(vector_u.shape) > 1:
            # import ipdb; ipdb.set_trace()
            vector_predict = np.max(matrix_V.dot(vector_u.T), axis=1)
        else:
            vector_predict = matrix_V.dot(vector_u)
    else:
        if gpu:
            import cupy as cp
            vector_predict = -cp.sum(cp.square(matrix_V - vector_u), axis=1)
        else:
            vector_predict = -np.sum(np.square(matrix_V - vector_u), axis=1)
    if bias is not None:
        if gpu:
            import cupy as cp
            vector_predict = vector_predict + cp.array(bias)
        else:
            vector_predict = vector_predict + bias

    if gpu:
        import cupy as cp
        candidate_index = cp.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
        vector_predict = cp.asnumpy(vector_predict).astype(np.float32)
    else:
        candidate_index = np.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]
