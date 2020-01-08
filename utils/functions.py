from tqdm import tqdm
import numpy as np
import scipy.sparse as sparse


def get_pmi_matrix(matrix, root):
    rows, cols = matrix.shape
    item_rated = matrix.sum(axis=0)
    # user_rated = matrix.sum(axis=1)
    nnz = matrix.nnz
    pmi_matrix = []
    for i in tqdm(range(rows)):
        row_index, col_index = matrix[i].nonzero()
        if len(row_index) > 0:
            # import ipdb; ipdb.set_trace()
            # values = np.asarray(user_rated[i].dot(item_rated)[:, col_index]).flatten()
            values = np.asarray(item_rated[:, col_index]).flatten()
            values = np.maximum(np.log(rows/np.power(values, root)), 0)
            pmi_matrix.append(sparse.coo_matrix((values, (row_index, col_index)), shape=(1, cols)))
        else:
            pmi_matrix.append(sparse.coo_matrix((1, cols)))
    return sparse.vstack(pmi_matrix)


def get_pmi_matrix_gpu(matrix, root):
    import cupy as cp
    rows, cols = matrix.shape
    item_rated = cp.array(matrix.sum(axis=0))
    pmi_matrix = []
    nnz = matrix.nnz
    for i in tqdm(xrange(rows)):
        row_index, col_index = matrix[i].nonzero()
        if len(row_index) > 0:
            values = cp.asarray(item_rated[:, col_index]).flatten()
            values = cp.maximum(cp.log(rows/cp.power(values, root)), 0)
            pmi_matrix.append(sparse.coo_matrix((cp.asnumpy(values), (row_index, col_index)), shape=(1, cols)))
        else:
            pmi_matrix.append(sparse.coo_matrix((1, cols)))
    return sparse.vstack(pmi_matrix)