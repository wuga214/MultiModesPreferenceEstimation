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
    for i in tqdm(range(rows)):
        row_index, col_index = matrix[i].nonzero()
        if len(row_index) > 0:
            values = cp.asarray(item_rated[:, col_index]).flatten()
            values = cp.maximum(cp.log(rows/cp.power(values, root)), 0)
            pmi_matrix.append(sparse.coo_matrix((cp.asnumpy(values), (row_index, col_index)), shape=(1, cols)))
        else:
            pmi_matrix.append(sparse.coo_matrix((1, cols)))
    return sparse.vstack(pmi_matrix)


def get_attention_example_items(inputs, outputs, size=9):
    rows, cols = inputs.shape

    values = []

    for i in tqdm(range(rows)):
        input_row = inputs[i]
        output_row = outputs[i]

        input_nonzeros = np.nonzero(input_row)[0]
        if len(input_nonzeros) < size:
            input_zeros = np.nonzero(1-input_row)[0]

            input_candidate = np.hstack([input_nonzeros, np.random.choice(input_zeros, size-len(input_nonzeros),
                                                                          replace=False)])

        else:
            input_candidate = np.random.choice(input_nonzeros, size, replace=False)

        output_candidate = np.argsort(output_row)[::-1][:size]

        values.append([input_candidate, output_candidate])

    return values


def write_latex(samples, attentions, kernels, latex_template):

    for instance in samples:
        input = instance[0]
        output = instance[1]
        attention = attentions[input]
        kernel = kernels[output]

        feeds = dict()
        for i in range(len(input)):
            feeds['item{0}'.format(i+1)] = input[i]
            feeds['recommend{0}'.formate(i+1)] = output[i]



        tex = latex_template.format(
        )

        #under construction