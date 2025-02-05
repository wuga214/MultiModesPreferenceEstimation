from tqdm import tqdm
import datetime
import os
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


def get_attention_example_items(inputs, outputs, labels, size=9):
    rows, cols = inputs.shape

    values = []

    for i in tqdm(range(rows)):
        input_row = inputs[i]
        output_row = outputs[i]
        label_row = labels[i]

        input_nonzeros = np.nonzero(input_row)[1]
        label_nonzeros = np.nonzero(label_row)[1]
        if len(input_nonzeros) < size:

            input_zeros = np.nonzero(1-input_row.todense())[1]

            input_candidate = np.hstack([input_nonzeros, np.random.choice(input_zeros, size-len(input_nonzeros),
                                                                          replace=False)])

        else:
            input_candidate = np.random.choice(input_nonzeros, size, replace=False)

        output_candidate = np.argsort(output_row)[::-1][:size+len(input_nonzeros)]

        output_candidate = np.delete(output_candidate, np.isin(output_candidate, input_nonzeros).nonzero()[0])[:size]

        overlap = [1 if x in label_nonzeros else 0 for x in output_candidate]

        values.append([input_candidate, output_candidate, overlap])

    return values


def read_template(path):
    with open(path, 'r') as file:
        return file.read()


def write_latex(samples, attentions, kernels, item_names, latex_template, path, size=9):

    for index, instance in enumerate(samples):
        input = instance[0][:size]
        output = instance[1][:size]
        overlap = instance[2][:size]

        if np.sum(overlap) < 5:
            continue

        attention = attentions[index].T[input]
        kernel = kernels[index][output]

        # attention = (attention-np.min(attention)) / (np.max(attention)-np.min(attention)+0.000001) * 100

        attention = np.square(attention)
        attention = attention/np.sum(attention, axis=0)

        feeds = dict()
        for i in range(len(input)):
            feeds['item{0}'.format(i+1)] = item_names[input[i]]
            feeds['recommend{0}'.format(i+1)] = item_names[output[i]]
            feeds['overlap{0}'.format(i+1)] = overlap[i]*30

        feeds['preference1'] = "Preference1"
        feeds['preference2'] = "Preference2"
        feeds['preference3'] = "Preference3"

        m, n = attention.shape
        for i in range(m):
            for j in range(n):
                feeds['attention_{0}_{1}'.format(i+1, j+1)] = attention[i][j]

        m = kernel.shape[0]

        for i in range(m):
            for j in range(3):
                if j == kernel[i]:
                    feeds['kernel_{0}_{1}'.format(j+1, i+1)] = 100
                else:
                    feeds['kernel_{0}_{1}'.format(j+1, i+1)] = 0

        timestr = datetime.datetime.now().strftime("%H:%M:%S:%f")

        latex_component_list = latex_template.split('###')

        # import ipdb;ipdb.set_trace()

        modified_component = latex_component_list[1].format(**feeds)

        modified_component = "\n".join([x for x in modified_component.split('\n') if "!0]" not in x])

        tex = "".join([latex_component_list[0], modified_component, latex_component_list[2]])

        tex = tex.replace("<", "{")
        tex = tex.replace(">", "}")
        tex = tex.replace("&", "")

        with open(os.path.join(path, '{0}.tex'.format(timestr)), 'w') as the_file:
            the_file.write(tex)

    return
