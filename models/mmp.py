import numpy as np
import scipy.sparse as sparse
from fbpca import pca
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import inv
from sklearn.utils.extmath import randomized_svd
from utils.progress import WorkSplitter, inhour
import time

import tensorflow as tf
from tqdm import tqdm
from utils.regularizers import Regularizer


class MultiModesPreferenceEstimation(object):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 mode_dim,
                 key_dim,
                 batch_size,
                 lamb=0.01,
                 learning_rate=1e-4,
                 optimizer=tf.train.RMSPropOptimizer,
                 user_embeddings=None,
                 item_embeddings=None,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.embed_dim = embed_dim
        self.mode_dim = mode_dim
        self.key_dim = key_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, self.input_dim+1), name="inputs")
        inputs = self.inputs[:, :-1]
        self.user_idx = tf.keras.backend.cast(self.inputs[:, -1], tf.int32)

        user_embeddings = tf.constant(self.user_embeddings, name="user_embeddings")
        item_embeddings = tf.constant(self.item_embeddings, name="item_embeddings")

        with tf.variable_scope('keys'):
            item_key_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.key_dim], stddev=1 / 500.0),
                                           name="item_key_weights")
            item_keys = tf.matmul(item_embeddings, item_key_weights)

            item_value_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.embed_dim], stddev=1 / 500.0),
                                             name="item_value_weights")

            item_values = tf.nn.relu(tf.matmul(item_embeddings, item_value_weights))

            item_query_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.embed_dim], stddev=1 / 500.0),
                                             name="item_value_weights")

            self.item_query = tf.matmul(item_embeddings, item_query_weights)

            user_key_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.key_dim*self.mode_dim],
                                                               stddev=1 / 500.0),
                                           name="user_key_weights")
            user_keys = tf.reshape(tf.matmul(user_embeddings, user_key_weights),
                                   [-1, self.key_dim, self.mode_dim])

        with tf.variable_scope('embedding_lookup'):
            selected_user_keys = tf.nn.embedding_lookup(user_keys, self.user_idx, name="selected_users")

        with tf.variable_scope('encoding'):
            attention = tf.matmul(tf.multiply(tf.expand_dims(inputs, -1), item_keys), selected_user_keys)
            attention = tf.nn.softmax(tf.transpose(attention, perm=[0, 2, 1]), axis=2)

            self.user_latent = tf.tensordot(attention, item_values, axes=[[2], [0]])

        with tf.variable_scope('decoding'):
            prediction = tf.tensordot(self.user_latent, tf.transpose(self.item_query), axes=[[2], [0]])
            self.prediction = tf.reduce_max(tf.transpose(prediction, perm=[0, 2, 1]), axis=2)

        with tf.variable_scope('loss'):
            l2_loss = tf.nn.l2_loss(item_key_weights) \
                      + tf.nn.l2_loss(user_key_weights) \
                      + tf.nn.l2_loss(item_value_weights) \
                      + tf.nn.l2_loss(item_query_weights)
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=self.prediction)
            self.loss = tf.reduce_mean(sigmoid_loss) + self.lamb*tf.reduce_mean(l2_loss)

        with tf.variable_scope('optimizer'):
            self.train = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def get_batches(self, rating_matrix, batch_size):

        remaining_size = rating_matrix.shape[0]
        index = np.arange(remaining_size)

        rating_matrix = sparse.hstack([rating_matrix, index.reshape(remaining_size, 1)]).tocsr()

        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_matrix[batch_index * batch_size:])
            else:
                batches.append(rating_matrix[batch_index * batch_size:(batch_index + 1) * batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def train_model(self, rating_matrix, epoch=100, batches=None, **unused):
        if batches is None:
            batches = self.get_batches(rating_matrix, self.batch_size)

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense()}
                training = self.sess.run([self.train], feed_dict=feed_dict)

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self.batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.inputs: batches[step].todense()}
            encoded = self.sess.run(self.user_latent, feed_dict=feed_dict)
            RQ.append(encoded)

        return np.vstack(RQ)

    def get_Y(self):
        return self.sess.run(self.item_query)


def get_pmi_matrix(matrix, root):

    rows, cols = matrix.shape
    item_rated = matrix.sum(axis=0)
    # user_rated = matrix.sum(axis=1)
    nnz = matrix.nnz
    pmi_matrix = []
    for i in tqdm(xrange(rows)):
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


def mmp(matrix_train, embeded_matrix=np.empty((0)),
             iteration=4, lamb=100, rank=200, fb=False, seed=1, root=1.1, **unused):
    """
    PureSVD algorithm
    :param matrix_train: rating matrix
    :param embeded_matrix: item or user embedding matrix(side info)
    :param iteration: number of random SVD iterations
    :param rank: SVD top K eigenvalue ranks
    :param fb: facebook package or sklearn package. boolean
    :param seed: Random initialization seed
    :param unused: args that not applicable for this algorithm
    :return:
    """
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    progress.subsection("Create PMI matrix")
    pmi_matrix = get_pmi_matrix(matrix_input, root)

    progress.subsection("Randomized SVD")
    start_time = time.time()
    if fb:
        P, sigma, Qt = pca(pmi_matrix,
                           k=rank,
                           n_iter=7,
                           raw=True)
    else:
        P, sigma, Qt = randomized_svd(pmi_matrix,
                                      n_components=rank,
                                      n_iter=7,
                                      power_iteration_normalizer='QR',
                                      random_state=seed)

    Q = Qt.T

    model = MultiModesPreferenceEstimation(matrix_train.shape[1], rank, 5, 10, 100, lamb,
                                           user_embeddings=P, item_embeddings=Q)
    model.train_model(matrix_train, iteration)

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    RQ = model.get_RQ(matrix_input)
    Y = model.get_Y()
    model.sess.close()
    tf.reset_default_graph()

    return RQ, Y.T, None