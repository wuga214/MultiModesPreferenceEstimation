from fbpca import pca
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import inv
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from utils.optimizers import Optimizer

import numpy as np
import scipy.sparse as sparse
import time
import tensorflow as tf


class MultiModesPreferenceEstimation(object):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 mode_dim,
                 key_dim,
                 batch_size,
                 alpha,
                 lamb=0.01,
                 learning_rate=1e-3,
                 optimizer=Optimizer['Adam'],
                 item_embeddings=None,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.embed_dim = embed_dim
        self.mode_dim = mode_dim
        self.key_dim = key_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.item_embeddings = item_embeddings
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter("/tmp/histogram_example")

    def get_graph(self):
        self.corruption = tf.placeholder(tf.float32)
        self.inputs = tf.placeholder(tf.float32, (None, self.input_dim), name="inputs")
        inputs = tf.floor(tf.nn.dropout(self.inputs, 1-self.corruption))

        item_embeddings = tf.constant(self.item_embeddings, name="item_embeddings")

        with tf.variable_scope('keys'):
            item_key_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.key_dim], stddev=1 / 500.0),
                                           name="item_key_weights")

            tf.summary.histogram("item_key_weights", item_key_weights)

            item_keys = tf.nn.relu(tf.matmul(item_embeddings, item_key_weights))

            tf.summary.histogram("item_keys", item_keys)

            item_value_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.embed_dim], stddev=1 / 500.0),
                                             name="item_value_weights")

            item_values = tf.matmul(item_embeddings, item_value_weights)

            # item_value_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.embed_dim, self.mode_dim], stddev=1 / 500.0),
            #                                  name="item_value_weights")
            #
            # item_values = tf.tensordot(item_embeddings, item_value_weights, axes=[[1], [0]])

            tf.summary.histogram("item_values", item_values)

            self.item_query = tf.Variable(tf.truncated_normal([self.input_dim, self.embed_dim], stddev=1 / 500.0),
                                             name="item_query_weights")

            tf.summary.histogram("item_query", self.item_query)

            user_keys = tf.nn.relu(tf.Variable(tf.truncated_normal([self.key_dim, self.mode_dim],
                                                                   stddev=1 / 500.0),
                                               name="user_key_weights"))

            tf.summary.histogram("user_keys", user_keys)

        with tf.variable_scope('encoding'):

            encode_bias = tf.Variable(tf.constant(0., shape=[self.mode_dim, self.embed_dim]), name="Bias")

            attention = tf.tensordot(tf.multiply(tf.expand_dims(inputs, -1), item_keys), user_keys, axes=[[2], [0]])
            attention = tf.exp(tf.transpose(attention/tf.sqrt(tf.cast(self.key_dim, dtype=tf.float32)), perm=[0, 2, 1]))
            attention = tf.multiply(attention/tf.expand_dims(tf.reduce_sum(attention, axis=2), -1), tf.reduce_sum(inputs))

            # attention = attention/tf.sqrt(tf.cast(self.key_dim, dtype=tf.float32))
            # attention = tf.nn.softmax(tf.transpose(attention, perm=[0, 2, 1]), axis=2)

            tf.summary.histogram("attention", attention)

            # self.user_latent = tf.nn.leaky_relu(tf.reduce_sum(tf.multiply(tf.expand_dims(attention, -1), tf.transpose(item_values,perm=[2,0,1])), axis=2) + encode_bias)

            self.user_latent = tf.nn.leaky_relu(tf.tensordot(attention, item_values, axes=[[2], [0]]) + encode_bias)

        with tf.variable_scope('decoding'):

            prediction = tf.tensordot(self.user_latent, tf.transpose(self.item_query), axes=[[2], [0]])
            # prediction = tf.transpose(prediction, perm=[0, 2, 1])
            # attention = tf.nn.softmax(prediction, axis=2)
            # self.prediction = tf.reduce_sum(tf.multiply(prediction,attention), axis=2)

            self.prediction = tf.reduce_max(tf.transpose(prediction, perm=[0, 2, 1]), axis=2)

        with tf.variable_scope('loss'):
            l2_loss = tf.nn.l2_loss(self.item_query)/(tf.cast(self.input_dim*self.embed_dim, dtype=tf.float32))

            loss_weights = 1+self.alpha*tf.log(1+self.inputs) #self.inputs + 0.4*(1-self.inputs)
            sigmoid_loss = tf.losses.mean_squared_error(labels=self.inputs, predictions=self.prediction, weights=loss_weights)
            self.loss = tf.reduce_mean(sigmoid_loss) + self.lamb*l2_loss

            tf.summary.histogram("label", self.inputs)

            tf.summary.histogram("prediction", self.prediction)

            tf.summary.scalar("loss", self.loss)

        self.summaries = tf.summary.merge_all()

        with tf.variable_scope('optimizer'):
            self.train = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def get_batches(self, rating_matrix, batch_size):

        remaining_size = rating_matrix.shape[0]

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

    def train_model(self, rating_matrix, corruption, epoch=100, batches=None, **unused):
        if batches is None:
            batches = self.get_batches(rating_matrix, self.batch_size)

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense(), self.corruption: corruption}
                training = self.sess.run([self.train], feed_dict=feed_dict)

            feed_dict = {self.inputs: batches[0].todense(), self.corruption: corruption}
            summ = self.sess.run(self.summaries, feed_dict=feed_dict)
            self.writer.add_summary(summ, global_step=i)

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self.batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.inputs: batches[step].todense(), self.corruption: 0.}
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


def mmp(matrix_train, embedded_matrix=np.empty((0)), mode_dim=5, key_dim=3, batch_size=32, optimizer="Adam",
        learning_rate=0.001, iteration=4, epoch=20, lamb=100, rank=200, corruption=0.5, fb=False, seed=1, root=1,
        alpha=1, **unused):
    """
    PureSVD algorithm
    :param matrix_train: rating matrix
    :param embedded_matrix: item or user embedding matrix(side info)
    :param iteration: number of random SVD iterations
    :param rank: SVD top K eigenvalue ranks
    :param fb: facebook package or sklearn package. boolean
    :param seed: Random initialization seed
    :param unused: args that not applicable for this algorithm
    :return:
    """
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embedded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embedded_matrix.T))

    progress.subsection("Create PMI matrix")
    pmi_matrix = get_pmi_matrix(matrix_input, root)

    progress.subsection("Randomized SVD")
    start_time = time.time()
    if fb:
        P, sigma, Qt = pca(pmi_matrix,
                           k=rank,
                           n_iter=iteration,
                           raw=True)
    else:
        P, sigma, Qt = randomized_svd(pmi_matrix,
                                      n_components=rank,
                                      n_iter=iteration,
                                      power_iteration_normalizer='QR',
                                      random_state=seed)

    Q = Qt.T #*np.sqrt(sigma)

    Q = (Q - np.mean(Q)) / np.std(Q)

    model = MultiModesPreferenceEstimation(matrix_train.shape[1], rank, mode_dim, key_dim, batch_size,
                                           alpha=alpha,
                                           lamb=lamb,
                                           learning_rate=learning_rate,
                                           optimizer=Optimizer[optimizer],
                                           item_embeddings=Q)

    model.train_model(matrix_train, corruption, epoch)

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    RQ = model.get_RQ(matrix_input)
    Y = model.get_Y()
    #Bias = model.get_Bias()
    model.sess.close()
    tf.reset_default_graph()

    return RQ, Y.T, None
