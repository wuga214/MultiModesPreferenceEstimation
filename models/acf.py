import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, lil_matrix
from utils.optimizers import Optimizer
from _collections import defaultdict
from utils.functions import get_pmi_matrix
from sklearn.utils.extmath import randomized_svd
from fbpca import pca
import time
from datetime import datetime

class ACF(object):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_dim,
                 key_dim,
                 batch_size,
                 lamb=0.01,
                 learning_rate=1e-3,
                 optimizer=Optimizer['Adam'],
                 item_embeddings=None,
                 uniform_sample=False,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.key_dim = key_dim
        self.item_features = item_embeddings
        self.batch_size = batch_size
        self.uniform = uniform_sample
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # self.writer = tf.summary.FileWriter("/tmp/acf/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/")
        # print("/tmp/acf/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/")

    def get_graph(self):

        # Placehoder
        self.user_idx = tf.placeholder(tf.int32, [None])
        self.item_idx_i = tf.placeholder(tf.int32, [None])
        self.item_idx_j = tf.placeholder(tf.int32, [None])
        self.user_trans = tf.placeholder(tf.float32, [None, self.num_items])
        self.label = tf.placeholder(tf.float32, [None])

        # Variable to learn
        self.user_embeddings = tf.Variable(tf.random_normal([self.num_users, self.embed_dim],
                                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        self.item_embeddings = tf.Variable(tf.random_normal([self.num_items, self.embed_dim],
                                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        # item feature
        item_features = tf.constant(self.item_features, name="item_embeddings")

        # select user and item embedding
        users = tf.nn.embedding_lookup(self.user_embeddings, self.user_idx, name="users")
        item_i = tf.nn.embedding_lookup(self.item_embeddings, self.item_idx_i, name="item_i")
        item_j = tf.nn.embedding_lookup(self.item_embeddings, self.item_idx_j, name="item_j")

        with tf.variable_scope("attention"):
            # wired init
            item_attention_weights = tf.Variable(tf.random.uniform([self.embed_dim, self.key_dim],
                                          minval=-np.sqrt(6. / (self.embed_dim + self.key_dim)),
                                          maxval=np.sqrt(6. / (self.embed_dim + self.key_dim)),
                                          name="item_attention_weights",
                                          dtype=tf.float32))
            #tf.summary.histogram("item_attention_weights", item_attention_weights)
            user_attention_weights = tf.Variable(tf.random.uniform([self.embed_dim, self.key_dim],
                                          minval=-np.sqrt(6. / (self.embed_dim + self.key_dim)),
                                          maxval=np.sqrt(6. / (self.embed_dim + self.key_dim)),
                                          name="user_attention_weights",
                                          dtype=tf.float32))
            #tf.summary.histogram("user_attention_weights", user_attention_weights)
            item_feature_attention_weights = tf.Variable(tf.random.uniform([self.embed_dim, self.key_dim],
                                                                   minval=-np.sqrt(
                                                                       6. / (self.embed_dim + self.key_dim)),
                                                                   maxval=np.sqrt(
                                                                       6. / (self.embed_dim + self.key_dim)),
                                                                   name="item_feature_attention_weights",
                                                                   dtype=tf.float32))
            #tf.summary.histogram("item_feature_attention_weights", item_feature_attention_weights)

            b = tf.Variable(tf.constant(0., shape=[self.key_dim]), name="bias_1")

            attend_sec_weights = tf.Variable(tf.random.uniform([self.key_dim, 1],
                                          minval=-np.sqrt(6. / (self.key_dim + 1)),
                                          maxval=np.sqrt(6. / (self.key_dim + 1)),
                                          name="attend_sec_weights",
                                          dtype=tf.float32))
            #tf.summary.histogram("attend_sec_weights", attend_sec_weights)

            c = tf.Variable(tf.constant(0., shape=[1]), name="bias_2")

            user_key = tf.matmul(users, user_attention_weights)
            user_key = tf.expand_dims(user_key, 1)
            item_key = tf.matmul(self.item_embeddings, item_attention_weights)
            item_key = tf.expand_dims(item_key, 0)
            item_feature_key = tf.matmul(item_features, item_feature_attention_weights)
            item_feature_key = tf.expand_dims(item_feature_key, 0)
            atten = tf.nn.relu(user_key + item_key + item_feature_key + b)
            #atten = tf.squeeze(tf.matmul(atten, attend_sec_weights) + c)
            atten = tf.matmul(atten, attend_sec_weights) + c
            #atten = tf.reshape(atten, [-1, 3533])
            atten = tf.multiply(tf.exp(atten), tf.expand_dims(self.user_trans, -1)) + 0.0001
            self.atten = atten/tf.expand_dims(tf.reduce_sum(atten, axis=1), -1)

            #tf.summary.histogram("atten", self.atten)
            users = users + tf.reduce_sum(self.atten * item_features, axis=1)


        with tf.variable_scope("bpr_loss"):
            x_uij = tf.reduce_sum(tf.multiply(users,
                                              item_i,
                                              name='x_ui'),
                                  axis=1) - tf.reduce_sum(tf.multiply(users,
                                                                      item_j,
                                                                      name='x_uj'),
                                                          axis=1)

            if self.uniform:
                bpr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_uij, labels=self.label))
            else:
                bpr_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(x_uij)))

        with tf.variable_scope('l2_loss'):
            unique_user_idx, _ = tf.unique(self.user_idx)
            unique_users = tf.nn.embedding_lookup(self.user_embeddings, unique_user_idx)

            unique_item_idx, _ = tf.unique(tf.concat([self.item_idx_i, self.item_idx_j], 0))
            unique_items = tf.nn.embedding_lookup(self.item_embeddings, unique_item_idx)

            l2_loss = tf.reduce_mean(tf.nn.l2_loss(unique_users)) + tf.reduce_mean(tf.nn.l2_loss(unique_items)) #+ tf.reduce_mean(tf.nn.l2_loss(item_attention_weights)) + tf.reduce_mean(tf.nn.l2_loss(user_attention_weights)) + tf.reduce_mean(tf.nn.l2_loss(item_feature_attention_weights))

        with tf.variable_scope('loss'):
            self.loss = bpr_loss + self.lamb * l2_loss
            #tf.summary.scalar("loss", self.loss)

        #self.summaries = tf.summary.merge_all()
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def get_uniform_batches(self, rating_matrix, batch_size):
        batches = []

        for i in tqdm(range(int(self.num_users / batch_size))):
            user_idx = np.random.choice(self.num_users, batch_size)
            item_idx_i = np.random.choice(self.num_items, batch_size)
            item_idx_j = np.random.choice(self.num_items, batch_size)

            label = np.max(np.asarray(rating_matrix[user_idx, item_idx_i] - rating_matrix[user_idx, item_idx_j]), 0)

            batches.append([user_idx, item_idx_i, item_idx_j, label])

        return batches

    @staticmethod
    def get_batches(user_item_pairs, rating_matrix, num_item, batch_size):
        batches = []
        index_shuf = list(range(len(user_item_pairs)))
        np.random.shuffle(index_shuf)
        user_item_pairs = user_item_pairs[index_shuf]
        for i in tqdm(range(int(len(user_item_pairs) / batch_size))):

            ui_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]

            negative_samples = np.random.randint(
                0,
                num_item,
                size=batch_size)

            label = np.max(np.asarray(rating_matrix[ui_pairs[:, 0], ui_pairs[:, 1]] - rating_matrix[ui_pairs[:, 0],
                                                                                                    negative_samples]),
                           0)
            u_trans = rating_matrix[ui_pairs[:, 0]]

            batches.append([ui_pairs[:, 0], ui_pairs[:, 1], negative_samples, label, u_trans])
            #batches.append([ui_pairs[:, 0], ui_pairs[:, 1], negative_samples, label])

        return batches


    def train_model(self, rating_matrix, epoch=100):
        print(epoch)
        if not self.uniform:
            user_item_matrix = lil_matrix(rating_matrix)
            user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

        # Training
        for i in tqdm(range(epoch)):
            if self.uniform:
                batches = self.get_uniform_batches(rating_matrix, self.batch_size)
            else:
                batches = self.get_batches(user_item_pairs, rating_matrix, self.num_items, self.batch_size)
            for step in range(len(batches)):
                feed_dict = {self.user_idx: batches[step][0],
                             self.item_idx_i: batches[step][1],
                             self.item_idx_j: batches[step][2],
                             self.label: batches[step][3],
                             self.user_trans: batches[step][4].todense()
                             }
                # feed_dict = {self.user_idx: batches[step][0],
                #              self.item_idx_i: batches[step][1],
                #              self.item_idx_j: batches[step][2],
                #              self.label: batches[step][3]
                #              }
                training = self.sess.run([self.optimizer], feed_dict=feed_dict)
            feed_dict = {self.user_idx: batches[0][0],
                             self.item_idx_i: batches[0][1],
                             self.item_idx_j: batches[0][2],
                             self.label: batches[0][3],
                             self.user_trans: batches[0][4].todense()}
            # feed_dict = {self.user_idx: batches[0][0],
            #              self.item_idx_i: batches[0][1],
            #              self.item_idx_j: batches[0][2],
            #              self.label: batches[0][3]}

            # summ = self.sess.run(self.summaries, feed_dict=feed_dict)
            # self.writer.add_summary(summ, global_step=i)

    def get_RQ(self):
        return self.sess.run(self.user_embeddings)

    def get_Y(self):
        return self.sess.run(self.item_embeddings)


def acf(matrix_train, embeded_matrix=np.empty((0)), epoch=300,
        iteration=100, lamb=80, rank=100,key_dim=3,
        batch_size=32, optimizer="Adam", learning_rate=0.001,
        seed=1, root=1, fb=False, **unused):

    print(epoch, lamb, rank)
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
                           n_iter=iteration,
                           raw=True)
    else:
        P, sigma, Qt = randomized_svd(pmi_matrix,
                                      n_components=rank,
                                      n_iter=iteration,
                                      power_iteration_normalizer='QR',
                                      random_state=seed)
    Q = Qt.T*np.sqrt(sigma)
    m, n = matrix_input.shape
    model = ACF(m, n, rank, key_dim, lamb=lamb, batch_size=batch_size, learning_rate=learning_rate,
                optimizer=Optimizer[optimizer], item_embeddings=Q)
    model.train_model(matrix_input, epoch)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    RQ = model.get_RQ()
    Y = model.get_Y().T
    model.sess.close()
    tf.reset_default_graph()

    return RQ, Y, None