import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from scipy import sparse
from tensorflow.contrib.distributions import RelaxedOneHotCategorical
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
from tqdm import tqdm



def set_rng_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

def predict(matrix_train, kfac, dfac, lam, lr, seed, tau, std, nogb, topk, batch_size, log_dir):
    set_rng_seed(seed)

    n_test = matrix_train.shape[0]
    n_items = matrix_train.shape[1]
    idxlist_test = list(range(n_test))

    tf.reset_default_graph()
    vae = DVAE(n_items, kfac, dfac, lam, lr, seed, tau, std, nogb)
    saver, logits_var, _, _ = vae.build_graph()
    prediction = []
    with tf.Session() as sess:
        saver.restore(sess, '{}/chkpt'.format(log_dir))
        for bnum, st_idx in tqdm(enumerate(range(0, n_test, batch_size))):
            end_idx = min(st_idx + batch_size, n_test)
            x = matrix_train[idxlist_test[st_idx:end_idx]]
            if sparse.isspmatrix(x):
                x = x.toarray()
            x = x.astype('float32')
            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: x})
            pred_val[x.nonzero()] = -np.inf
            batch_users = pred_val.shape[0]
            idx_topk_part = np.argpartition(-pred_val, topk, axis=1)
            topk_part = pred_val[np.arange(batch_users)[:, np.newaxis],
                               idx_topk_part[:, :topk]]
            idx_part = np.argsort(-topk_part, axis=1)
            # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
            # topk predicted score
            idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
            prediction.append(idx_topk)
    return np.vstack(prediction)

class DVAE(object):
    def __init__(self, num_items, kfac, dfac, lam, lr, seed, tau, std, nogb):
        kfac, dfac = kfac, dfac
        self.lam = lam
        self.lr = lr
        self.random_seed = seed
        self.tau = tau
        self.std = std
        self.nogb = nogb
        self.n_items = num_items
        self.kfac = kfac
        self.dfac = dfac

        # The first fc layer of the encoder Q is the context embedding table.
        self.q_dims = [num_items, dfac, dfac]
        self.weights_q, self.biases_q = [], []
        for i, (d_in, d_out) in enumerate(
                zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out *= 2  # mu & var
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            bias_key = "bias_q_{}".format(i + 1)
            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

        self.items = tf.get_variable(
            name="items", shape=[num_items, dfac],
            initializer=tf.contrib.layers.xavier_initializer(
                seed=self.random_seed))

        self.cores = tf.get_variable(
            name="cores", shape=[kfac, dfac],
            initializer=tf.contrib.layers.xavier_initializer(
                seed=self.random_seed))

        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, num_items])
        self.keep_prob_ph = tf.placeholder_with_default(1., shape=None)
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)

    def build_graph(self, save_emb=False):
        if save_emb:
            saver, facets_list = self.forward_pass(save_emb=True)
            return saver, facets_list, self.items, self.cores

        self.saver, self.logits, recon_loss, kl = self.forward_pass(save_emb=False)

        reg_var = apply_regularization(
            l2_regularizer(self.lam),
            self.weights_q + [self.items, self.cores])
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_elbo = recon_loss + self.anneal_ph * kl + 2. * reg_var

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_elbo)

        # add summary statistics
        tf.summary.scalar('trn/neg_ll', recon_loss)
        tf.summary.scalar('trn/kl_div', kl)
        tf.summary.scalar('trn/neg_elbo', neg_elbo)
        self.merged = tf.summary.merge_all()

        return self.saver, self.logits, self.train_op, self.merged

    def q_graph_k(self, x):
        mu_q, std_q, kl = None, None, None
        h = tf.nn.l2_normalize(x, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w, a_is_sparse=(i == 0)) + b
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                mu_q = tf.nn.l2_normalize(mu_q, axis=1)
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std0 = self.std
                std_q = tf.exp(0.5 * lnvarq_sub_lnvar0) * std0
                # Trick: KL is constant w.r.t. to mu_q after we normalize mu_q.
                kl = tf.reduce_mean(tf.reduce_sum(
                    0.5 * (-lnvarq_sub_lnvar0 + tf.exp(lnvarq_sub_lnvar0) - 1.),
                    axis=1))
        return mu_q, std_q, kl

    def train_model(self, matrix_train, beta, keep, epoch, batch_size, log_dir, seed):
        set_rng_seed(seed)
        n = matrix_train.shape[0]
        n_items = matrix_train.shape[1]
        idxlist = list(range(n))
        num_batches = int(np.ceil(float(n) / batch_size))
        total_anneal_steps = 5 * num_batches
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            best_ndcg = -np.inf
            update_count = 0.0
            for i in tqdm(range(epoch)):
                np.random.shuffle(idxlist)
                for bnum, st_idx in enumerate(range(0, n, batch_size)):
                    end_idx = min(st_idx + batch_size, n)
                    x = matrix_train[idxlist[st_idx:end_idx]]
                    if sparse.isspmatrix(x):
                        x = x.toarray()
                    x = x.astype('float32')
                    if total_anneal_steps > 0:
                        anneal = min(beta,
                                     1. * update_count / total_anneal_steps)
                    else:
                        anneal = beta
                    feed_dict = {self.input_ph: x,
                                 self.keep_prob_ph: keep,
                                 self.anneal_ph: anneal,
                                 self.is_training_ph: 1}
                    sess.run(self.train_op, feed_dict=feed_dict)
                    update_count += 1
            self.saver.save(sess, '{}/chkpt'.format(log_dir))

    def forward_pass(self, save_emb):
        # clustering
        cores = tf.nn.l2_normalize(self.cores, axis=1)
        items = tf.nn.l2_normalize(self.items, axis=1)
        cates_logits = tf.matmul(items, cores, transpose_b=True) / self.tau
        if self.nogb:
            cates = tf.nn.softmax(cates_logits, axis=1)
        else:
            cates_dist = RelaxedOneHotCategorical(1, cates_logits)
            cates_sample = cates_dist.sample()
            cates_mode = tf.nn.softmax(cates_logits, axis=1)
            cates = (self.is_training_ph * cates_sample +
                     (1 - self.is_training_ph) * cates_mode)

        z_list = []
        probs, kl = None, None
        for k in range(self.kfac):
            cates_k = tf.reshape(cates[:, k], (1, -1))

            # q-network
            x_k = self.input_ph * cates_k
            mu_k, std_k, kl_k = self.q_graph_k(x_k)
            epsilon = tf.random_normal(tf.shape(std_k))
            z_k = mu_k + self.is_training_ph * epsilon * std_k
            kl = (kl_k if (kl is None) else (kl + kl_k))
            if save_emb:
                z_list.append(z_k)

            # p-network
            z_k = tf.nn.l2_normalize(z_k, axis=1)
            logits_k = tf.matmul(z_k, items, transpose_b=True) / self.tau
            probs_k = tf.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = (probs_k if (probs is None) else (probs + probs_k))

        logits = tf.log(probs)
        logits = tf.nn.log_softmax(logits)
        recon_loss = tf.reduce_mean(tf.reduce_sum(
            -logits * self.input_ph, axis=-1))

        if save_emb:
            return tf.train.Saver(), z_list
        return tf.train.Saver(), logits, recon_loss, kl


def dvae(log_dir, matrix_train, num_items, kfac, dfac, lam, lr, seed, tau, std, nogb, beta, keep, topk, epoch, batch_size):

    tf.reset_default_graph()
    vae = DVAE(num_items, kfac, dfac, lam, lr, seed, tau, std, nogb)
    vae.build_graph()
    vae.train_model(matrix_train, beta, keep, epoch, batch_size, log_dir, seed)
    prediction = predict(matrix_train, kfac, dfac, lam, lr, seed, tau, std, nogb, topk, batch_size, log_dir)
    print(prediction.shape)