import numpy as np
import tensorflow as tf
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = -tf.reduce_sum(labels*tf.log(tf.nn.softmax(preds)+1e-7), axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum+1e-6, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(sp.csr_matrix(features))


def markov(adj):
    """Preprocessing of adjacency matrix for Markov Matrix setting"""
    ad=adj+sp.eye(adj.shape[0])
    rowsum = np.array(ad.sum(0))
    return sparse_to_tuple(sp.csr_matrix(ad/rowsum))


def NI_ATT(X, in_sz, out_sz, adj_mat, activation, nb_nodes, att_drop=1.0):

    Z = dot(X, glorot([in_sz, out_sz]), sparse=True)
    Z_expanded = tf.expand_dims(Z, axis=0)

    e_i = tf.layers.conv1d(Z_expanded, 1, 1)
    e_j = tf.layers.conv1d(Z_expanded, 1, 1)

    e_i = tf.reshape(e_i, (nb_nodes, 1))
    e_j = tf.reshape(e_j, (nb_nodes, 1))
    e_j = tf.transpose(e_j, [1,0])

    alpha = e_i * e_j
    alpha *= adj_mat
    ATT = tf.sparse_softmax(alpha)

    if att_drop != 1.0:
        ATT = tf.SparseTensor(indices=ATT.indices,
                                values=tf.nn.dropout(ATT.values, att_drop),
                                dense_shape=ATT.dense_shape)

    ATT = tf.sparse_reshape(ATT, [nb_nodes, nb_nodes])
    vals = tf.sparse_tensor_dense_matmul(ATT,  Z)
    ret = vals + zeros([out_sz])
    ret = tf.add(ret, Z)  # residual connection
    return activation(ret), Z
