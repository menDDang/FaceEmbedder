import tensorflow as tf
import numpy as np


def similarity(embedded, w, b, N=config.N, M=config.M, P=config.proj, center=None):
    """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
        Input center to test enrollment. (embedded for verification)
    :return: tf similarity matrix (NM x N)
    """
    embedded_split = tf.reshape(embedded, shape=[N, M, P])

    if center is None:
        center = normalize(tf.reduce_mean(embedded_split, axis=1))              # [N,P] normalized center vectors eq.(1)
        center_except = normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keep_dims=True)
                                             - embedded_split, shape=[N*M,P]))  # [NM,P] center vectors eq.(8)
        # make similarity matrix eq.(9)
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], axis=1, keep_dims=True) if i==j
                        else tf.reduce_sum(center[i:(i+1),:]*embedded_split[j,:,:], axis=1, keep_dims=True) for i in range(N)],
                       axis=1) for j in range(N)], axis=0)
    else :
        # If center(enrollment) exist, use it.
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keep_dims=True) for i
                        in range(N)],
                       axis=1) for j in range(N)], axis=0)

    S = tf.abs(w)*S+b   # rescaling

    return S


def similarity(embedded, center=None):
    if center is None:
        center = tf.reduct_mean(embedded, axis=1)