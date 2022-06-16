"""Test tree sampling"""
# Copyright - Tarepan
# Licenced from Tarepan to OSS community by BSD3

import tensorflow as tf

from tree_sampling import _interleave, _tree_to_pdf


def test_interleave():
    # batch_size = 2
    t_sample = 3
    bit_depth = 3
    _pcm_levels = 2**bit_depth

    # P(bit_k|bit_<k)
    bit_cond_probs: tf.Tensor = tf.constant([
        [ # batch_0
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,], # t_0
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,], # t_1
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,], # t_2
        #   | - | L1 |    L2   |        L3         |
        ],
        [ # batch_1
            [1.0, 1.0, 1.0, 1.0, 0.1, 0.2, 0.3, 0.4,], # t_0
            [1.0, 1.0, 1.0, 1.0, 0.5, 0.6, 0.7, 0.8,], # t_1
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,], # t_2
        #   | - | L1 |    L2   |        L3         |
        ],
    ], dtype=tf.float64)

    # (B=2, T_s=3, Prob=4)
    layer = 3
    assert 2**(layer-1) == 4, "Prerequisites"
    assert 2**layer == 8, "Prerequisites"
    l3: tf.Tensor = bit_cond_probs[:, :, 2**(layer-1) : 2**layer]

    l3_gt: tf.Tensor = tf.constant([
        [ # batch_0
            [1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0,], # t_0
            [1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0,], # t_1
            [1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0,], # t_2
        #   |   L      H      L      H      L      H      L      H  |
        ],
        [ # batch_1
            [1.0-0.1, 0.1, 1.0-0.2, 0.2, 1.0-0.3, 0.3, 1.0-0.4, 0.4,], # t_0
            [1.0-0.5, 0.5, 1.0-0.6, 0.6, 1.0-0.7, 0.7, 1.0-0.8, 0.8,], # t_1
            [1.0-0.0, 0.0, 1.0-0.0, 0.0, 1.0-0.0, 0.0, 1.0-0.0, 0.0,], # t_2
        #   |   L      H      L      H      L      H      L      H  |
        ],
    ], dtype=tf.float64)

    l3_cond_probs = _interleave(l3, t_sample, _pcm_levels)

    assert True == tf.reduce_all(tf.equal(l3_cond_probs, l3_gt)).numpy().item()
