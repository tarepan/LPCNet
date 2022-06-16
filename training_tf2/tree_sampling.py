"""Tree sampling"""

import tensorflow as tf


# Hard-coded constants
_pcm_bits = 8              # u-law depth
_pcm_levels = 2**_pcm_bits # Dynamic range size of u-law


def _interleave(p, samples: int):
    """
    Args:
        p - Probabilities of Level L
    """
    # todo: Remove hardcoded values after assertion test
    # assert p.shape[1] == samples, f"p.shape[1] {p.shape[1]} != samples {samples}"
    # t_s = p.shape[1]

    # (B, T_s, Prob) -> (B, T_s, Prob, 1)
    p2=tf.expand_dims(p, 3)
    # 2**(Q-L) =        2**Q // 2**L
    nb_repeats = _pcm_levels // (2 * p.shape[2])
    # (B, T_s, Prob=2**(L-1), 1) -> (B, T_s, Prob=2**(L-1), LH=2)
    low_high = tf.concat([1-p2, p2], 3)
    # (B, T_s, Prob=2**(L-1), LH=2) -> (B * T_s * (2**L) * (2**(Q-L)) ,) == (B * T_s * (2**Q),)
    repeated_flatten = tf.repeat(low_high, nb_repeats)
    # (B * T_s * (2**Q),) -> (B, T_s, Prob=2**Q)
    p3 = tf.reshape(repeated_flatten, (-1, samples, _pcm_levels))
    return p3

def _tree_to_pdf(p, samples):
    """
       P(high)                  P(low)/P(high)              P(level)
    L1        0.6                     0.4/0.6               0.4  0.6  0.4  0.6
            /     \                                          x    x    x    x
    L2    0.1     0.7      =>    0.9/0.1   0.3/0.7    =>    0.9  0.1  0.3  0.7
         .   .   .   .                             
        .36/.04/.18/.42                                     0.36 0.04 0.18 0.42

    Args:
        p::(B, T_s, Prob) - Probabilities
    """
    # todo: Remove `_pcm_levels` in `_interleave` after assertion check
    assert p.shape[2] == _pcm_levels, f"p.shape[2] {p.shape[2]} == _pcm_levels {_pcm_levels}"
    ulaw_level = p.shape[2]

    #                    L1=2**0                              L2=2**1                              L3=2**2                               L4=2**3
    return _interleave(p[:,:, 1: 2], samples) * _interleave(p[:,:, 2: 4], samples) * _interleave(p[:,:, 4:  8], samples) * _interleave(p[:,:,  8: 16], samples) \
         * _interleave(p[:,:,16:32], samples) * _interleave(p[:,:,32:64], samples) * _interleave(p[:,:,64:128], samples) * _interleave(p[:,:,128:256], samples)
    #                    L5=2**4                              L6=2**5                              L7=2**6                               L8=2**7

    # todo: test refactored
    # probs = _interleave(p[:,:, 2**(1-1): 2**(1)], samples)
    # for l in range(2, 9):
    #     probs = probs * _interleave(p[:,:, 2**(l-1) : 2**l], samples)
    # return probs

def tree_to_pdf_train(p):
    """
    """
    #FIXME: try not to hardcode the 2400 samples (15 frames * 160 samples/frame)
    return _tree_to_pdf(p, 2400)

def tree_to_pdf_infer(p):
    """
    """
    return _tree_to_pdf(p, 1)
