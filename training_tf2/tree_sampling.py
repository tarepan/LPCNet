"""Tree sampling"""

import tensorflow as tf


def _interleave(c_probs: tf.Tensor, len_t: int, num_bits: int, idx_layer: int) -> tf.Tensor:
    """
    Expand conditional probabilities for joint probability distribution

    Args:
        c_probs :: (B, T_s, Cond=2**(k-1)) - B(L_k=1|L_<k=cond) conditional probabilities of all conditions of layer k
        len_t - Time length of `c_probs`
        num_bits - The number of sample bits
        idx_layer - Index number 'k' of the `c_probs` layer
    """

    # Conditional Prob.s to Conditional Prob. distributions
    #   (B, T_s, Cond) -> (B, T_s, Cond, 1) -> (B, T_s, Cond, Dist=2)
    c_probs = tf.expand_dims(c_probs, 3)
    c_prob_dists = tf.concat([1.0-c_probs, c_probs], 3)
    #                        [B(Lk=0|.)  B(Lk=1|.)]

    # Expand for joint distribution
    #   (B, T_s, Cond=2**(k-1), Dist=2) -> (B * T_s * (2**(k-1)) * 2 * (2**(Q-L)) ,) == (B * T_s * (2**Q),) -> (B, T_s, Dist=2**Q)
    # x2 conditions per bit layer
    nb_repeats = 2 ** (num_bits - idx_layer)
    repeated_flatten = tf.repeat(c_prob_dists, nb_repeats)
    c_prob_dists_expanded = tf.reshape(repeated_flatten, (-1, len_t, 2**num_bits))
    return c_prob_dists_expanded


def _tree_to_pdf(p: tf.Tensor, len_t: int, pcm_bits: int) -> tf.Tensor:
    """
    Convert Hierarchical conditional bit probabilities to the joint level probability.

         B(Lk=1|L<k)     B(Lk=0|L<k=Bs<k)/B(Lk=1|L<k=Bs<k)      P(level)
    L1        0.6                     0.4/0.6               0.4  0.6  0.4  0.6
            /     \                                          x    x    x    x
    L2    0.1     0.7      =>    0.9/0.1   0.3/0.7    =>    0.9  0.1  0.3  0.7
         .   .   .   .
    P(s).36/.04/.18/.42                                     0.36 0.04 0.18 0.42

    Args:
        p ::(B, T_s, Prob) - Hierarchical conditional bit probabilities
        len_t              - Time length of `p`
        pcm_bits           - The number of sample bits
    Returns :: (B, T_s, Prob) - Joint probability distribution of all bits == PD of level
    """
    # todo: Remove `_pcm_levels` in `_interleave` after assertion check
    # assert p.shape[2] == pcm_levels, f"p.shape[2] {p.shape[2]} == pcm_levels {pcm_levels}"
    # ulaw_level = p.shape[2]

    # todo: Remove hardcoded values after assertion test
    # assert p.shape[1] == samples, f"p.shape[1] {p.shape[1]} != samples {samples}"

    # # Time Length
    # len_t = c_probs.shape[1]

    # Joint Probability Distribution == Π P(L_k|L_<k)
    prob_dist = _interleave(p[:,:, 2**(1-1): 2**(1)], len_t, pcm_bits, 1)
    for l in range(2, pcm_bits + 1):
        prob_dist = prob_dist * _interleave(p[:,:, 2**(l-1) : 2**l], len_t, pcm_bits, l)
    return prob_dist


def gen_tree_to_pdf(len_t: int, pcm_bits: int):
    return lambda p: _tree_to_pdf(p, len_t, pcm_bits)
