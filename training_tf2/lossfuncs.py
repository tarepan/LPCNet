"""
Custom Loss functions and metrics for training/analysis
"""

from tf_funcs import *
import tensorflow as tf

# The following loss functions all expect the lpcnet model to output the lpc prediction

# Computing the excitation by subtracting the lpc prediction from the target, followed by minimizing the cross entropy
def res_from_sigloss():
    def loss(y_true,y_pred):
        p = y_pred[:,:,0:1]
        model_out = y_pred[:,:,1:]
        e_gt = tf_l2u(y_true - p)
        e_gt = tf.round(e_gt)
        e_gt = tf.cast(e_gt,'int32')
        sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,model_out)
        return sparse_cel
    return loss

# Interpolated and Compensated Loss (In case of end to end lpcnet)
# Interpolates between adjacent embeddings based on the fractional value of the excitation computed (similar to the embedding interpolation)
# Also adds a probability compensation (to account for matching cross entropy in the linear domain), weighted by gamma
def interp_mulaw(gamma = 1):
    def loss(y_true,y_pred):
        y_true = tf.cast(y_true, 'float32')
        p = y_pred[:,:,0:1]
        model_out = y_pred[:,:,1:]
        e_gt = tf_l2u(y_true - p)
        prob_compensation = tf.squeeze((K.abs(e_gt - 128)/128.0)*K.log(256.0))
        alpha = e_gt - tf.math.floor(e_gt)
        alpha = tf.tile(alpha,[1,1,256])
        e_gt = tf.cast(e_gt,'int32')
        e_gt = tf.clip_by_value(e_gt,0,254) 
        interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
        sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
        loss_mod = sparse_cel + gamma*prob_compensation
        return loss_mod
    return loss

# Same as above, except a metric
def metric_oginterploss(y_true,y_pred):
    p = y_pred[:,:,0:1]
    model_out = y_pred[:,:,1:]
    e_gt = tf_l2u(y_true - p)
    prob_compensation = tf.squeeze((K.abs(e_gt - 128)/128.0)*K.log(256.0))
    alpha = e_gt - tf.math.floor(e_gt)
    alpha = tf.tile(alpha,[1,1,256])
    e_gt = tf.cast(e_gt,'int32')
    e_gt = tf.clip_by_value(e_gt,0,254) 
    interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
    loss_mod = sparse_cel + prob_compensation
    return loss_mod

# Interpolated cross entropy loss metric
def metric_icel(y_true, y_pred):
    p = y_pred[:,:,0:1]
    model_out = y_pred[:,:,1:]
    e_gt = tf_l2u(y_true - p)
    alpha = e_gt - tf.math.floor(e_gt)
    alpha = tf.tile(alpha,[1,1,256])
    e_gt = tf.cast(e_gt,'int32')
    e_gt = tf.clip_by_value(e_gt,0,254) #Check direction
    interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
    return sparse_cel


def metric_cel(y_t_true, y_t_pred):
    """
    Distance between e_t_ideal one-hot and e_t_estim dist, CrossEntropy loss.
    Non-interpolated (rounded)

    Args:
        y_t_true ::(B, T_s, 1)      - Ground truth s_t_clean series
        y_t_pred ::(B, T_s, 1+2**Q) - LP noisy series and residual Dist series
    """
    # Sample series, clean :: (B, T_s, 1)
    s_t_clean_series = tf.cast(y_t_true, "float32")
    # Prediction from noisy samples :: (B, T_s, 1)
    p_t_noisy_series = y_t_pred[:, :, 0:1]
    # Probability Distribution series of estimated residual :: (B, T_s, PD=2**Q)
    e_t_series_estim_dist = y_t_pred[:, :, 1:]

    # μ-law value series of idead residual under noisy AR :: (B, T_s, 1)
    e_t_series_ideal = tf_l2u(s_t_clean_series - p_t_noisy_series)
    e_t_series_ideal = tf.round(e_t_series_ideal)
    e_t_series_ideal = tf.cast(e_t_series_ideal, 'int32')
    e_t_series_ideal = tf.clip_by_value(e_t_series_ideal, 0, 255)

    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_t_series_ideal, e_t_series_estim_dist)
    return sparse_cel


# Variance metric of the output excitation
def metric_exc_sd(y_true,y_pred):
    p = y_pred[:,:,0:1]
    e_gt = tf_l2u(y_true - p)
    sd_egt = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(e_gt,128)
    return sd_egt

def loss_matchlar():
    def loss(y_true,y_pred):
        model_rc = y_pred[:,:,:16]
        #y_true = lpc2rc(y_true)
        loss_lar_diff = K.log((1.01 + model_rc)/(1.01 - model_rc)) - K.log((1.01 + y_true)/(1.01 - y_true))
        loss_lar_diff = tf.square(loss_lar_diff)
        return tf.reduce_mean(loss_lar_diff, axis=-1)
    return loss

