"""
Tensorflow/Keras helper functions to do the following:
    1. \mu law <-> Linear domain conversion
    2. Differentiable prediction from the input signal and LP coefficients
    3. Differentiable transformations Reflection Coefficients (RCs) <-> LP Coefficients
"""
from tensorflow.keras.layers import Lambda, Multiply, Layer, Concatenate
from tensorflow.keras import backend as K
import tensorflow as tf

# \mu law <-> Linear conversion functions
scale = 255.0/32768.0
scale_1 = 32768.0/255.0
def tf_l2u(x):
    """Linear to μ-law with TensorFlow functions"""
    s = K.sign(x)
    x = K.abs(x)
    u = (s*(128*K.log(1+scale*x)/K.log(256.0)))
    u = K.clip(128 + u, 0, 255)
    return u

def tf_u2l(u):
    """μ-law to linear with TensorFlow functions"""
    u = tf.cast(u,"float32")
    u = u - 128.0
    s = K.sign(u)
    u = K.abs(u)
    return s*scale_1*(K.exp(u/128.*K.log(256.0))-1)


class diff_pred(Layer):
    """Differentiable Linear Prediction, p_t = a1 * s_{t-1} + ..."""
    def call(self, inputs, lpcoeffs_N: int = 16, frame_size: int = 160):
        """
        Args:
            inputs
                s_t_1_series ::(B, T_s, 1)     - Sample series (waveform), 1 sample lagged/delayed
                coeff_series ::(B, T_f, Order) - Linear prediction coefficients
            lpcoeffs_N - The order of linear prediction (the number of coefficients) ?
        Returns::(B, L, 1) - p_t_series
        """
        s_t_1_series = inputs[0]
        coeff_series = inputs[1]
        LENGTH_T: int = 2400

        # Repeat LP coefficients :: (B, T_f, Order) -> (B, T_s==T_f*frame_size, Order)
        rept = Lambda(lambda x: K.repeat_elements(x , frame_size, 1))

        """
        [I/O]
            Input:  s_t_1_series    s0  s1 ...
            Output  p_t_series      p1  p2 ...

        [LP by Lagged series sum]
            s_t_16_series    0   0 ...   0   s0
            ...
            s_t_2_series     0  s0 ... s13  s14
            s_t_1_series    s0  s1 ... s14  s15
           -------------------------------------
            p_t_series      p1  p2 ... p15  p16
        """
        # Series of LP value pack (s_{t-1}, s_{t-2}, ..., s_{t-16})
        ## Zero padding :: (B, T_s=L, 1) -> (B, T_s=Order+L, 1), #0 ~ #{Order-1} is zero, #Order is s0
        zpX = Lambda(lambda x: K.concatenate([0*x[:, 0:lpcoeffs_N, :], x],axis = 1))
        ## sample series to value pack series
        #   (B, T_s=Order+L, 1) -> (B, T_{t-i}=L, 1) for (1, ..., Order) -> (B, T_s=L, Order)
        cX = Lambda((lambda x: K.concatenate(
            ### Prepare lagged series, [:, Order-0:..., :] is equal to `s_t_1_series`
            [x[:, (lpcoeffs_N - i) : (lpcoeffs_N - i + LENGTH_T), :] for i in range(lpcoeffs_N)],
            ### Concat as value pack
            axis = 2
        )))

        # Linear Prediction
        ## a_i*s_{t-i} for all t & i :: (B, T_s, Order) -> (B, T_s, Order)
        p_t_elem_series = -Multiply()([rept(coeff_series), cX(zpX(s_t_1_series))])
        # p_t = Σi=1 (a_i*s_{t-i}) :: (B, T_s, Order) -> (B, T_s, 1)
        p_t_series = K.sum(p_t_elem_series,axis = 2,keepdims = True)

        return p_t_series


# Differentiable Transformations (RC <-> LPC) computed using the Levinson Durbin Recursion 
class diff_rc2lpc(Layer):
    def call(self, inputs, lpcoeffs_N = 16):
        def pred_lpc_recursive(input):
            temp = (input[0] + K.repeat_elements(input[1],input[0].shape[2],2)*K.reverse(input[0],axes = 2))
            temp = Concatenate(axis = 2)([temp,input[1]])
            return temp
        Llpc = Lambda(pred_lpc_recursive)
        inputs = inputs[:,:,:lpcoeffs_N]
        lpc_init = inputs
        for i in range(1,lpcoeffs_N):
            lpc_init = Llpc([lpc_init[:,:,:i],K.expand_dims(inputs[:,:,i],axis = -1)])
        return lpc_init

class diff_lpc2rc(Layer):
    def call(self, inputs, lpcoeffs_N = 16):
        def pred_rc_recursive(input):
            ki = K.repeat_elements(K.expand_dims(input[1][:,:,0],axis = -1),input[0].shape[2],2)
            temp = (input[0] - ki*K.reverse(input[0],axes = 2))/(1 - ki*ki)
            temp = Concatenate(axis = 2)([temp,input[1]])
            return temp
        Lrc = Lambda(pred_rc_recursive)
        rc_init = inputs
        for i in range(1,lpcoeffs_N):
            j = (lpcoeffs_N - i + 1)
            rc_init = Lrc([rc_init[:,:,:(j - 1)],rc_init[:,:,(j - 1):]])
        return rc_init
