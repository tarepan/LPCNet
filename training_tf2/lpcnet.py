#!/usr/bin/python3
'''Copyright (c) 2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from typing import Tuple

import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, GaussianNoise
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.callbacks import Callback
from mdense import MDense
import numpy as np
from tf_funcs import *
from diffembed import diff_Embed
from tree_sampling import gen_tree_to_pdf


frame_size = 160         # Waveform samples per acoustic frame [samples/frame]
pcm_bits = 8             # u-law depth
embed_size = 128
pcm_levels = 2**pcm_bits # Dynamic range size of u-law


def quant_regularizer(x):
    """(maybe) Force weights to be integers for quantization in inference."""
    Q = 128
    #return .01 * tf.reduce_mean(1 - tf.math.cos(2*3.1415926535897931*(Q*x-tf.round(Q*x))))
    #                           (√√(1 - cos(2pi*points))), -0.5<=points<=+0.5
    return .01 * tf.reduce_mean(K.sqrt(K.sqrt(1.0001 - tf.math.cos(2*3.1415926535897931*(Q*x-tf.round(Q*x))))))


class SparsifyGRUA(Callback):
    def __init__(self, t_start: int, t_end: int, interval: int, density: Tuple[float, float, float], quantize: bool = False, from_step: int = 0):
        """
        Args:
            t_start: The global step at which weight processing gradually start [step]
            t_end:   The global step from which weight processing is full-scale [step]
            interval: Weight processing interval [step]
            density: Sparsification density parameter
            quantize: Whether to quantize, which also affects spasification schedule
            from_step - Resume training from this global step
        """
        super().__init__()

        self.t_start = t_start
        self.t_end = t_end
        self.interval = interval
        self.final_density = density
        self.quantize = quantize

        # The Number of global steps (processed batches)
        self.batch = from_step

    def on_batch_end(self, batch, logs=None):
        """(TF API) Called every batch end."""

        self.batch += 1

        # ## Shedules
        # Sparsification
        #               0 ----------- t_start ----------------------------- t_end ---------------------
        # [non-quant]      <nothing>           <once per interval, gradual>        <every steps, full>
        # [quant]         <                          every steps, full                               >

        # Quantization
        #               0 ----------- t_start ----------------------------- t_end ---------------------
        # [quant]          <nothing>           <once per interval, gradual>        <every steps, full>

        #    quantize_mode OR (global_step > t_start AND every `interval` steps) OR (global_step > t_end)
        if self.quantize or (self.batch > self.t_start and (self.batch-self.t_start) % self.interval == 0) or self.batch >= self.t_end:
            layer = self.model.get_layer('gru_a')
            # w::[Wih::(dim_i, 3*dim_h), Whh::(dim_h, 3*dim_h), b::(2*3*dim_h,)] - GRU weights
            w = layer.get_weights()
            # p::(dim_h, 3*dim_h) - Recurrent weight matrix Whh
            p = w[1]
            # The number of gates
            nb: int = p.shape[1]//p.shape[0]
            assert nb == 3, f"The number of gates should be 3 in GRU, but {nb}"
            # N - dim_h
            N = p.shape[0]

            # Sparsification
            for k in range(nb):
                # gate_k ( g_update | g_reset | g_state )
                ## density - Sparsification density of g_k
                density: float = self.final_density[k]
                ## A::(dim_h, dim_h) - g_k sub-matrix
                A = p[:, k*N:(k+1)*N]

                # Density attenuation until t_end (quantize_mode is always full sparsification)
                if self.batch < self.t_end and not self.quantize:
                    r = 1 - (self.batch-self.t_start)/(self.t_end - self.t_start)
                    #         1 -   sparsity * attenuation_rate
                    density = 1 - (1-density)*(1 - r*r*r)

                #### Block masking by keeping topN blocks without considering diagonal terms ###################
                # Exclude diagonal terms, which are preserved independently
                A = A - np.diag(np.diag(A))

                # This is needed because of the CuDNNGRU strange weight ordering
                A = np.transpose(A, (1, 0))

                # Derivatives: 16x1 @original -> 8x4 @lpcnet_efficiency (from @3ae54e9)
                # 8x4 Block-nize (value squared)
                ## (dim_h, dim_h) -> (dim_h/4, 4, dim_h/8, 8)
                L=np.reshape(A, (N//4, 4, N//8, 8))
                ## (dim_h/4, 4, dim_h/8, 8) -> **2 -> sum -> (dim_h/4, 4, dim_h/8)
                S=np.sum(L*L, axis=-1)
                ## (dim_h/4, 4, dim_h/8) -> (dim_h/4, dim_h/8)
                S=np.sum(S, axis=1)

                # Zero mask threshold for keeping TopN blocks
                SS=np.sort(np.reshape(S, (-1,)))
                idx_thresh = round(N*N//32*(1-density)) # num_blocks*sparsity
                thresh = SS[idx_thresh]

                # ZeroMask generation
                ## Block-wise mask (False=0 | True=1)
                mask = (S>=thresh).astype('float32')
                ## Block to sub-matrix A
                mask = np.repeat(mask, 4, axis=0)
                mask = np.repeat(mask, 8, axis=1)
                ## Diagonal term preservation
                ##                non_diag/0    diag/1
                ##      masked/0      0           1
                ##  non_masked/1      1         2 -> clipped to 1
                mask = np.minimum(1, mask + np.diag(np.ones((N,))))

                # This is needed because of the CuDNNGRU strange weight ordering
                mask = np.transpose(mask, (1, 0))

                # In-place apply of sparsification mask
                p[:, k*N:(k+1)*N] = p[:, k*N:(k+1)*N]*mask
                #################################################################################
            # /Sparsification

            #  quantize_mode AND ((global_step > t_start AND every `interval` steps) OR global_step > t_end)
            if self.quantize and ((self.batch > self.t_start and (self.batch-self.t_start) % self.interval == 0) or self.batch >= self.t_end):
                # Threshold update until t_end
                if self.batch < self.t_end:
                    threshold = .5*(self.batch - self.t_start)/(self.t_end - self.t_start)
                else:
                    threshold = .5
                quant = np.round(p*128.)
                res = p*128.-quant
                mask = (np.abs(res) <= threshold).astype('float32')
                p = mask/128.*quant + (1-mask)*p

            w[1] = p
            layer.set_weights(w)

class SparsifyGRUB(Callback):
    def __init__(self, t_start, t_end, interval, grua_units, density, quantize=False, from_step:int=0):
        super().__init__()

        self.t_start = t_start
        self.t_end = t_end
        self.interval = interval
        self.final_density = density
        self.grua_units = grua_units
        self.quantize = quantize

        # The Number of global steps (processed batches)
        self.batch = from_step

    def on_batch_end(self, batch, logs=None):
        """(TF API) Called every batch end."""

        self.batch += 1
        if self.quantize or (self.batch > self.t_start and (self.batch-self.t_start) % self.interval == 0) or self.batch >= self.t_end:
            #print("constrain");
            layer = self.model.get_layer('gru_b')
            w = layer.get_weights()
            p = w[0]
            N = p.shape[0]
            M = p.shape[1]//3
            for k in range(3):
                density = self.final_density[k]
                if self.batch < self.t_end and not self.quantize:
                    r = 1 - (self.batch-self.t_start)/(self.t_end - self.t_start)
                    density = 1 - (1-self.final_density[k])*(1 - r*r*r)
                A = p[:, k*M:(k+1)*M]
                #This is needed because of the CuDNNGRU strange weight ordering
                A = np.reshape(A, (M, N))
                A = np.transpose(A, (1, 0))
                N2 = self.grua_units
                A2 = A[:N2, :]
                L=np.reshape(A2, (N2//4, 4, M//8, 8))
                S=np.sum(L*L, axis=-1)
                S=np.sum(S, axis=1)
                SS=np.sort(np.reshape(S, (-1,)))
                thresh = SS[round(M*N2//32*(1-density))]
                mask = (S>=thresh).astype('float32')
                mask = np.repeat(mask, 4, axis=0)
                mask = np.repeat(mask, 8, axis=1)
                A = np.concatenate([A2*mask, A[N2:,:]], axis=0)
                #This is needed because of the CuDNNGRU strange weight ordering
                A = np.transpose(A, (1, 0))
                A = np.reshape(A, (N, M))
                p[:, k*M:(k+1)*M] = A
                #print(thresh, np.mean(mask))
            if self.quantize and ((self.batch > self.t_start and (self.batch-self.t_start) % self.interval == 0) or self.batch >= self.t_end):
                if self.batch < self.t_end:
                    threshold = .5*(self.batch - self.t_start)/(self.t_end - self.t_start)
                else:
                    threshold = .5
                quant = np.round(p*128.)
                res = p*128.-quant
                mask = (np.abs(res) <= threshold).astype('float32')
                p = mask/128.*quant + (1-mask)*p

            w[0] = p
            layer.set_weights(w)
            

class PCMInit(Initializer):
    """
    (Local class)
    """
    def __init__(self, gain=.1, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=None):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        if self.seed is not None:
            np.random.seed(self.seed)
        a = np.random.uniform(-1.7321, 1.7321, flat_shape)
        #a[:,0] = math.sqrt(12)*np.arange(-.5*num_rows+.5,.5*num_rows-.4)/num_rows
        #a[:,1] = .5*a[:,0]*a[:,0]*a[:,0]
        a = a + np.reshape(math.sqrt(12)*np.arange(-.5*num_rows+.5,.5*num_rows-.4)/num_rows, (num_rows, 1))
        return self.gain * a.astype("float32")

    def get_config(self):
        return {
            'gain': self.gain,
            'seed': self.seed
        }

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2, **kwargs):
        self.c = c

    def __call__(self, p):
        # Ensure that abs of adjacent weights don't sum to more than 127. Otherwise there's a risk of
        # saturation when implementing dot products with SSSE3 or AVX2.
        return self.c*p/tf.maximum(self.c, tf.repeat(tf.abs(p[:, 1::2])+tf.abs(p[:, 0::2]), 2, axis=1))
        #return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}


def new_lpcnet_model(
    rnn_units1: int = 384,
    rnn_units2: int= 16,
    dim_feat: int = 20,
    batch_size: int = 128,
    training: bool = False,
    adaptation: bool = False,
    quantize: bool = False,
    flag_e2e: bool = False,
    cond_size: int = 128,
    lpc_order: int = 16,
):
    """
    Construct the `lpcnet` model.

    Args:
        rnn_units1 - Dimension size of GRUa
        rnn_units2 - Dimension size of GRUb
        dim_feat - Dimension size of `feat`, which is used as direct input to FrameRateNetwork
        batch_size - Batch size
        training - mode?
        adaptation - mode?
        quantize - mode?
        flag_e2e - Whether to use End-to-End mode
        cond_size - Dimension size of FrameRateNetwork hidden and output feature
        lpc_order - Order of linear prediction
    Returns:
        model - Whole model for joint Encoder-Decoder (e.g. for training)
            inputs:
                s_t_1_noisy_series ::(B, T_s, 1)      - Lagged/Delayed sample series (waveform) with noise
                feat_series        ::(B, T_f, Feat)   - Acoustic feature series
                pitch_series       ::(B, T_f, 1)      - Pitch series
                lpcoeff_series     ::(B, T_f, Order)  - Linear Prediction coefficient series, for default (non-E2E)
            outputs:
                o_t_series     ::(B, T_s, 1+Dist) - Series of p_t_noisy and P(e_t)
                cond_series    ::(B, T_f, Feat)   - Condition series, for E2E regularization

        encoder - FrameRateNetwork/Encoder  sub model for separated enc/dec (e.g. for compression)
        decoder - SampleRateNetwork/Decoder sub model for separated enc/dec (e.g. for compression)
            inputs:
                dpcm       ::(B, T_sample, Feat=3) - SampleRateNetwork inputs (p_t, e_t-1, s_t-1)
                dec_feat   ::(B, T_?,      Feat)   - Conditioining vector series == FrameRateNetwork output (not `rep` only in decoder, so T_sample?)
                dec_state1 ::(B,           h_GRUa) - Initial hidden state of GRUa
                dec_state2 ::(B,           h_GRUb) - Initial hidden state of GRUb
    """

    #### Hardcoded constants ######################################################################
    SIZE_PITCH_CODEBOOK, DIM_PITCH_EMB = 256, 64
    SIZE_FCONV_KERNEL = 3
    
    #### Inputs ###################################################################################
    #                           Time, Feat,                 Batch
    s_t_1_noisy_series = Input(shape=(None, 1),        batch_size=batch_size)
    feat_series        = Input(shape=(None, dim_feat), batch_size=batch_size)
    pitch_series       = Input(shape=(None, 1),        batch_size=batch_size)

    #### FrameRateNetwork #########################################################################
    # Pitch Period embedding: (B, T_f, 1) -> (B, T_f, 1, Emb) -> (B, T_f, Emb)
    pembed = Embedding(SIZE_PITCH_CODEBOOK, DIM_PITCH_EMB, name='embed_pitch')
    pitch_embedded = Reshape((-1, DIM_PITCH_EMB))(pembed(pitch_series))

    # Concat for network input: ((B, T_f, Feat=feat), (B, T_f, Emb=emb)) -> (B, T_f, Feat=feat+emb)
    i_cond_net_series = Concatenate()([feat_series, pitch_embedded])

    # ConvSegFC: Conv1d_c/k/s1-tanh-Conv1d_c/k/s1-tanh-SegFC_c-tanh-SegFC_c-tanh
    padding = 'valid' if training else 'same'
    fconv2 = Conv1D(cond_size, SIZE_FCONV_KERNEL, padding=padding, activation='tanh', name='feature_conv2')
    fconv1 = Conv1D(cond_size, SIZE_FCONV_KERNEL, padding=padding, activation='tanh', name='feature_conv1')
    fdense1 = Dense(cond_size,                                     activation='tanh', name='feature_dense1')
    fdense2 = Dense(cond_size,                                     activation='tanh', name='feature_dense2')
    # Derivatives: Deprecated residual connection (ON @original -> OFF @efficiency, from @5ae0b07)
    # (B, T_f, Feat=i) -> (B, T_f, Feat=o), T_f becomes small by Conv w/o padding
    cond_series = fdense2(fdense1(fconv2(fconv1(i_cond_net_series))))

    # todo: when to `quantize`
    if flag_e2e and quantize:
        fconv1.trainable = False
        fconv2.trainable = False
        fdense1.trainable = False
        fdense2.trainable = False

    #### Linear Prediction ########################################################################
    # LP coefficinets
    ## E2E: RC_in_Cond-to-LPC
    if flag_e2e:
        # (B, T_f, Feat=RC+α) -> (B, T_f, Order)
        lpcoeff_series = diff_rc2lpc(name = "rc2lpc")(cond_series)
    ## Defalt: Input
    else:
        # Inputs                       T_f,     Order              Batch
        lpcoeff_series = Input(shape=(None, lpc_order), batch_size=batch_size)

    # Linear Prediction
    # ((B, T_s, 1), (B, T_f, Order)) -> (B, T_s, 1)
    p_t_noisy_series = diff_pred(name = "lpc2preds")([s_t_1_noisy_series, lpcoeff_series])

    # Residual
    #                        s_t_1_series - p_t_1_series (1 sample lagged)
    residual_past = Lambda(lambda x: x[0] - tf.roll(x[1], 1, axis = 1))
    # Derivatives: Noise source at residual (`e_t_1 = s_t_1_clean - p_t_1_noisy` @original -> `e_t_1 = s_t_1_noisy - p_t_1_noisy` from @b858ea9)
    e_t_1_noisy_series = residual_past([s_t_1_noisy_series, p_t_noisy_series])

    #### SampleRateNetwork ########################################################################
    ###### Embedding #########################################################
    # s_{t-1} ---> μLaw ---.
    # p_t     ---> μLaw ---|---> GaussianNoise -> Embedding -> Reshape -> `i_sample_net_embedded`
    # e_{t-1} ---> μLaw ---'

    # ((B, T_s, 1), (B, T_s, 1), (B, T_s, 1)) -> (B, T_s, 3)
    i_sample_net_cat = Concatenate()([tf_l2u(s_t_1_noisy_series), tf_l2u(p_t_noisy_series), tf_l2u(e_t_1_noisy_series)])
    # Derivatives: Additional noise (OFF @original -> ON from @5ab3f11)
    i_sample_net_cat = GaussianNoise(.3)(i_sample_net_cat)
    # Derivatives: Single embedding for three features (Emb_s_p & Emb_e @original -> 1 Emb from @2b698fd)
    # Differential embedding for E2E mode (in normal mode, no difference from original LPCNet)
    embed = diff_Embed(name='embed_sig', initializer = PCMInit())
    # (B, T_s, 3) -> (B, T_s, 3, Emb=emb) -> (B, T_s, Emb=3*emb)
    i_sample_net_embedded = Reshape((-1, embed_size*3))(embed(i_sample_net_cat))

    ###### Networks ##########################################################
    # Definition of GRU_A `rnn` & GRU_B `rnn2`
    quant = quant_regularizer if quantize else None
    constraint = WeightClip(0.992)
    # `CuDNNGRU` and `GRU` seems to be layer in TF1.0, but it may not be upgraded because of backward compatibility.
    #   "以前の keras.layers.CuDNNLSTM/CuDNNGRU レイヤーは使用廃止となったため、実行するハードウェアを気にせずにモデルを構築することができます。" from https://www.tensorflow.org/guide/keras/rnn?hl=ja
    # Derivatives: State preservation for exposure bias (stateful=False @original -> stateful=True @efficiency)
    if training:
        # [CuDNNGRU](https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/layers/CuDNNGRU)
        # gru_a: (B, T_s, Feat=i) => (B, T_s, Feat=o), (B, T_s, Feat=h)
        #   weight clipping and `quant` is applied on recurrent_kernel.
        rnn = CuDNNGRU(rnn_units1, return_sequences=True, return_state=True, name='gru_a', stateful=True,
              recurrent_constraint = constraint, recurrent_regularizer=quant)
        # gru_b: (B, T_s, Feat=i) => (B, T_s, Feat=o), (B, T_s, Feat=h)
        #   state is preserved between batches.
        #   weight clipping and `quant` is applied on both recurrent_kernel and kernel.
        rnn2 = CuDNNGRU(rnn_units2, return_sequences=True, return_state=True, name='gru_b', stateful=True,
               kernel_constraint=constraint, recurrent_constraint = constraint, kernel_regularizer=quant, recurrent_regularizer=quant)
    else:
        # [GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
        # `reset_after` & `recurrent_activation` are for GPU/CUDA compatibility.
        rnn = GRU(     rnn_units1, return_sequences=True, return_state=True, name='gru_a', stateful=True,
              recurrent_constraint = constraint, recurrent_regularizer=quant,
              recurrent_activation="sigmoid", reset_after='true')
        rnn2 = GRU(     rnn_units2, return_sequences=True, return_state=True, name='gru_b', stateful=True,
              kernel_constraint=constraint, recurrent_constraint = constraint, kernel_regularizer=quant, recurrent_regularizer=quant,
              recurrent_activation="sigmoid", reset_after='true')

    # Frame repeat :: (B, T_f, Feat) -> (B, T_s==T_f*frame_size, Feat)
    rep = Lambda(lambda x: K.repeat_elements(x, frame_size, 1))

    # B(l_k=1|l_<k), DualFC-sigmoid
    md = MDense(pcm_levels, activation='sigmoid', name='dual_fc')

    # rep(cond_series) ------------------------------------.
    #                        |                             | 
    # i_sample_net_embedded ---> `rnn` -> `GaussianNoise` ---> `rnn2` -> `md` -> Lambda(tree_to_pdf_train)

    rnn_a_in = Concatenate()([i_sample_net_embedded, rep(cond_series)])
    gru_out1, _ = rnn(rnn_a_in)
    # Training-specific Noise addition (not described in original LPC?)
    gru_out1 = GaussianNoise(.005)(gru_out1)
    rnn_b_in = Concatenate()([gru_out1, rep(cond_series)])
    # -> (B, T_s, Feat)
    gru_out2, _ = rnn2(rnn_b_in)

    # Derivatives: Hierarchical Probability Distribution (proposed in `lpcnet_efficiency`, from @d24f49e)
    # (B, T_s, Feat) -> (B, T_s, Prob=2**Q)
    bit_cond_probs = md(gru_out2)
    # P(e_t) series :: (B, T_s, Prob=2**Q) -> (B, T_s, Dist=2**Q)
    e_t_pd_series = Lambda(gen_tree_to_pdf(2400, pcm_bits))(bit_cond_probs)
    #FIXME: try not to hardcode the 2400 samples (15 frames * 160 samples/frame)

    # Series of p_t_noisy and P(e_t) :: ((B, T_s, 1), (B, T_s, Dist)) -> (B, T_s, 1+Dist)
    o_t_series = Concatenate(name='pdf')([p_t_noisy_series, e_t_pd_series])

    #### Model-nize #################################################################################
    if adaptation:
        rnn.trainable=False
        rnn2.trainable=False
        md.trainable=False
        embed.Trainable=False

    # The whole model
    ## Default
    if not flag_e2e:
        model = Model([s_t_1_noisy_series, feat_series, pitch_series, lpcoeff_series], o_t_series)
    ## E2E
    else:
        # w/o explicit LP coefficient input
        # w/ conditioning series output for regularization
        model = Model([s_t_1_noisy_series, feat_series, pitch_series], [o_t_series, cond_series])

    # Register parameters
    model.rnn_units1 = rnn_units1
    model.rnn_units2 = rnn_units2
    model.nb_used_features = dim_feat
    model.frame_size = frame_size

    #### Sub models (separated encoder and decoder) ###############################################
    # Inputs
    #                         Time, Feat
    dpcm =       Input(shape=(None, 3),        batch_size=batch_size)
    dec_feat =   Input(shape=(None, cond_size))
    dec_state1 = Input(shape=(rnn_units1,))
    dec_state2 = Input(shape=(rnn_units2,))

    # Encoder
    if not flag_e2e:
        encoder = Model([feat_series, pitch_series], cond_series)
    else:
        encoder = Model([feat_series, pitch_series], [cond_series, lpcoeff_series])

    # Decoder
    cpcm_decoder = Reshape((-1, embed_size*3))(embed(dpcm))
    dec_rnn_in = Concatenate()([cpcm_decoder, dec_feat])
    dec_gru_out1, state1 = rnn(dec_rnn_in, initial_state=dec_state1)
    dec_gru_out2, state2 = rnn2(Concatenate()([dec_gru_out1, dec_feat]), initial_state=dec_state2)
    dec_ulaw_prob = Lambda(gen_tree_to_pdf(1, pcm_bits))(md(dec_gru_out2))

    decoder = Model([dpcm, dec_feat, dec_state1, dec_state2], [dec_ulaw_prob, state1, state2])
    ###############################################################################################

    return model, encoder, decoder
