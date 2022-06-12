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

import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation, GaussianNoise
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.callbacks import Callback
from mdense import MDense
import numpy as np
import h5py
import sys
from tf_funcs import *
from diffembed import diff_Embed

frame_size = 160 # Waveform samples per acoustic frame [samples/frame]
pcm_bits = 8
embed_size = 128
pcm_levels = 2**pcm_bits

def _interleave(p, samples):
    """
    """
    p2=tf.expand_dims(p, 3)
    nb_repeats = pcm_levels//(2*p.shape[2])
    p3 = tf.reshape(tf.repeat(tf.concat([1-p2, p2], 3), nb_repeats), (-1, samples, pcm_levels))
    return p3

def _tree_to_pdf(p, samples):
    """
    """
    return _interleave(p[:,:,1:2], samples) * _interleave(p[:,:,2:4], samples) * _interleave(p[:,:,4:8], samples) * _interleave(p[:,:,8:16], samples) \
         * _interleave(p[:,:,16:32], samples) * _interleave(p[:,:,32:64], samples) * _interleave(p[:,:,64:128], samples) * _interleave(p[:,:,128:256], samples)

def _tree_to_pdf_train(p):
    """
    (Local function)
    """
    #FIXME: try not to hardcode the 2400 samples (15 frames * 160 samples/frame)
    return _tree_to_pdf(p, 2400)

def _tree_to_pdf_infer(p):
    """
    (Local function)
    """
    return _tree_to_pdf(p, 1)

def quant_regularizer(x):
    Q = 128
    Q_1 = 1./Q
    #return .01 * tf.reduce_mean(1 - tf.math.cos(2*3.1415926535897931*(Q*x-tf.round(Q*x))))
    return .01 * tf.reduce_mean(K.sqrt(K.sqrt(1.0001 - tf.math.cos(2*3.1415926535897931*(Q*x-tf.round(Q*x))))))

class Sparsify(Callback):
    def __init__(self, t_start, t_end, interval, density, quantize:bool=False):
        """
        Args:
            t_start: The global step at which weight processing gradually start
            t_end:   The global step from which weight processing is full-scale
            interval: Weight processing interval
            density: Sparsification density parameter
            quantize: Whether to quantize, which also affects spasification schedule
        """
        super(Sparsify, self).__init__()
        # Number of global step (processed batches)
        self.batch = 0
        self.t_start = t_start
        self.t_end = t_end
        self.interval = interval
        self.final_density = density
        self.quantize = quantize

    def on_batch_end(self, batch, logs=None):
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
            w = layer.get_weights()
            p = w[1]
            nb = p.shape[1]//p.shape[0]
            N = p.shape[0]
            #print("nb = ", nb, ", N = ", N);
            #print(p.shape)
            #print ("density = ", density)

            # Sparsification
            for k in range(nb):
                density = self.final_density[k]
                # Density attenuation until t_end (quantize_mode is always full sparsification)
                if self.batch < self.t_end and not self.quantize:
                    r = 1 - (self.batch-self.t_start)/(self.t_end - self.t_start)
                    density = 1 - (1-self.final_density[k])*(1 - r*r*r)
                # /Density update
                A = p[:, k*N:(k+1)*N]
                A = A - np.diag(np.diag(A))
                #This is needed because of the CuDNNGRU strange weight ordering
                A = np.transpose(A, (1, 0))
                L=np.reshape(A, (N//4, 4, N//8, 8))
                S=np.sum(L*L, axis=-1)
                S=np.sum(S, axis=1)
                SS=np.sort(np.reshape(S, (-1,)))
                thresh = SS[round(N*N//32*(1-density))]
                mask = (S>=thresh).astype('float32')
                mask = np.repeat(mask, 4, axis=0)
                mask = np.repeat(mask, 8, axis=1)
                mask = np.minimum(1, mask + np.diag(np.ones((N,))))
                #This is needed because of the CuDNNGRU strange weight ordering
                mask = np.transpose(mask, (1, 0))
                p[:, k*N:(k+1)*N] = p[:, k*N:(k+1)*N]*mask
                #print(thresh, np.mean(mask))
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
    def __init__(self, t_start, t_end, interval, grua_units, density, quantize=False):
        super(SparsifyGRUB, self).__init__()
        self.batch = 0
        self.t_start = t_start
        self.t_end = t_end
        self.interval = interval
        self.final_density = density
        self.grua_units = grua_units
        self.quantize = quantize

    def on_batch_end(self, batch, logs=None):
        #print("batch number", self.batch)
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
    (Local class)
    '''
    def __init__(self, c=2):
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
    rnn_units1=384,
    rnn_units2=16,
    nb_used_features:int=20,
    batch_size=128,
    training=False,
    adaptation=False,
    quantize=False,
    flag_e2e = False,
    cond_size=128,
    lpc_order=16
):
    """
    Construct the `lpcnet` model.

    FrameRateNetwork: feat & emb(pitch) => ConvSegFC => f
    SampleRateNetwork: (f, ) => GRU => sampling => e/residual
    LinearPrediction: Cx + e

    Args:
        nb_used_features - Dimension size of `feat`, which is used as direct input to FrameRateNetwork
        batch_size - Batch size
        cond_size - Dimension size of FrameRateNetwork hidden and output feature
    Returns:
        model - Whole model for joint Encoder-Decoder (e.g. for training)
            inputs:
                pcm      ::(B, T_sample, 1)     - Sample series (waveform) for AR teacher-forcing in LinearPrediction/Residual/SampleRateNet
                feat     ::(B, T_frame,  Feat)  - Acoustic feature series for conditioning
                pitch    ::(B, T_frame,  1)     - Pitch series aligned with `feat`
                lpcoeffs ::(B, T_frame,  Order) - Linear Prediction coefficients
            outputs:
                m_out::()

        encoder - Encoder sub model for separated enc/dec (e.g. for compression)
        decoder - Decoder sub model for separated enc/dec (e.g. for compression)
            inputs:
                dec_feat    
                dec_state1::(B, h_GRUa) - Initial hidden state of GRUa
                dec_state2::(B, h_GRUb) - Initial hidden state of GRUb
    """

    # Inputs
    #                         Time, Feat
    pcm =        Input(shape=(None, 1),                batch_size=batch_size)
    feat =       Input(shape=(None, nb_used_features), batch_size=batch_size)
    pitch =      Input(shape=(None, 1),                batch_size=batch_size)

    # Inputs of Decoder sub-model
    dpcm =       Input(shape=(None, 3),                batch_size=batch_size)
    dec_feat =   Input(shape=(None, cond_size))
    dec_state1 = Input(shape=(rnn_units1,))
    dec_state2 = Input(shape=(rnn_units2,))

    #### FrameRateNetwork #########################################################################
    # Pitch embedding: (B, T, Feat=1) -> (B, T, Emb=64) -> (B, T, Emb=64)
    pembed = Embedding(256, 64, name='embed_pitch')
    pitch_embedded = Reshape((-1, 64))(pembed(pitch))

    # Concat: ((B, T, Feat=feat), (B, T, Emb=64)) -> (B, T, Feat=feat+64)
    cat_feat = Concatenate()([feat, pitch_embedded])

    # ConvSegFC: Conv1d_k3cCs1-tanh-Conv1d_k3cCs1-tanh-SegFC_C-tanh-SegFC_C-tanh
    padding = 'valid' if training else 'same'
    fconv1 = Conv1D(cond_size, 3, padding=padding, activation='tanh', name='feature_conv1')
    fconv2 = Conv1D(cond_size, 3, padding=padding, activation='tanh', name='feature_conv2')
    fdense1 = Dense(cond_size, activation='tanh', name='feature_dense1')
    fdense2 = Dense(cond_size, activation='tanh', name='feature_dense2')
    cfeat = fconv2(fconv1(cat_feat))
    cfeat = fdense2(fdense1(cfeat))

    if flag_e2e and quantize:
        fconv1.trainable = False
        fconv2.trainable = False
        fdense1.trainable = False
        fdense2.trainable = False

    #### Linear Prediction ########################################################################
    if flag_e2e:
        lpcoeffs = diff_rc2lpc(name = "rc2lpc")(cfeat)
    else:
        # Inputs
        #                       Time,   Coeff
        lpcoeffs = Input(shape=(None, lpc_order), batch_size=batch_size)
    error_calc = Lambda(lambda x: tf_l2u(x[0] - tf.roll(x[1],1,axis = 1)))

    # Linear Prediction
    tensor_preds = diff_pred(name = "lpc2preds")([pcm,lpcoeffs])
    # Residual μ-law
    past_errors = error_calc([pcm,tensor_preds])

    #### SampleRateNetwork ########################################################################
    # past_errors/reidual ---------------------------|
    # tensor_preds -> μLaw `tf_l2u` -----------------|
    # `pcm` (teacher forcing) -> μLaw `tf_l2u` -----------> GaussianNoise -> diff_Embed `embed` -> Reshape => `cpcm`
    embed = diff_Embed(name='embed_sig',initializer = PCMInit())
    cpcm = Concatenate()([tf_l2u(pcm),tf_l2u(tensor_preds),past_errors])
    cpcm = GaussianNoise(.3)(cpcm)
    cpcm = Reshape((-1, embed_size*3))(embed(cpcm))

    cpcm_decoder = Reshape((-1, embed_size*3))(embed(dpcm))

    # Definition of GRU_A `rnn` & GRU_B `rnn2`
    quant = quant_regularizer if quantize else None
    constraint = WeightClip(0.992)
    if training:
        # [CuDNNGRU](https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/layers/CuDNNGRU)
        # gru_a: (B, T, F_i) => (B, T, F_o), (B, T, F_h)
        #   state is preserved between batches.
        #   weight clipping and `quant` is applied on recurrent_kernel.
        rnn = CuDNNGRU(rnn_units1, return_sequences=True, return_state=True, name='gru_a', stateful=True,
              recurrent_constraint = constraint, recurrent_regularizer=quant)
        # gru_b: (B, T, F_i) => (B, T, F_o), (B, T, F_h)
        #   state is preserved between batches.
        #   weight clipping and `quant` is applied on both recurrent_kernel and kernel.
        rnn2 = CuDNNGRU(rnn_units2, return_sequences=True, return_state=True, name='gru_b', stateful=True,
               kernel_constraint=constraint, recurrent_constraint = constraint, kernel_regularizer=quant, recurrent_regularizer=quant)
    else:
        # [GRU](https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/layers/GRU)
        # `reset_after` & `recurrent_activation` are for GPU/CUDA compatibility.
        rnn = GRU(rnn_units1, return_sequences=True, return_state=True, name='gru_a', stateful=True,
              recurrent_constraint = constraint, recurrent_regularizer=quant,
              recurrent_activation="sigmoid", reset_after='true')
        rnn2 = GRU(rnn_units2, return_sequences=True, return_state=True, name='gru_b', stateful=True,
              kernel_constraint=constraint, recurrent_constraint = constraint, kernel_regularizer=quant, recurrent_regularizer=quant,
              recurrent_activation="sigmoid", reset_after='true')

    # Frame repeat :: (B, T_frame, Feat) -> (B, T_sample=T_frame*frame_size, Feat)
    rep = Lambda(lambda x: K.repeat_elements(x, frame_size, 1))

    md = MDense(pcm_levels, activation='sigmoid', name='dual_fc')

    # rep(cfeat)------------------------------|
    #            |                            | 
    # cpcm -------> `rnn` -> `GaussianNoise` --> `rnn2` -> `md` -> Lambda(_tree_to_pdf_train)
    rnn_a_in = Concatenate()([cpcm, rep(cfeat)])
    gru_out1, _ = rnn(rnn_a_in)
    gru_out1 = GaussianNoise(.005)(gru_out1)
    rnn_b_in = Concatenate()([gru_out1, rep(cfeat)])
    gru_out2, _ = rnn2(rnn_b_in)

    # Derivatives from original LPCNet: "binary probability tree" (proposed in `lpcnet_efficiency`) from @d24f49e
    ulaw_prob = Lambda(_tree_to_pdf_train)(md(gru_out2))

    if adaptation:
        rnn.trainable=False
        rnn2.trainable=False
        md.trainable=False
        embed.Trainable=False

    # Output#1
    m_out = Concatenate(name='pdf')([tensor_preds,ulaw_prob])

    # The whole model
    if not flag_e2e:
        model = Model([pcm, feat, pitch, lpcoeffs], m_out)
    else:
        model = Model([pcm, feat, pitch], [m_out, cfeat])

    # Register parameters
    model.rnn_units1 = rnn_units1
    model.rnn_units2 = rnn_units2
    model.nb_used_features = nb_used_features
    model.frame_size = frame_size

    # Sub models (separated encoder and decoder)
    if not flag_e2e:
        encoder = Model([feat, pitch], cfeat)
        dec_rnn_in = Concatenate()([cpcm_decoder, dec_feat])
    else:
        encoder = Model([feat, pitch], [cfeat,lpcoeffs])
        dec_rnn_in = Concatenate()([cpcm_decoder, dec_feat])
    dec_gru_out1, state1 = rnn(dec_rnn_in, initial_state=dec_state1)
    dec_gru_out2, state2 = rnn2(Concatenate()([dec_gru_out1, dec_feat]), initial_state=dec_state2)
    dec_ulaw_prob = Lambda(_tree_to_pdf_infer)(md(dec_gru_out2))
    if flag_e2e:
        decoder = Model([dpcm, dec_feat, dec_state1, dec_state2], [dec_ulaw_prob, state1, state2])
    else:
        decoder = Model([dpcm, dec_feat, dec_state1, dec_state2], [dec_ulaw_prob, state1, state2])

    return model, encoder, decoder
