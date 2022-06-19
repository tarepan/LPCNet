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

import lpcnet
import sys
import numpy as np
from ulaw import ulaw2lin, lin2ulaw
import h5py


# ==== Model and its parameter load =====================================================================================================================
filename = sys.argv[1]
with h5py.File(filename, "r") as f:
    units = min(f['model_weights']['gru_a']['gru_a']['recurrent_kernel:0'].shape)
    units2 = min(f['model_weights']['gru_b']['gru_b']['recurrent_kernel:0'].shape)
    cond_size = min(f['model_weights']['feature_dense1']['feature_dense1']['kernel:0'].shape)
    e2e = 'rc2lpc' in f['model_weights']

model, enc, dec = lpcnet.new_lpcnet_model(training = False, rnn_units1=units, rnn_units2=units2, flag_e2e = e2e, cond_size=cond_size, batch_size=1)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.load_weights(filename);

sample_per_frame: int = model.frame_size # [sample/frame] in the model
nb_features: int = 36 # Dimension of features in the file
dim_feat: int = model.nb_used_features # Dimension of `feat` model input
# =======================================================================================================================================================


feature_file = sys.argv[2] # Path of input feature file
out_file     = sys.argv[3] # Path of output file to which int16 PCM audio will be written

fout = open(out_file, 'wb') # Output int16 PCM file

features = np.fromfile(feature_file, dtype='float32')
features = np.resize(features, (-1, nb_features))
nb_frames = 1
feature_chunk_size = features.shape[0]
pcm_chunk_size = sample_per_frame*feature_chunk_size

#                               (        1,  len_sample_series, BFCC18+Pitch2+LPcoeff16)
features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))

# Pitch Period, re-scaled
periods = (.1 + 50*features[:,:,18:19]+100).astype('int16')

# CONSTANTS
ORDER: int = 16 # LP order
EMPH_COEFF: float = 0.85 # Preemphasis coefficient

# States
pcm = np.zeros((nb_frames*pcm_chunk_size, ))
fexc = np.zeros((1, 1, 3), dtype='int16') + 128           # AR state (B=1, T=1, Feat=(quant_s_{t-1}, ??, e_{t-1})), initialized with 0::Int8
state1 = np.zeros((1, model.rnn_units1), dtype='float32') # GRU_A state, initialized with zeros
state2 = np.zeros((1, model.rnn_units2), dtype='float32') # GRU_B state, initialized with zeros
emph_mem: float = 0                                       # Preemphasis memory


skip = ORDER + 1
for c in range(0, nb_frames):
    if not e2e:
        cfeat = enc.predict([features[c:c+1, :, :dim_feat], periods[c:c+1, :, :]])
    else:
        cfeat,lpcs = enc.predict([features[c:c+1, :, :dim_feat], periods[c:c+1, :, :]])
    for fr in range(0, feature_chunk_size):
        f = c*feature_chunk_size + fr

        # Linear Prediction coefficients
        if not e2e:
            # Last ORDER dims are LP coefficients
            a = features[c, fr, nb_features-ORDER:]
        else:
            a = lpcs[c,fr]

        for i in range(skip, sample_per_frame):
            # Single sample

            # Linear Prediction
            p_t_estim = -sum(a * pcm[f*sample_per_frame + i - 1:f*sample_per_frame + i - ORDER-1:-1])

            # Packing of (Q(p_t), Q(s_{t-1}), e_{t-1})
            fexc[0, 0, 1] = lin2ulaw(p_t_estim)

            # Probability distribution of e_t by Decoder
            pd_e, state1, state2 = dec.predict([fexc, cfeat[:, fr:fr+1, :], state1, state2])


            # ==== Sampling ==================================================================

            # ======== Tempering =========================================================
            # todo: Here is old tempering (now tree sampling and thresholding is default)

            # Lower the temperature for voiced frames to reduce noisiness
            pitch_corr = features[c, fr, 19]
            pd_e *= np.power(pd_e, np.maximum(0, 1.5*pitch_corr - 0.5))
            # Re-normalize
            pd_e = pd_e/(1e-18 + np.sum(pd_e))

            # Cut off the tail of the remaining distribution
            THRESHOLD = 0.002
            pd_e = np.maximum(pd_e - THRESHOLD, 0).astype('float64')
            # Re-normalize
            pd_e = pd_e/(1e-8 + np.sum(pd_e))
            # ============================================================================

            # Sample e_t once from Distribution p[0,0,:]
            e_t_estim = np.argmax(np.random.multinomial(1, pd_e[0,0,:], 1)) # bitON only sampled index
            # ================================================================================


            # s_t = LP_t + e_t
            s_t_estim = p_t_estim + ulaw2lin(e_t_estim)
            pcm[f*sample_per_frame + i] = s_t_estim

            # Update for next
            fexc[0, 0, 0] = lin2ulaw(s_t_estim) # quant_s_{t-1} <- quant(s_t)
            fexc[0, 0, 2] = e_t_estim           #       e_{t-1} <- e_t

            # De-emphasis
            emph_mem = EMPH_COEFF*emph_mem + s_t_estim

            # Save a sample as int16 PCM
            np.array([np.round(emph_mem)], dtype='int16').tofile(fout)

        skip = 0
