"""Train LPCNet."""

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

import sys
import argparse
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import h5py

from ulaw import ulaw2lin, lin2ulaw
from tf_funcs import *
from lossfuncs import *
from dataloader import LPCNetLoader


#### Args ####################################################################################################################
parser = argparse.ArgumentParser(description='Train an LPCNet model')

# Default model:           FrameRateNet128-GRUa384Sparse-GRUb16Dense,  fp32
# Efficient model (paper):                 GRUa384Sparse-GRUb32Sparse, int8

# Data
parser.add_argument('features', metavar='<features file>', help='binary features file (float32)')
parser.add_argument('data', metavar='<audio data file>', help='binary audio data file (uint8)')
parser.add_argument('output', metavar='<output>', help='trained model file (.h5)')
# Model mode
parser.add_argument('--model', metavar='<model>', default='lpcnet', help='LPCNet model python definition (without .py)')
group1 = parser.add_mutually_exclusive_group()
group1.add_argument('--quantize', metavar='<input weights>', help='quantize model')
group1.add_argument('--retrain', metavar='<input weights>', help='continue training model')
# Model params
## Sparsity
parser.add_argument('--density', metavar='<global density>', type=float, help='average density of the recurrent weights (default 0.1)')
parser.add_argument('--density-split', nargs=3, metavar=('<update>', '<reset>', '<state>'), type=float, help='density of each recurrent gate (default 0.05, 0.05, 0.2)')
parser.add_argument('--grub-density', metavar='<global GRU B density>', type=float, help='average density of the recurrent weights (default 1.0)')
parser.add_argument('--grub-density-split', nargs=3, metavar=('<update>', '<reset>', '<state>'), type=float, help='density of each GRU B input gate (default 1.0, 1.0, 1.0)')
## Model size
parser.add_argument('--grua-size', metavar='<units>', default=384, type=int, help='number of units in GRU A (default 384)')
parser.add_argument('--grub-size', metavar='<units>', default=16, type=int, help='number of units in GRU B (default 16)')
parser.add_argument('--cond-size', metavar='<units>', default=128, type=int, help='number of units in conditioning network, aka frame rate network (default 128)')
# Train
## In original LPCNet paper, 120 epochs is 230k steps (c.f. 20epochs/767Ksteps@lpcnet_efficiency)
parser.add_argument('--epochs', metavar='<epochs>', default=120, type=int, help='number of epochs to train for (default 120)')
## Derivatives from original LPCNet (64@original -> 128@lpcnet_efficiency):
parser.add_argument('--batch-size', metavar='<batch size>', default=128, type=int, help='batch size to use (default 128)')
parser.add_argument('--end2end', dest='flag_e2e', action='store_true', help='Enable end-to-end training (with differentiable LPC computation')
## Optim
parser.add_argument('--lr', metavar='<learning rate>', type=float, help='learning rate')
parser.add_argument('--decay', metavar='<decay>', type=float, help='learning rate decay')
parser.add_argument('--gamma', metavar='<gamma>', type=float, help='adjust u-law compensation (default 2.0, should not be less than 1.0)')
#
parser.add_argument('--lookahead', metavar='<nb frames>', default=2, type=int, help='Number of look-ahead frames (default 2)')
parser.add_argument('--logdir', metavar='<log dir>', help='directory for tensorboard log files')
parser.add_argument('--resume-model', metavar='<epoch>', type=str, help='Resume training from this model')
parser.add_argument('--from-epoch', metavar='<epoch>', type=int, default=0, help='Resume training from this epoch')
parser.add_argument('--from-step', metavar='<step>', type=int, default=0, help='Resume training from this global step')

args = parser.parse_args()


#### Confs ###################################################################################################################
# Model module (default: `./lpcnet.py`)
import importlib
lpcnet = importlib.import_module(args.model)

batch_size = args.batch_size
quantize = args.quantize is not None
retrain = args.retrain is not None
flag_e2e = args.flag_e2e
lpc_order = 16


#### Model ###################################################################################################################
## Training resume
if args.resume_model is not None:
    model = keras.models.load_model(args.resume_model)
    print(f"Resumed from Model {args.resume_model}")
    model.summary()
    # todo: Check whether `load_model` restore variable in the model (In our case, .frame_size & .nb_used_features)
    print(f"values from the model: .frame_size={model.frame_size}, .nb_used_features={model.nb_used_features}")
## From scratch
else:
    gamma = 2.0 if args.gamma is None else args.gamma

    # Optimizer/Scheduler
    ##       arg                                      quantize                default
    lr =    args.lr    if (args.lr    is not None) else 0.00003 if quantize else 0.001
    decay = args.decay if (args.decay is not None) else 0       if quantize else 2.5e-5
    opt = Adam(lr, decay=decay, beta_2=0.99)

    # Model
    ## For Distributed learning
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
    ##/
        model, _, _ = lpcnet.new_lpcnet_model(rnn_units1=args.grua_size, rnn_units2=args.grub_size, batch_size=batch_size, training=True, quantize=quantize, flag_e2e = flag_e2e, cond_size=args.cond_size)
        if not flag_e2e:
            model.compile(optimizer=opt, loss=metric_cel, metrics=metric_cel)
        else:
            model.compile(optimizer=opt, loss = [interp_mulaw(gamma=gamma), loss_matchlar()], loss_weights = [1.0, 2.0], metrics={'pdf':[metric_cel,metric_icel,metric_exc_sd,metric_oginterploss]})
        # Report model architecture
        model.summary()

## Additional training mode
if quantize or retrain:
    input_model = args.quantize if quantize else args.retrain
    model.load_weights(input_model)


#### Data ####################################################################################################################
sample_per_frame = model.frame_size                  # Waveform samples per acoustic frame [samples/frame]
nb_used_features = model.nb_used_features            # Feature dim size of `feat`, which is used as direct input to FrameRateNetwork
frame_per_chunk = 15                                 # (maybe) The number of frames per chunk (item)
sample_per_chunk = sample_per_frame*frame_per_chunk  # (maybe) The number of samples per chunk (item)
# u for unquantised, load 16 bit PCM samples and convert to mu-law

# np.memmap for partial access to single big file
## all-utterance file (<data.s16>, 16 bit unsigned short PCM samples)
samples = np.memmap(args.data, dtype='int16', mode='r')
## all-feature (acoustic features and pitchs) file (<features.f32>)
features = np.memmap(args.features, dtype='float32', mode='r')

# The number of chunks in <data.s16> (variable naming is wrong...?)
nb_frames = (len(samples)//(2*sample_per_chunk)-1)//batch_size*batch_size

#### Samples ####
# Discard head samples
samples = samples[(4-args.lookahead)*2*sample_per_frame:]
# Discard chippings
samples = samples[:nb_frames*2*sample_per_chunk]
# samples :: (Chunk, T_sample, IO=2)
samples = np.reshape(samples, (nb_frames, sample_per_chunk, 2))

#### Acoustic Feature series and LP coefficient series ####
sizeof = features.strides[-1]
nb_features = nb_used_features + lpc_order # Feature dim size of <features.f32>, equal to `dim_feat + order_lpc`
# features :: (Chunk, )
features = np.lib.stride_tricks.as_strided(features, shape=(nb_frames, frame_per_chunk+4, nb_features),
                                           strides=(frame_per_chunk*nb_features*sizeof, nb_features*sizeof, sizeof))
#features = features[:, :, :nb_used_features]

#### Pitch Period series ####
# idx=-1 is Pitch Correlation series, idx=-2 is Pitch Period series (maybe)
periods = (.1 + 50*features[:,:,nb_used_features-2:nb_used_features-1]+100).astype("int16")
#periods = np.minimum(periods, 255)

# Construct data loader
loader = LPCNetLoader(samples, features, periods, batch_size, e2e=flag_e2e)


#### Sparsification and Quantization #########################################################################################
# GRUa sparsification params: '<update>', '<reset>', '<state>'
grua_density: Tuple[float, float, float] = (0.05, 0.05, 0.2)
if args.density_split is not None:
    grua_density = args.density_split
elif args.density is not None:
    # 1:1:4
    grua_density = [0.5*args.density,      0.5*args.density,      2.0*args.density];

# GRUb sparsification params
grub_density: Tuple[float, float, float] = (1., 1., 1.)
if args.grub_density_split is not None:
    grub_density = args.grub_density_split
elif args.grub_density is not None:
    # 1:1:4
    grub_density = [0.5*args.grub_density, 0.5*args.grub_density, 2.0*args.grub_density];

# Schedules
if quantize:
    # Full-scale sparsification from step#0 & Gradual quantization from step#10000
    t_start, t_end, interval = 10000, 30000, 100
elif retrain:
    # Full-scale sparsification from step#0 (no quantization)
    t_start, t_end, interval =     0,     0,   1
else:
    # Gradual sparsification from step#2000 (no quantization). c.f. Total 230_000 steps in original LPCNet paper
    t_start, t_end, interval =  2000, 40000, 400

# Construct callbacks
grua_sparsify = lpcnet.SparsifyGRUA(t_start, t_end, interval,                 grua_density, quantize=quantize, from_step=args.from_step)
grub_sparsify = lpcnet.SparsifyGRUB(t_start, t_end, interval, args.grua_size, grub_density, quantize=quantize, from_step=args.from_step)


##############################################################################################################################
# Checkpointing (Model weights & Optimizer state)
checkpoint = ModelCheckpoint('{}_{}_{}.h5'.format(args.output, args.grua_size, '{epoch:02d}'))
model.save_weights(f"{args.output}_{args.grua_size}_initial.h5")

callbacks = [checkpoint, grua_sparsify, grub_sparsify]

# Logging
if args.logdir is not None:
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=f'{args.logdir}/{args.output}_{args.grua_size}_logs'
    ))

# Run training (always start from epoch#0 (`initial_epoch` is not used))
model.fit(loader, initial_epoch=args.from_epoch, epochs=args.epochs, validation_split=0.0, callbacks=callbacks)
