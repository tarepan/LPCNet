/* Copyright (c) 2018 Mozilla
                 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
/*
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
*/

/**
 * Neural network components.
 *
 * - `vec_swish`
 * - `compute_activation`
 * - `sgemv_accum`
 * - for 'frame rate network' only
 *   - `compute_embedding`
 *   - `compute_conv1d`
 *   - `_lpcnet_compute_dense`
 * - for 'sample rate network' only
 *   - `compute_gru_a_input`
 *   - `compute_sparse_gru`
 *   - `compute_gruB`
 *   - `sample_mdense`
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <math.h>
#include "opus_types.h"
#include "arch.h"
#include "common.h"
#include "tansig_table.h"
#include "nnet.h"
#include "nnet_data.h"

#ifdef NO_OPTIMIZATIONS
#warning Compiling without any vectorization. This code will be very slow
#endif


#define SOFTMAX_HACK

#define MAX_ACTIVATIONS (4096)

static OPUS_INLINE void vec_swish(float *y, const float *x, int N)
{
   int i;
   float tmp[MAX_ACTIVATIONS];
   celt_assert(N <= MAX_ACTIVATIONS);
   vec_sigmoid(tmp, x, N);
   for (i=0;i<N;i++)
      y[i] = x[i]*tmp[i];
}

static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}

static void sgemv_accum(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   if (rows % 16 == 0)
   {
      sgemv_accum16(out, weights, rows, cols, col_stride, x);
   } else {
      for (i=0;i<rows;i++)
      {
         for (j=0;j<cols;j++)
            out[i] += weights[j*col_stride + i]*x[j];
      }
   }
}

/**
 *
 * Args:
 *   activation - Enum of activation function type (see `nnet.h`)
 */
void compute_activation(float *output, const float *input, int N, int activation)
{
   int i;
   if (activation == ACTIVATION_SIGMOID) {
      vec_sigmoid(output, input, N);
   } else if (activation == ACTIVATION_TANH) {
      vec_tanh(output, input, N);
   } else if (activation == ACTIVATION_SWISH) {
      vec_swish(output, input, N);
   } else if (activation == ACTIVATION_RELU) {
      for (i=0;i<N;i++)
         output[i] = relu(input[i]);
   } else if (activation == ACTIVATION_SOFTMAX) {
#ifdef SOFTMAX_HACK
      RNN_COPY(output, input, N);
      /*for (i=0;i<N;i++)
         output[i] = input[i];*/
#else
      float sum = 0;
      softmax(output, input, N);
      for (i=0;i<N;i++) {
         sum += output[i];
      }
      sum = 1.f/(sum+1e-30);
      for (i=0;i<N;i++)
         output[i] = sum*output[i];
#endif
   } else {
      celt_assert(activation == ACTIVATION_LINEAR);
      for (i=0;i<N;i++)
         output[i] = input[i];
   }
}

/**
 * Compute 'Linear-σ'.
 * Args:
 *   layer -
 *   output - address to which calculation results are written
 *   input -
 */
void _lpcnet_compute_dense(const DenseLayer *layer, float *output, const float *input)
{
   int i;
   int N, M;
   int stride;
   // M - dimension size of input
   M = layer->nb_inputs;
   // N - dimension size of output
   N = layer->nb_neurons;
   stride = N;
   celt_assert(input != output);
   // Assign bias to output for subsequent Multiply-Accumulation 
   for (i=0;i<N;i++)
      output[i] = layer->bias[i];
   // Linear (SGEMV)
   sgemv_accum(output, layer->input_weights, N, M, stride, input);
   // σ                                   <Enum of σ type>
   compute_activation(output, output, N, layer->activation);
}

int sample_mdense(const MDenseLayer *layer, const float *input, const float *sampling_logit_table, kiss99_ctx *rng)
{
   int b, j, N, M, C, stride;
   int val=0;
   float thresholds[8];
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   C = layer->nb_channels;
   celt_assert(N*C <= MAX_MDENSE_TMP);
   stride = M*C;
   
   celt_assert(N <= DUAL_FC_OUT_SIZE);

   /* Computing all the random thresholds in advance. These thresholds are directly
      based on the logit to avoid computing the sigmoid.*/
   // b0 ~ b7
   for (b=0;b<8;b+=4) {
       uint32_t r = kiss99_rand(rng);
       thresholds[b] = sampling_logit_table[r&0xFF];
       thresholds[b+1] = sampling_logit_table[(r>>8)&0xFF];
       thresholds[b+2] = sampling_logit_table[(r>>16)&0xFF];
       thresholds[b+3] = sampling_logit_table[(r>>24)&0xFF];
   }

   for (b=0;b<8;b++)
   {
      int bit;
      int i;
      float sum1, sum2;
      
      i = (1<<b) | val;

      sum1 = layer->bias[i];
      sum2 = layer->bias[i + N];
      for (j=0;j<M;j++) {
         sum1 += layer->input_weights[i*stride + j]*input[j];
         sum2 += layer->input_weights[i*stride + j + M]*input[j];
      }
      // FC1 = a1 ○ tanh(W1x)
      sum1 = layer->factor[i]*tanh_approx(sum1);
      // FC2 = a2 ○ tanh(W2x)
      sum2 = layer->factor[N + i]*tanh_approx(sum2);
      // o = FC1 + FC2
      sum1 += sum2;
      /*sum1 = 1.f/(1 + exp(-sum1));*/
#if 1 /* Sample the decision based on the logit. */
      bit = thresholds[b] < sum1;
#else
      sum1 = sigmoid_approx(sum1);
      bit = .025+.95*((rand()+.5f)/(RAND_MAX+1.f)) < sum1;
#endif
      val = (val << 1) | bit;
   }
   return val;

}


void compute_gruB(const GRULayer *gru, const float* gru_b_condition, float *state, const float *input)
{
   int i;
   int N, M;
   int stride;
   float zrh[3*MAX_RNN_NEURONS];
   float recur[3*MAX_RNN_NEURONS];
   float *z;
   float *r;
   float *h;
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2*N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   stride = 3*N;
   /* Compute update gate. */
#ifdef USE_SU_BIAS
   for (i=0;i<3*N;i++)
      zrh[i] = gru->subias[i] + gru_b_condition[i];
#else
   for (i=0;i<3*N;i++)
      zrh[i] = gru->bias[i] + gru_b_condition[i];
#endif
   sparse_sgemv_accum8x4(zrh, gru->input_weights, 3*N, M, gru->input_weights_idx, input);
#ifdef USE_SU_BIAS
   for (i=0;i<3*N;i++)
      recur[i] = gru->subias[3*N + i];
#else
   for (i=0;i<3*N;i++)
      recur[i] = gru->bias[3*N + i];
#endif
   sgemv_accum8x4(recur, gru->recurrent_weights, 3*N, N, stride, state);
   for (i=0;i<2*N;i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID);
   for (i=0;i<N;i++)
      h[i] += recur[2*N+i]*r[i];
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<N;i++)
      state[i] = h[i];
}


/* The input of this GRU is after the input matrix multiply. */
void compute_sparse_gru(const SparseGRULayer *gru, float *state, const float *input)
{
   int i, k;
   int N;
   float recur[3*MAX_RNN_NEURONS];
   float *z;
   float *r;
   float *h;
   const float *bias;
   N = gru->nb_neurons;
   z = recur;
   r = &recur[N];
   h = &recur[2*N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
#ifdef USE_SU_BIAS
   bias = &gru->subias[3*N];
#else
   bias = &gru->bias[3*N];   
#endif
   for (k=0;k<2;k++)
   {
      for (i=0;i<N;i++)
         recur[k*N + i] = bias[k*N + i] + gru->diag_weights[k*N + i]*state[i] + input[k*N + i];
   }
   for (;k<3;k++)
   {
      for (i=0;i<N;i++)
         recur[k*N + i] = bias[k*N + i] + gru->diag_weights[k*N + i]*state[i];
   }
   sparse_sgemv_accum8x4(recur, gru->recurrent_weights, 3*N, N, gru->idx, state);
   compute_activation(recur, recur, 2*N, ACTIVATION_SIGMOID);
   for (i=0;i<N;i++)
      h[i] = h[i]*r[i] + input[2*N+i];
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      state[i] = z[i]*state[i] + (1-z[i])*h[i];
}

void compute_conv1d(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   int i;
   int N, M;
   int stride;
   float tmp[MAX_CONV_INPUTS];
   celt_assert(input != output);
   celt_assert(layer->nb_inputs*layer->kernel_size <= MAX_CONV_INPUTS);
   RNN_COPY(tmp, mem, layer->nb_inputs*(layer->kernel_size-1));
   RNN_COPY(&tmp[layer->nb_inputs*(layer->kernel_size-1)], input, layer->nb_inputs);
   M = layer->nb_inputs*layer->kernel_size;
   N = layer->nb_neurons;
   stride = N;
   for (i=0;i<N;i++)
      output[i] = layer->bias[i];
   sgemv_accum(output, layer->input_weights, N, M, stride, tmp);
   compute_activation(output, output, N, layer->activation);
   RNN_COPY(mem, &tmp[layer->nb_inputs], layer->nb_inputs*(layer->kernel_size-1));
}

void compute_embedding(const EmbeddingLayer *layer, float *output, int input)
{
   int i;
   celt_assert(input >= 0);
   celt_assert(input < layer->nb_inputs);
   /*if (layer->dim == 64) printf("%d\n", input);*/
   for (i=0;i<layer->dim;i++)
   {
      output[i] = layer->embedding_weights[input*layer->dim + i];
   }    
}

void compute_gru_a_input(float *output, const float *input, int N, const EmbeddingLayer *layer1, int val1, const EmbeddingLayer *layer2, int val2, const EmbeddingLayer *layer3, int val3) {
   int i;
   for (i=0;i<3*N;i++) {
      output[i] = input[i] + layer1->embedding_weights[val1*layer1->dim + i]
                           + layer2->embedding_weights[val2*layer2->dim + i]
                           + layer3->embedding_weights[val3*layer3->dim + i];
   }
}
