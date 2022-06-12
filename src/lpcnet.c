/* Copyright (c) 2018 Mozilla */
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdio.h>
#include "nnet_data.h"
#include "nnet.h"
#include "common.h"
#include "arch.h"
#include "lpcnet.h"
#include "lpcnet_private.h"

#define PREEMPH 0.85f

#define PDF_FLOOR 0.002

#define FRAME_INPUT_SIZE (NB_FEATURES + EMBED_PITCH_OUT_SIZE)


#ifdef END2END
void rc2lpc(float *lpc, const float *rc)
{
  int i, j, k;
  float tmp[LPC_ORDER];
  float ntmp[LPC_ORDER] = {0.0};
  RNN_COPY(tmp, rc, LPC_ORDER);
  for(i = 0; i < LPC_ORDER ; i++)
    { 
        for(j = 0; j <= i-1; j++)
        {
            ntmp[j] = tmp[j] + tmp[i]*tmp[i - j - 1];
        }
        for(k = 0; k <= i-1; k++)
        {
            tmp[k] = ntmp[k];
        }
    }
  for(i = 0; i < LPC_ORDER ; i++)
  {
    lpc[i] = tmp[i];
  }
}
#endif


/**
 * Run 'frame rate network'.
 *
 * <in>-Conv1d-Conv1d-Linear-σ-Linear-σ---Linear-σ-<gru_a_condition>
 *                                      |-Linear-σ-<gru_b_condition>
 *
 * Args:
 *   gru_a_condition - Output adress which will be used for conditioning input of GRUa (?)
 *   gru_b_condition - Output adress which will be used for conditioning input of GRUb (?)
 */
void run_frame_network(LPCNetState *lpcnet, float *gru_a_condition, float *gru_b_condition, float *lpc, const float *features)
{
    NNetState *net;
    float condition[FEATURE_DENSE2_OUT_SIZE];
    float in[FRAME_INPUT_SIZE];
    float conv1_out[FEATURE_CONV1_OUT_SIZE];
    float conv2_out[FEATURE_CONV2_OUT_SIZE];
    float dense1_out[FEATURE_DENSE1_OUT_SIZE];
    int pitch;
    float rc[LPC_ORDER];
    /* Matches the Python code -- the 0.1 avoids rounding issues. */
    pitch = (int)floor(.1 + 50*features[NB_BANDS]+100);
    pitch = IMIN(255, IMAX(33, pitch));
    net = &lpcnet->nnet;
    RNN_COPY(in, features, NB_FEATURES);
    compute_embedding(&embed_pitch, &in[NB_FEATURES], pitch);
    // <in>-Conv1d?-<conv1_out>
    compute_conv1d(&feature_conv1, conv1_out, net->feature_conv1_state, in);
    if (lpcnet->frame_count < FEATURE_CONV1_DELAY) RNN_CLEAR(conv1_out, FEATURE_CONV1_OUT_SIZE);
    // <conv1_out>-Conv1d?-<conv2_out>
    compute_conv1d(&feature_conv2, conv2_out, net->feature_conv2_state, conv1_out);
    celt_assert(FRAME_INPUT_SIZE == FEATURE_CONV2_OUT_SIZE);
    if (lpcnet->frame_count < FEATURES_DELAY) RNN_CLEAR(conv2_out, FEATURE_CONV2_OUT_SIZE);
    // <conv2_out>-Linear-σ-<dense1_out>
    _lpcnet_compute_dense(&feature_dense1, dense1_out, conv2_out);
    // <dense1_out>-Linear-σ-<condition>
    _lpcnet_compute_dense(&feature_dense2, condition, dense1_out);
    RNN_COPY(rc, condition, LPC_ORDER);
    // <condition>-Linear-σ-<gru_a_condition>
    _lpcnet_compute_dense(&gru_a_dense_feature, gru_a_condition, condition);
    // <condition>-Linear-σ-<gru_b_condition>
    _lpcnet_compute_dense(&gru_b_dense_feature, gru_b_condition, condition);
#ifdef END2END
    rc2lpc(lpc, rc);
#elif FEATURES_DELAY>0    
    memcpy(lpc, lpcnet->old_lpc[FEATURES_DELAY-1], LPC_ORDER*sizeof(lpc[0]));
    memmove(lpcnet->old_lpc[1], lpcnet->old_lpc[0], (FEATURES_DELAY-1)*LPC_ORDER*sizeof(lpc[0]));
    lpc_from_cepstrum(lpcnet->old_lpc[0], features);
#else
    lpc_from_cepstrum(lpc, features);
#endif
    if (lpcnet->frame_count < 1000) lpcnet->frame_count++;
}


/**
 * Run 'sample rate network' (AR residual generation).
 *
 */
int run_sample_network(NNetState *net, const float *gru_a_condition, const float *gru_b_condition, int last_exc, int last_sig, int pred, const float *sampling_logit_table, kiss99_ctx *rng)
{
    float gru_a_input[3*GRU_A_STATE_SIZE];
    float in_b[GRU_A_STATE_SIZE+FEATURE_DENSE2_OUT_SIZE];
    float gru_b_input[3*GRU_B_STATE_SIZE];
    compute_gru_a_input(gru_a_input, gru_a_condition, GRU_A_STATE_SIZE, &gru_a_embed_sig, last_sig, &gru_a_embed_pred, pred, &gru_a_embed_exc, last_exc);
    compute_sparse_gru(&sparse_gru_a, net->gru_a_state, gru_a_input);
    RNN_COPY(in_b, net->gru_a_state, GRU_A_STATE_SIZE);
    RNN_COPY(gru_b_input, gru_b_condition, 3*GRU_B_STATE_SIZE);
    compute_gruB(&gru_b, gru_b_input, net->gru_b_state, in_b);

    // Derivatives from original LPCNet (proposed in `lpcnet_efficiency` from @d24f49e):
    //   - Add "binary probability tree"
    //     - Remove pitch correlation based voicing decision / probability modification
    return sample_mdense(&dual_fc, net->gru_b_state, sampling_logit_table, rng);
}


// #### Network state handling ###############################################################
LPCNET_EXPORT int lpcnet_get_size()
{
    return sizeof(LPCNetState);
}

LPCNET_EXPORT int lpcnet_init(LPCNetState *lpcnet)
{
    int i;
    const char* rng_string="LPCNet";
    memset(lpcnet, 0, lpcnet_get_size());
    lpcnet->last_exc = lin2ulaw(0.f);
    for (i=0;i<256;i++) {
        float prob = .025+.95*i/255.;
        lpcnet->sampling_logit_table[i] = -log((1-prob)/prob);
    }
    kiss99_srand(&lpcnet->rng, (const unsigned char *)rng_string, strlen(rng_string));
    return 0;
}

LPCNET_EXPORT LPCNetState *lpcnet_create()
{
    LPCNetState *lpcnet;
    lpcnet = (LPCNetState *)calloc(lpcnet_get_size(), 1);
    lpcnet_init(lpcnet);
    return lpcnet;
}

LPCNET_EXPORT void lpcnet_destroy(LPCNetState *lpcnet)
{
    free(lpcnet);
}
// ######################################################################################


/**
 * Synthesize 1-frame-equivalent waveform.
 *
 * Args:
 *   output - memory to which generated samples (waveform) will be written
 *   N - the number of samples to be generated (the number of 'sample rate network' loop)
 */
void lpcnet_synthesize_tail_impl(LPCNetState *lpcnet, short *output, int N, int preload)
{
    int i;

    if (lpcnet->frame_count <= FEATURES_DELAY)
    {
        RNN_CLEAR(output, N);
        return;
    }

   /* Loop SampleRateNetwork `N` times */
    for (i=0;i<N;i++)
    {
        // LPC order index
        int j;
        // generated sample
        float pcm;
        int exc;
        int last_sig_ulaw;
        // generated residual (μ-law)
        int pred_ulaw;

       /* `LPC_ORDER`-th order Linear Prediction */
        float pred = 0;
        for (j=0;j<LPC_ORDER;j++) pred -= lpcnet->last_sig[j]*lpcnet->lpc[j];
        last_sig_ulaw = lin2ulaw(lpcnet->last_sig[0]);
        pred_ulaw = lin2ulaw(pred);

        /* Residual */
        exc = run_sample_network(&lpcnet->nnet, lpcnet->gru_a_condition, lpcnet->gru_b_condition, lpcnet->last_exc, last_sig_ulaw, pred_ulaw, lpcnet->sampling_logit_table, &lpcnet->rng);
        if (i < preload) exc = lin2ulaw(output[i]-PREEMPH*lpcnet->deemph_mem - pred);

        /* Sum */
        pcm = pred + ulaw2lin(exc);

        RNN_MOVE(&lpcnet->last_sig[1], &lpcnet->last_sig[0], LPC_ORDER-1);
        lpcnet->last_sig[0] = pcm;
        lpcnet->last_exc = exc;
        pcm += PREEMPH*lpcnet->deemph_mem;
        lpcnet->deemph_mem = pcm;

        /* Clip in audio range */
        if (pcm<-32767) pcm = -32767;
        if (pcm>32767) pcm = 32767;

        /* Write i-th sample into output adress */
        output[i] = (int)floor(.5 + pcm);
    }
}

/**
 * Synthesize 1-frame-equivalent waveform from features.
 *
 * Used for normal synthesis and PLC-synthesis (?)
 *
 * Args:
 *   output - memory to which generated samples (waveform) will be written
 *   N - the number of samples to be generated (the number of 'sample rate network' loop)
 */
void lpcnet_synthesize_impl(LPCNetState *lpcnet, const float *features, short *output, int N, int preload)
{
    // frame_rate_network :: features -> (lpcnet.gru_a_condition, lpcnet.gru_b_condition, lpcnet->lpc)
    run_frame_network(lpcnet, lpcnet->gru_a_condition, lpcnet->gru_b_condition, lpcnet->lpc, features);
    // feat2wave :: lpcnet(gru_a_condition/gru_b_condition/lpc) -> output::(N,)
    lpcnet_synthesize_tail_impl(lpcnet, output, N, preload);
}

/**
 *
 * Args:
 *   output - memory to which generated samples (waveform) will be written
 *   N - the number of samples to be generated (the number of 'sample rate network' loop)
 */
LPCNET_EXPORT void lpcnet_synthesize(LPCNetState *lpcnet, const float *features, short *output, int N) {
    lpcnet_synthesize_impl(lpcnet, features, output, N, 0);
}

LPCNET_EXPORT int lpcnet_decoder_get_size()
{
  return sizeof(LPCNetDecState);
}

LPCNET_EXPORT int lpcnet_decoder_init(LPCNetDecState *st)
{
  memset(st, 0, lpcnet_decoder_get_size());
  lpcnet_init(&st->lpcnet_state);
  return 0;
}

LPCNET_EXPORT LPCNetDecState *lpcnet_decoder_create()
{
  LPCNetDecState *st;
  st = malloc(lpcnet_decoder_get_size());
  lpcnet_decoder_init(st);
  return st;
}

LPCNET_EXPORT void lpcnet_decoder_destroy(LPCNetDecState *st)
{
  free(st);
}

LPCNET_EXPORT int lpcnet_decode(LPCNetDecState *st, const unsigned char *buf, short *pcm)
{
  int k;
  float features[4][NB_TOTAL_FEATURES];
  decode_packet(features, st->vq_mem, buf);
  for (k=0;k<4;k++) {
    lpcnet_synthesize(&st->lpcnet_state, features[k], &pcm[k*FRAME_SIZE], FRAME_SIZE);
  }
  return 0;
}

