/* Copyright (c) 2017-2018 Mozilla */
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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "freq.h"
#include "pitch.h"
#include "arch.h"
#include <assert.h>
#include "lpcnet.h"
#include "lpcnet_private.h"


/**
 * Args:
 *   y - Output
 */
static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

/**
 * Sampling from U[-0.5, +0.5]
 */
static float uni_rand() {
  // [0, RAND_MAX] -> [0, 1] -> [-0.5, +0.5]
  return rand()/(double)RAND_MAX-.5;
}

/**
 * 4 samplings from U[-0.375, +0.375]
 */
static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

void compute_noise(int *noise, float noise_std) {
  int i;
  for (i=0;i<FRAME_SIZE;i++) {
    //                               log(~[0, +∞))
    // = floor(0.5 + noise_std*.707*log(~U[0, 1]/~U[0, 1]))
    noise[i] = (int)floor(.5 + noise_std*.707*(log_approx((float)rand()/RAND_MAX)-log_approx((float)rand()/RAND_MAX)));
  }
}

/**
 * Convert float value to Int16 with ... and clipping
 */
static short float2short(float x)
{
  int i;
  i = (int)floor(.5+x);
  return IMAX(-32767, IMIN(32767, i));
}


/**
 * Write sample series to the file with noise addition.
 *
 * Args:
 *   st - State containing 'frame feature' and 'samples over loop'
 *   s_t_clean - Sample t series (waveform) without noise
 *   noise - Noise series for argumentation of training sample series
 *   file - File pointer of output
 *   nframes - The number of frames which `s_t_clean` and `noise` chunks contain
 */
void write_audio(LPCNetEncState *st, const short *s_t_clean, const int *noise, FILE *file, int nframes) {
  int t, idx_f;
  int OFFSET_COEFF = NB_BANDS + 2 // BFCC+pitches

  for (idx_f=0; idx_f<nframes; idx_f++) {
    /* Processing of single frame */

    // series of s_{t-1} (lagged/delayed) & s_t, linearlized
    short s_t_1_s_t_series[2*FRAME_SIZE]; // Write buffer
    int offset_frame = idx_f * FRAME_SIZE

    for (t=0; t<FRAME_SIZE; t++) {
      /* Process a sample t in the frame */

      float p_t_noisy = 0; // Prediction, noise added
      int j;               // LP order index

      int idx_t = t + offset_frame // t_in_frame + frame_offset

      /* Linear Prediction */
      //                                                                 a_{j+1} * s_{t-(j+1)}_noisy
      for (j=0;j<LPC_ORDER;j++) p_t_noisy -= st->features[idx_f][j+OFFSET_COEFF] * st->sig_mem[j];

      /* LP Residual t=T */
      float e_t_ideal = lin2ulaw(s_t_clean[idx_t] - p_t_noisy);

      /* Sample t=T-1 (lagged/delayed) with noise */
      float s_t_1_noisy = st->sig_mem[0]
      s_t_1_s_t_series[2*t] = float2short(s_t_1_noisy);

      /* Sample t=T without noise */
      s_t_1_s_t_series[2*t+1] = s_t_clean[idx_t];

      /* Noise addition */
      // Derivatives: Noise source at residual (at sample @original -> at residual from @b858ea9)
      float e_t_noisy = e_t_ideal + noise[idx_t];
      e_t_noisy = IMIN(255, IMAX(0, e_t_noisy));
      float s_t_noisy = p_t_noisy + ulaw2lin(e_t_noisy);

      /* State update over loop */
      //// Update t-2 ~ t-LPC_ORDER : sig_mem[1:] = sig_mem[0:LPC_ORDER-1]
      RNN_MOVE(&st->sig_mem[1], &st->sig_mem[0], LPC_ORDER-1);
      //// Update t-1 (s_t_1_noisy)
      st->sig_mem[0] = s_t_noisy

      // EXCitation_MEMory (Not used...?)
      st->exc_mem = e_t_noisy;
    }

    // Append `s_t_1_s_t_series` of a frame into `file` file
    fwrite(s_t_1_s_t_series, 4*FRAME_SIZE, 1, file);
  }
}


int main(int argc, char **argv) {
  int i;
  char *argv0;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_resp_x[2]={0};
  float mem_preemph=0;       // preemphasis memory over frames
  float x[FRAME_SIZE];       // samples of a frame
  int gain_change_count=0;
  FILE *f1;        // Input  file containing sample series (waveform) <input.s16>
  FILE *ffeat;     // Output file containing              <features.f32>
  FILE *fpcm=NULL; // Output file containing              <data.s16>
  short s_frame_clean[FRAME_SIZE]={0}; // samples of a frame
  short s_4frames_clean[FRAME_SIZE*4]={0};      // samples of 4 frames
  int noisebuf[FRAME_SIZE*4]={0};
  short tmp[FRAME_SIZE] = {0};
  float savedX[FRAME_SIZE] = {0};
  float speech_gain=1;
  int last_silent = 1;
  float old_speech_gain = 1;
  int one_pass_completed = 0;
  LPCNetEncState *st;
  float noise_std=0;
  // `training`, `encode`, `decode`, `quantize`
  int training = -1; // -1 | 0 | 1, should be updated to 0|1 based on arguments
  int encode = 0;
  int decode = 0;
  int quantize = 0;
  srand(getpid());
  st = lpcnet_encoder_create();
  argv0=argv[0];

  // train: `-train <input.s16> <features.f32> <data.s16>`
  if (argc == 5 && strcmp(argv[1], "-train")==0) training = 1;
  if (argc == 5 && strcmp(argv[1], "-qtrain")==0) {
      training = 1;
      quantize = 1;
  }
  // test/inference (== not train)
  if (argc == 4 && strcmp(argv[1], "-test")==0) training = 0;
  if (argc == 4 && strcmp(argv[1], "-qtest")==0) {
      training = 0;
      quantize = 1;
  }
  if (argc == 4 && strcmp(argv[1], "-encode")==0) {
      training = 0;
      quantize = 1;
      encode = 1;
  }
  if (argc == 4 && strcmp(argv[1], "-decode")==0) {
      training = 0;
      decode = 1;
  }

  // Validation and file open
  if (training == -1) {
    fprintf(stderr, "usage: %s -train <speech> <features out> <pcm out>\n", argv0);
    fprintf(stderr, "  or   %s -test <speech> <features out>\n", argv0);
    return 1;
  }
  //// input sample series (waveform) file <input.s16>
  f1 = fopen(argv[2], "r");
  if (f1 == NULL) {
    fprintf(stderr,"Error opening input .s16 16kHz speech input file: %s\n", argv[2]);
    exit(1);
  }
  //// output file for ... <features.f32>
  ffeat = fopen(argv[3], "wb");
  if (ffeat == NULL) {
    fprintf(stderr,"Error opening output feature file: %s\n", argv[3]);
    exit(1);
  }
  if (decode) {
    float vq_mem[NB_BANDS] = {0};
    while (1) {
      int ret;
      unsigned char buf[8];
      float features[4][NB_TOTAL_FEATURES];
      /*int c0_id, main_pitch, modulation, corr_id, vq_end[3], vq_mid, interp_id;*/
      /*ret = fscanf(f1, "%d %d %d %d %d %d %d %d %d\n", &c0_id, &main_pitch, &modulation, &corr_id, &vq_end[0], &vq_end[1], &vq_end[2], &vq_mid, &interp_id);*/
      ret = fread(buf, 1, 8, f1);
      if (ret != 8) break;
      decode_packet(features, vq_mem, buf);
      for (i=0;i<4;i++) {
        fwrite(features[i], sizeof(float), NB_TOTAL_FEATURES, ffeat);
      }
    }
    return 0;
  }
  if (training) {
    //// output file for ... <data.s16>
    fpcm = fopen(argv[4], "wb");
    if (fpcm == NULL) {
      fprintf(stderr,"Error opening output PCM file: %s\n", argv[4]);
      exit(1);
    }
  }

  while (1) {
    // Loop single frame processing

    size_t ret;

    // Read samples of single frame from <input.s16> on `x`
    ////         len(x)
    for (i=0;i<FRAME_SIZE;i++) x[i] = tmp[i];
    ret = fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1) || ret != FRAME_SIZE) {
      if (!training) break;
      rewind(f1);
      ret = fread(tmp, sizeof(short), FRAME_SIZE, f1);
      if (ret != FRAME_SIZE) {
        fprintf(stderr, "error reading\n");
        exit(1);
      }
      one_pass_completed = 1;
    }

    // Silent clipping (disabled @5627af3)
    float E=0;
    int silent;
    for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];
    if (0 && training) {
      silent = E < 5000 || (last_silent && E < 20000);
      if (!last_silent && silent) {
        for (i=0;i<FRAME_SIZE;i++) savedX[i] = x[i];
      }
      if (last_silent && !silent) {
          for (i=0;i<FRAME_SIZE;i++) {
            float f = (float)i/FRAME_SIZE;
            tmp[i] = (int)floor(.5 + f*tmp[i] + (1-f)*savedX[i]);
          }
      }
      if (last_silent) {
        last_silent = silent;
        continue;
      }
      last_silent = silent;
    }

    // Loop escape
    //// For too small data, loop more than 2 passes
    //// if      enough_data           && >=1_pass_completed
    if (count*FRAME_SIZE_5MS>=10000000 && one_pass_completed) break;

    // Parameters
    if (training && ++gain_change_count > 2821) {
      float tmp, tmp2;

      // speech_gain
      speech_gain = pow(10., (-20+(rand()%40))/20.);
      if (rand()%20==0) speech_gain *= .01;
      if (rand()%100==0) speech_gain = 0;
      gain_change_count = 0;

      // ~U[-0.375, +0.375]
      rand_resp(a_sig, b_sig);

      // Noise parameter `noise_std` for training sample series augmentation
      //// tmp, tmp2 ~ U[0, 1]
      tmp = (float)rand()/RAND_MAX;
      tmp2 = (float)rand()/RAND_MAX;
      noise_std = ABS16(-1.5*log(1e-4+tmp)-.5*log(1e-4+tmp2));
    }

    // In-place ??
    biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);

    // In-place Preemphasis
    preemphasis(x, &mem_preemph, x, PREEMPHASIS, FRAME_SIZE);

    // In-place Gain
    ////         len(x)
    for (i=0;i<FRAME_SIZE;i++) {
      float g;
      float f = (float)i/FRAME_SIZE;
      // Gain smoothing
      g = f*speech_gain + (1-f)*old_speech_gain;
      x[i] *= g;
    }

    // (maybe) In-place noise addition ~ U[-0.5, +0.5]
    for (i=0;i<FRAME_SIZE;i++) x[i] += rand()/(float)RAND_MAX - .5;

    /* PCM is delayed by 1/2 frame to make the features centered on the frames. */
    for (i=0;i<FRAME_SIZE-TRAINING_OFFSET;i++) s_frame_clean[i+TRAINING_OFFSET] = float2short(x[i]);



    compute_frame_features(st, x);

    // 
    int frame_start = st->pcount*FRAME_SIZE
    // Stack a frame into the buffer for 4 frame grouped processing mode
    RNN_COPY(&s_4frames_clean[frame_start], s_frame_clean, FRAME_SIZE);

    /* ?? Noise */
    if (fpcm) { // `if (fpcm)` check for non-train mode
        // Noise for training sample series augmentation
        compute_noise(&noisebuf[frame_start], noise_std);
    }

    // A. non-quantize
    if (!quantize) {
      process_single_frame(st, ffeat);
      // Modify samples and Dump them into the `fpcm` file
      if (fpcm) write_audio(st, s_frame_clean, &noisebuf[frame_start], fpcm, 1);
    }
    st->pcount++;
    /* Running on groups of 4 frames. */
    if (st->pcount == 4) {
      // B. quantize
      if (quantize) {
        unsigned char buf[8];
        process_superframe(st, buf, ffeat, encode, quantize);
        if (fpcm) write_audio(st, s_4frames_clean, noisebuf, fpcm, 4);
      }
      // counter reset
      st->pcount = 0;
    }
    /* ================================================================================================== */


    for (i=0;i<TRAINING_OFFSET;i++) s_frame_clean[i] = float2short(x[i+FRAME_SIZE-TRAINING_OFFSET]);

    // Gain trainsition
    old_speech_gain = speech_gain;

    // Increment the number of processed frames
    count++;
  }

  // Termination (, LPCNet)
  fclose(f1);
  fclose(ffeat);
  if (fpcm) fclose(fpcm);
  lpcnet_encoder_destroy(st);
  return 0;
}

