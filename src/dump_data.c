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
 * ✅ Biquadratic filter (2nd-order feed-forward & 2nd-order feed-back IIR)
 *
 * Args:
 *   y   - Output series
 *   mem - y_{i-1} and y_{i-2} memory over loop
 *   x   - Input series
 *   b   - Feedforward parameter
 *   a   - Feedback    parameter
 *   N   - Length of input series `x` and output series `y`
 */
static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi; // Input and Output
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi); // (b_1 * x_{i-2} - a_1 * y_{i-2}) + (b_0 * x_{i-1} - a_0 * y_{i-1})
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

/**
 * ✅ Sampling from U[-0.5, +0.5]
 */
static float uni_rand() {
  // [0, RAND_MAX] -> [0, 1] -> [-0.5, +0.5]
  return rand()/(double)RAND_MAX-.5;
}

/**
 * ✅ 4 samplings from U[-0.375, +0.375]
 */
static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

/* △ Compute noise for s_t_1_noisy */
void compute_noise(int *noise, float noise_std) {
  int i;
  for (i=0;i<FRAME_SIZE;i++) {
    //                        (1/√2)*log(~[0, +∞))
    // = floor(0.5 + noise_std*.707*log(~U[0, 1]/~U[0, 1]))
    noise[i] = (int)floor(.5 + noise_std*.707*(log_approx((float)rand()/RAND_MAX)-log_approx((float)rand()/RAND_MAX)));
  }
}

/*
    noise[i] = short(std/√2 * (lnU[0,1] - lnU[0,1]))
*/

/**
 * ✅ Cast fp32 to int16
 */
static short float2short(float x_fp32)
{
  int i;
  i = (int)floor(.5+x_fp32);
  return IMAX(-32767, IMIN(32767, i));
}


/**
 * ✅ Write two sample series (s_t_1_noisy & s_t_clean_s16) to the file.
 * 
 * Args:
 *   st            - State containing 'LP coefficients' and 'samples over loop'
 *   s_t_clean_s16 - Sample t series (signed int16 waveform) without noise
 *   noise         - Noise series for s_t_1_noisy argumentation
 *   file          - Output file pointer
 */
void write_audio(LPCNetEncState *st, const short *s_t_clean_s16, const int *noise, FILE *file) {

  // series of s_{t-1}_noisy (lagged/delayed) & s_t_clean, linearlized, signed int16
  short s_t_1_s_t_s16[2*FRAME_SIZE];

  int t;
  for (t=0; t<FRAME_SIZE; t++) {
    /* Process a sample t in the frame */

    float p_t_noisy_fp32 = 0; // Prediction, noise added

    /* Linear Prediction - prediction from noisy samples */
    //                                                             a_{j+1} * s_{t-(j+1)}_noisy
    int j; //                                                    BFCC+pitches
    for (j=0;j<LPC_ORDER;j++) p_t_noisy_fp32 -= st->features[0][j+NB_BANDS+2] * st->sig_mem[j];

    /* Ideal LP Residual - Ideal residual inference under erroneous (noisy) previous samples */
    float e_t_ideal_fp32 = lin2ulaw(s_t_clean_s16[t] - p_t_noisy_fp32);

    /* Sample t=T-1 (lagged/delayed) with noise */
    float s_t_1_noisy_fp32 = st->sig_mem[0];
    s_t_1_s_t_s16[2*t] = float2short(s_t_1_noisy_fp32);

    /* Copy s_t_clean_s16[t] */
    s_t_1_s_t_s16[2*t+1] = s_t_clean_s16[t];

    /* Noise addition - Emulate the situation 'Try to yield ideal e_t under erroneous samples, but has some error in the e_t' */
    // Derivatives: Noise source at residual (at sample @original -> at residual from @b858ea9)
    float e_t_noisy = e_t_ideal_fp32 + noise[t];
    e_t_noisy = IMIN(255, IMAX(0, e_t_noisy));
    float s_t_noisy_fp32 = p_t_noisy_fp32 + ulaw2lin(e_t_noisy);

    /* State update over loop */
    //// Update s_t_x_noisy's {t-2} ~ {t-LPC_ORDER} (sig_mem[1:] = sig_mem[0:LPC_ORDER-1])
    RNN_MOVE(&st->sig_mem[1], &st->sig_mem[0], LPC_ORDER-1);
    //// Update t-1 (s_t_1_noisy)
    st->sig_mem[0] = s_t_noisy_fp32;
  }

  // Append `s_t_1_s_t_s16` into the file (2 [series] * 2[byte] * FRAME_SIZE [samples])
  fwrite(s_t_1_s_t_s16, 4*FRAME_SIZE, 1, file);
}


int main(int argc, char **argv) {
  int i;
  char *argv0;
  int frame_count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_resp_x[2]={0};
  float mem_preemph=0;       // preemphasis memory over frames
  float x_fp32[FRAME_SIZE];  // samples of a frame. Basically, [-2**15, +2**15]
  int gain_change_count=0;
  FILE *f1;        // Input  file containing sample series (waveform) <input.s16>
  FILE *ffeat;     // Output file containing              <features.f32>
  FILE *fpcm=NULL; // Output file containing              <data.s16>
  short s_t_clean_s16[FRAME_SIZE]={0}; // samples of a frame, signed int16
  short tmp_s16[FRAME_SIZE] = {0};
  float speech_gain=1;
  float old_speech_gain = 1;
  int one_pass_completed = 0;
  LPCNetEncState *st; // State containing 'frame feature' and 'samples' over loop
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
      fprintf(stderr, "-qtrain is disabled.");
      return 1;
  }
  // test/inference (== not train)
  if (argc == 4 && strcmp(argv[1], "-test")==0) training = 0;
  if (argc == 4 && strcmp(argv[1], "-qtest")==0) {
      training = 0;
      quantize = 1;
      fprintf(stderr, "-qtest is disabled.");
      return 1;
  }
  if (argc == 4 && strcmp(argv[1], "-encode")==0) {
      training = 0;
      quantize = 1;
      encode = 1;
      fprintf(stderr, "-encode is disabled.");
      return 1;
  }
  if (argc == 4 && strcmp(argv[1], "-decode")==0) {
      training = 0;
      decode = 1;
      fprintf(stderr, "-decode is disabled.");
      return 1;
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
  //// output file for Feature series <features.f32>
  ffeat = fopen(argv[3], "wb");
  if (ffeat == NULL) {
    fprintf(stderr,"Error opening output feature file: %s\n", argv[3]);
    exit(1);
  }
  //// output file for Waveforms <data.s16>
  if (training) {
    fpcm = fopen(argv[4], "wb");
    if (fpcm == NULL) {
      fprintf(stderr,"Error opening output PCM file: %s\n", argv[4]);
      exit(1);
    }
  }

  while (1) {
    // Frame-wise preprocessing

    size_t ret;


    /* ==== Data Load =================================================================================== */
    // Read samples of single frame from <input.s16> on `x_fp32`
    ////         len(x_fp32)               `tmp_s16` is initialized with 0
    for (i=0;i<FRAME_SIZE;i++) x_fp32[i] = tmp_s16[i]; // implicit cast?

    // Load Int16 samples of a frame from <input.s16> onto `tmp_s16` for next loop...?
    ret = fread(tmp_s16, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1) || ret != FRAME_SIZE) {
      if (!training) break;
      // N-th pass from frame#0
      rewind(f1);
      ret = fread(tmp_s16, sizeof(short), FRAME_SIZE, f1);
      if (ret != FRAME_SIZE) {
        // Even first frame do not have enough length, something wrong.
        fprintf(stderr, "error reading\n");
        exit(1);
      }
      one_pass_completed = 1;
    }


    /* ==== Loop escape ================================================================================= */
    // For too small data, loop more than 2 passes
    // if             enough_data            && >=1_pass_completed
    if (frame_count*FRAME_SIZE_5MS>=10000000 && one_pass_completed) break;


    /* ==== Parameter generation ======================================================================== */
    //   Updated once per 2821 frames
    //   todo: Where dose the magic number 2821 come from ...?
    if (training && ++gain_change_count > 2821) {

      /* ✅ `a_sig` & `b_sig` - SpecAug's 2nd-order feedback/forward params ~U[-0.375, +0.375] */
      rand_resp(a_sig, b_sig);

      /* ✅ `speech_gain` - Gain factor at frame end */
      speech_gain = pow(10., (-20+(rand()%40))/20.);
      if (rand()%20==0) speech_gain *= .01;
      if (rand()%100==0) speech_gain = 0;

      /* △ `noise_std` - Noise for s_t_1_noisy */
      float tmp1 = (float)rand()/RAND_MAX; // ~ U[0, 1]
      float tmp2 = (float)rand()/RAND_MAX; // ~ U[0, 1]
      noise_std = ABS16(-1.5*log(1e-4 + tmp1) - .5*log(1e-4 + tmp2));

      /* ✅ Reset parameter change count */
      gain_change_count = 0;
    }


    /* ==== Augmentation ================================================================================ */
    // Augment `x_fp32`, samples of a frame, by "Equalizer -> Preemphasis -> Gain -> NoiseAddition" in place.

    /* SpecAug - biquad Equalizers */
    // Derivatives: Random spectral augmentation (OFF @original -> ON @efficiency from @396274f)
    // ✅ Fixed High-pass filter
    biquad(x_fp32, mem_hp_x,   x_fp32, b_hp,   a_hp, FRAME_SIZE);
    // ✅ Random equalizer
    biquad(x_fp32, mem_resp_x, x_fp32, b_sig, a_sig, FRAME_SIZE);

    /* ✅ Preemphasis - Preemphasize with coeff PREEMPHASIS==0.85 */
    preemphasis(x_fp32, &mem_preemph, x_fp32, PREEMPHASIS, FRAME_SIZE);

    /* ✅ Gain - Sample-wise gain with smooth gain transition */
    for (i=0;i<FRAME_SIZE;i++) {
      float g; // gain of a sample
      float f = (float)i/FRAME_SIZE; // ratio in a frame
      // Gain smoothing
      g = f*speech_gain + (1-f)*old_speech_gain;
      x_fp32[i] *= g;
    }

    /* △ Noise addition - Sample-wise noise addition ~ U[-0.5, +0.5] */
    // todo: It looks strange. x will be used for clean target, so does it make output noisy...?
    //       But `x_fp32` is basically [-2**15, +2**15], so effect is super tiny. what's for ...?
    //       Aliasing something ...?
    for (i=0;i<FRAME_SIZE;i++) x_fp32[i] += rand()/(float)RAND_MAX - .5;


    /* ==== ✅ Shift ==================================================================================== */
    /* PCM is shifted to make the features centered on the frames. */
    /*
                       t                     t+F-ost       t+F
       augumented x  --|------------------------------------|
                       |__________________________|
                                         ⤵
                                (t)                     (t+F-ost)
                                 |__________________________|
      s_t_clean_s16    |------------------------------------|
                    (t-ost)     (t)
    */
    for (i=0;i<FRAME_SIZE-TRAINING_OFFSET;i++) s_t_clean_s16[i+TRAINING_OFFSET] = float2short(x_fp32[i]);


    /* ==== Feature-nize ================================================================================ */

    /* Feature extraction - Calculate BFC and LPCoeff from non-shifted sample series (x) */
    compute_frame_features(st, x_fp32);
    /* Pitch generation and Dump - Calculate pitches and Dump full `.features` into the `ffeat` file */
    process_single_frame(st, ffeat);

    /* △ Sample series (s_t_1_noisy & s_t_clean) augmentation with noise and Dump for training */
    if (fpcm) {
      int noisebuf[FRAME_SIZE]={0};
      compute_noise(&noisebuf[0], noise_std);
      write_audio(st, s_t_clean_s16, &noisebuf[0], fpcm);
    }


    /* ==== ✅ Shift =================================================================================== */
    /* Shift remainings
                       t                     t+F-ost       t+F
       augumented x  --|------------------------------------|--
                                                  |_________|
                                  ←___________________/
                   (t+F-ost)  (t+F)
                       |________|
      s_t_clean_s16    |------------------------------------|
                                 (t)                    (t+F-ost)
    */
    for (i=0;i<TRAINING_OFFSET;i++) s_t_clean_s16[i] = float2short(x_fp32[i+FRAME_SIZE-TRAINING_OFFSET]);


    /* ==== ✅ State updates =========================================================================== */

    /* Gain - Gain factor at next frame start */
    old_speech_gain = speech_gain;

    // Frame Count - Increment the number of processed frames
    frame_count++;

    /* Frame Count - Increment frame counting for supra-frame processings */
    st->pcount++;
    if (st->pcount == 4) {
      st->pcount = 0;
    }
    /* ================================================================================================== */
    /* ================================================================================================== */
  }

  /* Termination */
  fclose(f1);
  fclose(ffeat);
  if (fpcm) fclose(fpcm);
  lpcnet_encoder_destroy(st);

  return 0;
}

