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
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi); // (b_1 * x_{i-2} - a_1 * x_{i-2}) + (b_0 * x_{i-1} - a_0 * y_{i-1})
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
 * ✅ 4 samplings from U[-0.375, +0.375]
 */
static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

/* Compute noise for s_t_1_noisy */
void compute_noise(int *noise, float noise_std) {
  int i;
  for (i=0;i<FRAME_SIZE;i++) {
    //                        (1/√2)*log(~[0, +∞))
    // = floor(0.5 + noise_std*.707*log(~U[0, 1]/~U[0, 1]))
    noise[i] = (int)floor(.5 + noise_std*.707*(log_approx((float)rand()/RAND_MAX)-log_approx((float)rand()/RAND_MAX)));
  }
}

/**
 * Cast fp32 to int16
 */
static short float2short(float x)
{
  int i;
  i = (int)floor(.5+x);
  return IMAX(-32767, IMIN(32767, i));
}


/**
 * Write two sample series (s_t_1_noisy & s_t_clean) to the file.
 * 
 * Args:
 *   st - State containing 'LP coefficients' and 'samples over loop'
 *   s_t_clean - Sample t series (waveform) without noise
 *   noise - Noise series for s_t_1_noisy argumentation
 *   file - File pointer of output
 *   nframes - The number of frames which `s_t_clean` and `noise` chunks contain
 */
void write_audio(LPCNetEncState *st, const short *s_t_clean, const int *noise, FILE *file, int nframes) {
  int t, idx_f;
  int OFFSET_COEFF = NB_BANDS + 2; // BFCC+pitches

  for (idx_f=0; idx_f<nframes; idx_f++) {
    /* Processing of single frame (for nframes==1, just process once) */

    // series of s_{t-1} (lagged/delayed) & s_t, linearlized
    short s_t_1_s_t_series[2*FRAME_SIZE]; // Write buffer
    int offset_frame = idx_f * FRAME_SIZE;

    for (t=0; t<FRAME_SIZE; t++) {
      /* Process a sample t in the frame */

      float p_t_noisy = 0; // Prediction, noise added
      int j;               // LP order index

      int idx_t = t + offset_frame; // t_in_frame + frame_offset

      /* Linear Prediction - noisy prediction from noisy samples */
      //                                                                 a_{j+1} * s_{t-(j+1)}_noisy
      for (j=0;j<LPC_ORDER;j++) p_t_noisy -= st->features[idx_f][j+OFFSET_COEFF] * st->sig_mem[j];

      /* Ideal LP Residual - Ideal residual inference under erroneous (noisy) previous samples */
      float e_t_ideal = lin2ulaw(s_t_clean[idx_t] - p_t_noisy);

      /* Sample t=T-1 (lagged/delayed) with noise */
      float s_t_1_noisy = st->sig_mem[0];
      s_t_1_s_t_series[2*t] = float2short(s_t_1_noisy);

      /* Sample t=T without noise */
      s_t_1_s_t_series[2*t+1] = s_t_clean[idx_t];

      /* Noise addition - Emulate the situation 'Try to yield ideal e_t under erroneous samples, but has some error in the e_t' */
      // Derivatives: Noise source at residual (at sample @original -> at residual from @b858ea9)
      float e_t_noisy = e_t_ideal + noise[idx_t];
      e_t_noisy = IMIN(255, IMAX(0, e_t_noisy));
      float s_t_noisy = p_t_noisy + ulaw2lin(e_t_noisy);

      /* ✅ State update over loop */
      //// Update s_t_x_noisy's {t-2} ~ {t-LPC_ORDER} (sig_mem[1:] = sig_mem[0:LPC_ORDER-1])
      RNN_MOVE(&st->sig_mem[1], &st->sig_mem[0], LPC_ORDER-1);
      //// Update t-1 (s_t_1_noisy)
      st->sig_mem[0] = s_t_noisy;

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
  int noisebuf[FRAME_SIZE*4]={0}; // Noise buffer for s_t_1_noisy
  short tmp[FRAME_SIZE] = {0};
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
    ////         len(x)               `tmp` is initialized with 0
    for (i=0;i<FRAME_SIZE;i++) x[i] = tmp[i]; // implicit cast?

    // Load Int16 samples of a frame from <input.s16> onto `tmp` for next loop...?
    ret = fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1) || ret != FRAME_SIZE) {
      if (!training) break;
      // N-th pass from frame#0
      rewind(f1);
      ret = fread(tmp, sizeof(short), FRAME_SIZE, f1);
      if (ret != FRAME_SIZE) {
        // Even first frame do not have enough length, something wrong.
        fprintf(stderr, "error reading\n");
        exit(1);
      }
      one_pass_completed = 1;
    }

    // Silent clipping (disabled @5627af3)

    /* Loop escape */
    // For too small data, loop more than 2 passes
    // if        enough_data           && >=1_pass_completed
    if (count*FRAME_SIZE_5MS>=10000000 && one_pass_completed) break;

    /* Parameter generation */
    //   Updated once per 2821 frames
    //   todo: Where dose the magic number 2821 come from ...?
    if (training && ++gain_change_count > 2821) {

      /* ✅ `a_sig` & `b_sig` - SpecAug's 2nd-order feedback/forward params ~U[-0.375, +0.375] */
      rand_resp(a_sig, b_sig);

      /* ✅ `speech_gain` - Gain factor at frame end */
      speech_gain = pow(10., (-20+(rand()%40))/20.);
      if (rand()%20==0) speech_gain *= .01;
      if (rand()%100==0) speech_gain = 0;

      /* `noise_std` - Noise for s_t_1_noisy */
      float tmp1 = (float)rand()/RAND_MAX; // ~ U[0, 1]
      float tmp2 = (float)rand()/RAND_MAX; // ~ U[0, 1]
      noise_std = ABS16(-1.5*log(1e-4 + tmp1) - .5*log(1e-4 + tmp2));

      /* ✅ Reset parameter change count */
      gain_change_count = 0;
    }


    /* ==== Augmentation ================================================================================ */
    // Augment `x`, samples of a frame, by "SpecAug -> Preemphasis -> Gain -> NoiseAddition"

    /* SpecAug */
    // Derivatives: Random spectral augmentation (OFF @original -> ON @efficiency from @396274f)
    //   c.f. Eq.7 of Valin, et al. (2017). *A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement*. arxiv:1709.08243
    // High-pass filter...? (a_hp=[-1.99599, 0.99600], b_hp=[-2, 1]) (seems to come from original RNNoise...?)
    biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    // ✅ Random spectral augmentation
    biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);

    /* ✅ Preemphasis - Preemphasize a frame `x` (len==FRAME_SIZE) with coeff `PREEMPHASIS` in place */
    preemphasis(x, &mem_preemph, x, PREEMPHASIS, FRAME_SIZE);

    /* ✅ Gain - Sample-wise gain with smooth gain transition */
    for (i=0;i<FRAME_SIZE;i++) {
      float g; // gain of a sample
      float f = (float)i/FRAME_SIZE; // ratio in a frame
      // Gain smoothing
      g = f*speech_gain + (1-f)*old_speech_gain;
      x[i] *= g;
    }

    /* ✅ Noise addition - Sample-wise noise addition ~ U[-0.5, +0.5] */
    // todo: It looks strange. x will be used for clean target, so does it make output noisy...?
    for (i=0;i<FRAME_SIZE;i++) x[i] += rand()/(float)RAND_MAX - .5;
    /* ================================================================================================== */


    /* PCM is shifted to make the features centered on the frames. */
    /*
                       t                     t+F-ost       t+F
       augumented x  --|------------------------------------|
                       |__________________________|
                                         ⤵
                                (t)                     (t+F-ost)
                                 |__________________________|
      s_frame_clean    |------------------------------------|
                    (t-ost)     (t)
    */
    for (i=0;i<FRAME_SIZE-TRAINING_OFFSET;i++) s_frame_clean[i+TRAINING_OFFSET] = float2short(x[i]);


    /* ==== Feature-nize ================================================================================ */
    // Generate features

    /* Feature extraction */
    // Calculate parts of `.features` (bfcc & lpcoeff) from non-shifted s_t_clean (x) and store them in `st`
    compute_frame_features(st, x);

    /* Data stock */
    int frame_start = st->pcount*FRAME_SIZE;
    // Stack a frame into the buffer for 4 frame grouped processing mode
    RNN_COPY(&s_4frames_clean[frame_start], s_frame_clean, FRAME_SIZE);

    /* Noise generation for noisy sample augmentation (s_t_1_noisy) */
    if (fpcm) { // `if (fpcm)` check for non-train mode
        compute_noise(&noisebuf[frame_start], noise_std);
    }

    // A. non-quantize
    if (!quantize) {
      /* Pitch generation and Dump */
      // Calculate remaining `.features` (pitches) and Dump full `.features` into the `ffeat` file
      process_single_frame(st, ffeat);

      /* Sample series (s_t_1_noisy & s_t_clean) augmentation and Dump */
      //                                                                     singleFrame
      if (fpcm) write_audio(st, s_frame_clean, &noisebuf[frame_start], fpcm, 1);
    }
    st->pcount++;
    /* Running on groups of 4 frames when quantize mode. */
    if (st->pcount == 4) {
      // B. quantize
      if (quantize) {
        unsigned char buf[8];

        /* Pitch generation and Dump */
        process_superframe(st, buf, ffeat, encode, quantize);

        /* Sample series generation and dump */
        if (fpcm) write_audio(st, s_4frames_clean, noisebuf, fpcm, 4);
      }
      // counter reset
      st->pcount = 0;
    }
    /* ================================================================================================== */


    /* Shift remainings
                       t                     t+F-ost       t+F
       augumented x  --|------------------------------------|--
                                                  |_________|
                                  ←___________________/
                   (t+F-ost)  (t+F)
                       |________|
      s_frame_clean    |------------------------------------|
                                 (t)                    (t+F-ost)
    */
    for (i=0;i<TRAINING_OFFSET;i++) s_frame_clean[i] = float2short(x[i+FRAME_SIZE-TRAINING_OFFSET]);

    /* ✅ Gain - Gain factor at next frame start */
    old_speech_gain = speech_gain;

    // Increment the number of processed frames
    count++;
  }

  /* Termination */
  fclose(f1);
  fclose(ffeat);
  if (fpcm) fclose(fpcm);
  lpcnet_encoder_destroy(st);

  return 0;
}

