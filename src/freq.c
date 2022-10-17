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
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "freq.h"
#include "pitch.h"
#include "arch.h"
#include <assert.h>

#define SQUARE(x) ((x)*(x))

static const opus_int16 eband5ms[] = {
/*
              Δ+0.2                    Δ+0.4         Δ+0.8     Δ+1.2
|________________________________|_______________|___________|_______|
0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 2.0 2.4 2.8 3.2 4.0 4.8 5.6 6.8 8.0 [k] */
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40
};

static const float compensation[] = {
    0.8f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.666667f, 0.5f, 0.5f, 0.5f, 0.333333f, 0.25f, 0.25f, 0.2f, 0.166667f, 0.173913f
};

typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[OVERLAP_SIZE];
  float dct_table[NB_BANDS*NB_BANDS];
} CommonState;


/**
 * LinFreqComplexSpc-to-BarkLinPowSpc
 *
 * Args:
 *   bandE - Output Bark-frequency   Linear Power Spectrum
 *   X     - Input  Linear-frequency Complex      Spectrum
 */
void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    // i-th band
    int j;
    int band_size;
    // e.g. eband5ms[9]-eband5ms[8] == 10 - 8
    band_size = (eband5ms[i+1]-eband5ms[i])*WINDOW_SIZE_5MS;
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      // power = F(~)**2 = r**2 + i**2
      float power = SQUARE(X[(eband5ms[i]*WINDOW_SIZE_5MS) + j].r) + SQUARE(X[(eband5ms[i]*WINDOW_SIZE_5MS) + j].i);
      sum[i]   += (1-frac) * power;
      sum[i+1] +=   frac   * power;
    }
  }
  // Double head and tail (because of half filterBank?)
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;

  // Write out all band energies
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

/**
 * BarkLinPowSpc-to-LinFreqLinPowSpc
 * Args:
 *   g     - Interpolated Linear-frequency Linear Power Spectrum
 *   bandE - BarkLinPowSpc [0:NB_BANDS]
 */
void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    // Linearly interpolate a band
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])*WINDOW_SIZE_5MS;
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[(eband5ms[i]*WINDOW_SIZE_5MS) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
    }
  }
}

// ==== Frequency Analysis Initialization =============================================
CommonState common;

/**
 * Initialize Frequency Analyzers if not init.
 */
static void check_init(void) {
  int i;

  /* Init Check - Once initialized, never reset */
  if (common.init) return;

  /* Initialization */
  //// ? fft?
  common.kfft = opus_fft_alloc_twiddles(WINDOW_SIZE, NULL, NULL, NULL, 0);

  //// Window
  for (i=0;i<OVERLAP_SIZE;i++)
    common.half_window[i] = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/OVERLAP_SIZE) * sin(.5*M_PI*(i+.5)/OVERLAP_SIZE));
  /*
    sin(
      .5 * M_PI
      *
      sin(.5*M_PI*(i+.5)/OVERLAP_SIZE)
      *
      sin(.5*M_PI*(i+.5)/OVERLAP_SIZE)
    )
  */

  //// DCT table
  for (i=0;i<NB_BANDS;i++) {
    int j;
    for (j=0;j<NB_BANDS;j++) {
      common.dct_table[i*NB_BANDS + j] = cos((i+.5)*j*M_PI/NB_BANDS);
      if (j==0) common.dct_table[i*NB_BANDS + j] *= sqrt(.5);
    }
  }
  common.init = 1;
}
// ====================================================================================


// ==== Discrete Cosine Transform =====================================================
void dct(float *out, const float *in) {
  int i;
  check_init();

  // <vec_x, vec_cos_i>: Simple looping Dot product
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./NB_BANDS);
  }
}

void idct(float *out, const float *in) {
  int i;
  check_init();

  // vec_f * vec_cos_t: Simple looping multiplication
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sqrt(2./NB_BANDS);
  }
}
// ====================================================================================


// ==== Discrete Fourier Transform ====================================================
/**
 * wave-to-LinFreqComplexSpc by Fourier Transform.
 *
 * Args:
 *   out - complex spectrum (only rFT part)
 *   in  - Sample series after windowing
 */
void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE]; // all samples
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();

  /* Cast: Real-to-Complex */
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }

  /* FT: complex FFT with opus implementation */
  opus_fft(common.kfft, x, y, 0);

  /* Write: only rFFT part */
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();

  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  opus_fft(common.kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}
// ====================================================================================


// ==== Linear Prediction Coefficients ================================================
float _lpcnet_lpc(
      opus_val16 *lpc, /* out: [0...p-1] LPC coefficients      */
      opus_val16 *rc,
const opus_val32 *ac,  /* in:  [0...p] autocorrelation values  */
int          p
)
{
   int i, j;
   opus_val32 r;
   opus_val32 error = ac[0];

   RNN_CLEAR(lpc, p);
   RNN_CLEAR(rc, p);
   if (ac[0] != 0)
   {
      for (i = 0; i < p; i++) {
         /* Sum up this iteration's reflection coefficient */
         opus_val32 rr = 0;
         for (j = 0; j < i; j++)
            rr += MULT32_32_Q31(lpc[j],ac[i - j]);
         rr += SHR32(ac[i + 1],3);
         r = -SHL32(rr,3)/error;
         rc[i] = r;
         /*  Update LPC coefficients and total error */
         lpc[i] = SHR32(r,3);
         for (j = 0; j < (i+1)>>1; j++)
         {
            opus_val32 tmp1, tmp2;
            tmp1 = lpc[j];
            tmp2 = lpc[i-1-j];
            lpc[j]     = tmp1 + MULT32_32_Q31(r,tmp2);
            lpc[i-1-j] = tmp2 + MULT32_32_Q31(r,tmp1);
         }

         error = error - MULT32_32_Q31(MULT32_32_Q31(r,r),error);
         /* Bail out once we get 30 dB gain */
         if (error<.001f*ac[0])
            break;
      }
   }
   return error;
}

/**
 * BarkLinPowSpc-to-LPCoeff.
 *
 * Args:
 *   lpc - Output LP coefficients
 *   Ex  - Input  BarkLinPowSpc
 */
float lpc_from_bands(float *lpc, const float *Ex)
{
   int i;
   float e;
   float ac[LPC_ORDER+1];
   float rc[LPC_ORDER];
   float Xr[FREQ_SIZE];
   kiss_fft_cpx X_auto[FREQ_SIZE];
   float x_auto[WINDOW_SIZE];

   /* BarkLinPowSpc-to-PSD (LinFreqLinPowSpc) */
   interp_band_gain(Xr, Ex);
   Xr[FREQ_SIZE-1] = 0;

   /* PSD-to-AC (lag 0 ~ LPC_ORDER) */
   RNN_CLEAR(X_auto, FREQ_SIZE);
   for (i=0;i<FREQ_SIZE;i++) X_auto[i].r = Xr[i];
   inverse_transform(x_auto, X_auto);
   for (i=0;i<LPC_ORDER+1;i++) ac[i] = x_auto[i];

   /* -40 dB noise floor. */
   ac[0] += ac[0]*1e-4 + 320/12/38.;
   /* Lag windowing. */
   for (i=1;i<LPC_ORDER+1;i++) ac[i] *= (1 - 6e-5*i*i);
   e = _lpcnet_lpc(lpc, rc, ac, LPC_ORDER);
   return e;
}

/**
 * BFC-to-LPCoeff.
 *
 * Args:
 *   cepstrum - Bark-frequency Cepstrum (== dct of BarkLogPowSpc)
 **/
float lpc_from_cepstrum(float *lpc, const float *cepstrum)
{
   int i;
   float Ex[NB_BANDS];
   float tmp[NB_BANDS];
   RNN_COPY(tmp, cepstrum, NB_BANDS);

   /* BFC-to-Bands */
   tmp[0] += 4;
   //// BFC-to-BarkLogPowSpc
   idct(Ex, tmp);
   //// BarkLogPowSpc-to-BarkLinPowSpc (compensated...?)
   for (i=0;i<NB_BANDS;i++) Ex[i] = pow(10.f, Ex[i])*compensation[i];

   // BarkLinPowSpc-to-LPCoeff
   return lpc_from_bands(lpc, Ex);
}
// ====================================================================================


// ==== Window ========================================================================
/**
 * Apply window symmetrically.
 */
void apply_window(float *x) {
  int i;
  check_init();

  /*
          WINDOW_SIZE == OVERLAP_SIZE + FRAME_SIZE
         |________________________________________|
  `x`    |----------------------------------------|
         |____________|              |____________|
          OVERLAP_SIZE                OVERLAP_SIZE
  */

  for (i=0;i<OVERLAP_SIZE;i++) {
    x[i]                   *= common.half_window[i];
    x[WINDOW_SIZE - 1 - i] *= common.half_window[i];
  }
}
// ====================================================================================

