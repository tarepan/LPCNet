## Models
```
      Encode           Decode
Wave --------> Latent --------> Wave
```

Default: 8-bit quantization with AVX*/Neon  

Input: 16bit/16kHz PCM  

Compressed latent: 8bytes/40ms  


## APIs
From [`lpcnet.h`](https://github.com/tarepan/LPCNet/blob/master/include/lpcnet.h)  

- LPCNet
  - Size: `lpcnet_get_size`
  - Init: `lpcnet_init`
  - Create: `lpcnet_create`
  - Destroy: `lpcnet_destroy`
  - Run: `lpcnet_synthesize`
- Decoder
  - Size: `lpcnet_decoder_get_size`
  - Init: `lpcnet_decoder_init`
  - Create: `lpcnet_decoder_create`
  - Destroy: `lpcnet_decoder_destroy`
  - Run: `lpcnet_decode`
- Encoder
  - Size: `lpcnet_encoder_get_size`
  - Init: `lpcnet_encoder_init`
  - Create: `lpcnet_encoder_create`
  - Destroy: `lpcnet_encoder_destroy`
  - Run:
    - `lpcnet_encode`
    - `lpcnet_compute_features`
    - `lpcnet_compute_single_frame_features`

/* `lpcnet_init`: Delete states of argument LPCNetState and set initial values */
/* `lpcnet_create`: Allocate LPCNetState memory and initialize it */
/* `lpcnet_destroy`: Free LPCNetState */

### `lpcnet_synthesize`
Synthesizes speech from the LPCNet state and the input features, then output to the `output` adress.  

```C
/** Synthesizes speech from an LPCNet feature vector.
  * @param [in] st <tt>LPCNetState*</tt>: Synthesis state
  * @param [in] features <tt>const float *</tt>: Compressed packet
  * @param [out] output <tt>short **</tt>: Synthesized speech
  * @param [in] N <tt>int</tt>: Number of samples to generate
  * @retval 0 Success
  */
LPCNET_EXPORT void lpcnet_synthesize(LPCNetState *st, const float *features, short *output, int N);
```

#### Internal
```C
/* `LPCNetState`: Container struct of weights and values */
struct LPCNetState {
    NNetState nnet;
    int last_exc;
    float last_sig[LPC_ORDER];
#if FEATURES_DELAY>0
    float old_lpc[FEATURES_DELAY][LPC_ORDER];
#endif
    float sampling_logit_table[256];
    int frame_count;
    float deemph_mem;
};



int run_sample_network(NNetState *net, const float *gru_a_condition, const float *gru_b_condition, int last_exc, int last_sig, int pred, const float *sampling_logit_table)
{
    float gru_a_input[3*GRU_A_STATE_SIZE];
    float in_b[GRU_A_STATE_SIZE+FEATURE_DENSE2_OUT_SIZE];
    float gru_b_input[3*GRU_B_STATE_SIZE];
#if 1
    compute_gru_a_input(gru_a_input, gru_a_condition, GRU_A_STATE_SIZE, &gru_a_embed_sig, last_sig, &gru_a_embed_pred, pred, &gru_a_embed_exc, last_exc);
#else
    RNN_COPY(gru_a_input, gru_a_condition, 3*GRU_A_STATE_SIZE);
    accum_embedding(&gru_a_embed_sig, gru_a_input, last_sig);
    accum_embedding(&gru_a_embed_pred, gru_a_input, pred);
    accum_embedding(&gru_a_embed_exc, gru_a_input, last_exc);
#endif
    /*compute_gru3(&gru_a, net->gru_a_state, gru_a_input);*/
    compute_sparse_gru(&sparse_gru_a, net->gru_a_state, gru_a_input);
    RNN_COPY(in_b, net->gru_a_state, GRU_A_STATE_SIZE);
    RNN_COPY(gru_b_input, gru_b_condition, 3*GRU_B_STATE_SIZE);
    compute_gruB(&gru_b, gru_b_input, net->gru_b_state, in_b);
    return sample_mdense(&dual_fc, net->gru_b_state, sampling_logit_table);
}

`compute_sparse_gru`
`compute_gruB`
`sample_mdense`
```

```C
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

```

##### `sgemv_accum8x4`

## Implementations
### Vector operations
Highly-optimized vector operations through intrinsic SIMD functions.  

- root: [`vec.h`](https://github.com/tarepan/LPCNet/blob/master/src/vec.h)
    - (internal) x86 SIMD: [`vec_avx.h`](https://github.com/tarepan/LPCNet/blob/master/src/vec_avx.h)
    - (internal) ARM SIMD: [`vec_neon.h`](https://github.com/tarepan/LPCNet/blob/master/src/vec_neon.h)