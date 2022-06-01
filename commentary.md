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

##### `sgemv_accum8x4`

## Implementations
LPCNet = 'frame rate network' (Conv & Dense) + 'sample rate network' (AR GRU) + LinearPrediction  
LPCNet is implemented in `lpcnet.c`.  
Neural network layers in LPCNet are implemented in `nnet.c`.  

### Vector operations
Highly-optimized vector operations through intrinsic SIMD functions.  

- root: [`vec.h`](https://github.com/tarepan/LPCNet/blob/master/src/vec.h)
    - (internal) x86 SIMD: [`vec_avx.h`](https://github.com/tarepan/LPCNet/blob/master/src/vec_avx.h)
    - (internal) ARM SIMD: [`vec_neon.h`](https://github.com/tarepan/LPCNet/blob/master/src/vec_neon.h)