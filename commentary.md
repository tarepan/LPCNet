## Implementations
### Vector operations
Highly-optimized vector operations through intrinsic SIMD functions.  

- root: [`vec.h`](https://github.com/tarepan/LPCNet/blob/master/src/vec.h)
    - (internal) x86 SIMD: [`vec_avx.h`](https://github.com/tarepan/LPCNet/blob/master/src/vec_avx.h)
    - (internal) ARM SIMD: [`vec_neon.h`](https://github.com/tarepan/LPCNet/blob/master/src/vec_neon.h)