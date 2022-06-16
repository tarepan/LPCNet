# LPCNet

Low complexity implementation of the LPCNet algorithm, as described in:

- J.-M. Valin, J. Skoglund, [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf), *Proc. International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, arXiv:1810.11846, 2019.
- J.-M. Valin, J. Skoglund, [A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet](https://jmvalin.ca/papers/lpcnet_codec.pdf), *Proc. INTERSPEECH*, arxiv:1903.12087, 2019.
- J. Skoglund, J.-M. Valin, [Improving Opus Low Bit Rate Quality with Neural Speech Synthesis](https://jmvalin.ca/papers/opusnet.pdf), *Proc. INTERSPEECH*, arxiv:1905.04628, 2020.

## Introduction

Work in progress software for researching low CPU complexity algorithms for speech synthesis and compression by applying Linear Prediction techniques to WaveRNN. High quality speech can be synthesised on regular CPUs (around 3 GFLOP) with SIMD support (SSE2, SSSE3, AVX, AVX2/FMA, NEON currently supported). The code also supports very low bitrate compression at 1.6 kb/s.

The BSD licensed software is written in C and Python/Keras. For training, a GTX 1080 Ti or better is recommended.

## Inference
### Setup
Following commands build inference program written by C for x86/64 and ARM CPU.  

#### Step 0 - Model data
If use your trained model data, skip this step.  
Else, you can download pretrained model.  
```bash
./download_model.sh
```
#### Step 1 - Env
Set variables for vectorization.  
By default, the program attempt to use 8-bit dot product instructions on AVX\*/Neon.  
##### x86
```
export CFLAGS='-Ofast -g -march=native'
```
##### ARM
```
export CFLAGS='-Ofast -g -mfpu=neon'
```
While not strictly required, the -Ofast flag will help with auto-vectorization, especially for dot products that cannot be optimized without -ffast-math (which -Ofast enables).  
Additionally, -falign-loops=32 has been shown to help on x86.  

#### Step 2 - Build
```
./autogen.sh    # Latest model download & `autoreconf`
./configure     # Run the generated configure script
make
```
Note that the autogen.sh script is used when building from Git  

##### Options
- `configure`
  - `--disable-dot-product`: Disable usage of 8-bit dot product instructions

use case: avoid quantization effects when retraining  

##### Restriction
ARMv7: Needs `--disable-dot-product` because of not yet complete implementation  

### Demo
You can test the capabilities of LPCNet using the lpcnet\_demo application.  
The same functionality is available in the form of a library. See include/lpcnet.h for the API.  

#### Speech Compression
Speech encoding & decoding.  

```
# Encode `input.pcm` (16bit/16kHz PCM, machine endian)
#   to `compressed.bin` (8 bytes per 40-ms packet, raw, no header)
./lpcnet_demo -encode input.pcm compressed.bin

# Decode `compressed.bin` to `output.pcm` (16bit/16kHz PCM)
./lpcnet_demo -decode compressed.bin output.pcm
```

#### Speech Synthesis
Uncompressed analysis/synthesis.  

```
# (maybe) Feature-rize
./lpcnet_demo -features  input.pcm uncompressed.bin

# Synthesis
./lpcnet_demo -synthesis uncompressed.bin output.pcm
```

## Training

### 1. Setup
Set up a Keras system with GPU.

### 2. Preprocessing
Convert input waveforms into preprocessed features.  

```bash
./dump_data -train <input.s16> <features.f32> <data.s16>
```

- \<input.s16>: input file containing 16 kHz 16-bit raw PCM audio (no header)
- \<features.f32>: output file containing ...
- \<data.s16>: output file containing ...

This program makes several passes over the data with different filters to generate a large amount of training data.

### 3. Train
```bash
python3 training_tf2/train_lpcnet.py <features.f32> <data.s16> <model_name> # --batch-size=32
```
and it will generate an h5 file for each iteration, with model\_name as prefix.

### 4. Evaluation
#### 4-a. Synthesis on GPU
※ very slower than CPU
```bash
./dump_data -test <test_input.s16> <test_features.f32>
./training_tf2/test_lpcnet.py <lpcnet_model_name.h5> <test_features.f32> test.s16
```
#### 4-b. Synthesis on CPU
much faster than GPU

First extract the model files `nnet_data.h` and `nnet_data.c`.  
```
./training_tf2/dump_lpcnet.py <lpcnet_model_name.h5>
mv nnet_data.c nnet_data.h src/
```

Then follow "inference" section (software rebuild).  

## Dataset

Suitable training material can be obtained from [Open Speech and Language Resources](https://www.openslr.org/).  
See the datasets.txt file for details on suitable training data.  

## Reading Further

1. [LPCNet: DSP-Boosted Neural Speech Synthesis](https://people.xiph.org/~jm/demo/lpcnet/)
1. [A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet](https://people.xiph.org/~jm/demo/lpcnet_codec/)
1. Sample model files (check compatibility): https://media.xiph.org/lpcnet/data/ 

