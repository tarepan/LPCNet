import numpy as np
from tensorflow.keras.utils import Sequence
from ulaw import lin2ulaw

def lpc2rc(lpc):
    #print("shape is = ", lpc.shape)
    order = lpc.shape[-1]
    rc = 0*lpc
    for i in range(order, 0, -1):
        rc[:,:,i-1] = lpc[:,:,-1]
        ki = rc[:,:,i-1:i].repeat(i-1, axis=2)
        lpc = (lpc[:,:,:-1] - ki*lpc[:,:,-2::-1])/(1-ki*ki)
    return rc


class LPCNetLoader(Sequence):
    """Data loader for LPCNet."""
    def __init__(self, samples, features, periods, batch_size: int, e2e:bool=False):
        """
        Args:
            samples  ::(Chunk, T_sample, IO=2) - I/O sample series (waveform), partially accessible
            features ::(B,     T_frame,  Feat) - Acoustic feature series and LP coefficient series, partially accessible
            periods  ::(B,     T_frame,  1)    - Pitch period series, partially accessible
            batch_size - Batch size
            e2e - Whether to use End-to-End mode
        """
        self.batch_size = batch_size
        self.e2e = e2e

        # Item (chunk) selection for chipping-less batching
        self.nb_batches = np.minimum(np.minimum(samples.shape[0], features.shape[0]), periods.shape[0]) // batch_size
        self._num_item = self.nb_batches * batch_size
        self.data     =  samples[:self._num_item]
        self.features = features[:self._num_item]
        self.periods  =  periods[:self._num_item]

        # Item index shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """Item index shuffle at epoch end."""
        self.indices = np.arange(self._num_item)
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """
        Prepare a batch.

        Returns:
            inputs - Batch of input chunks
                pcm      ::(B, T_sample,  1)     - Sample series (waveform) for AR teacher-forcing in LinearPrediction/Residual/SampleRateNet
                feat     ::(B, T_frame,   Feat)  - Acoustic feature series for conditioning
                pitch    ::(B, T_frame,   1)     - Pitch period series
                lpcoeffs ::(B, T_frame-4, Order) - Linear Prediction coefficient series
            outputs - Batch of clean sample series (waveform)
        """
        # Batching (index is shuffled at every epoch end)
        idx_start =  index   * self.batch_size
        idx_end   = (index+1)* self.batch_size
        indices_item = self.indices[idx_start:idx_end]
        samples =      self.data[indices_item]
        features = self.features[indices_item]
        periods =   self.periods[indices_item]

        # I/O sample series (waveform), with noise for input, without noise for output
        ## i, o ::(B, T_sample, 1)
        i_sample_series = samples[: , :, :1]
        o_sample_series = samples[: , :, 1:]

        # Acoustic Feature series (last 16 elements of `features` are LP coefficients)
        acoustic_feat_series = features[:, :, :-16]

        # Total I/O
        inputs = [i_sample_series, acoustic_feat_series, periods]
        outputs = [o_sample_series]

        # LP coefficient series (last 16 elements of `features` are LP coefficients)
        lpc = features[:, 2:-2, -16:]
        if self.e2e:
            outputs.append(lpc2rc(lpc))
        else:
            inputs.append(lpc)

        return (inputs, outputs)

    def __len__(self):
        return self.nb_batches
