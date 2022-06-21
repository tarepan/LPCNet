import numpy as np
from tensorflow.keras.utils import Sequence


def lpc2rc(lpc):
    order = lpc.shape[-1]
    rc = 0*lpc
    for i in range(order, 0, -1):
        rc[:,:,i-1] = lpc[:,:,-1]
        ki = rc[:,:,i-1:i].repeat(i-1, axis=2)
        lpc = (lpc[:,:,:-1] - ki*lpc[:,:,-2::-1])/(1-ki*ki)
    return rc


class LPCNetLoader(Sequence):
    """Data loader for LPCNet."""
    def __init__(self, s_t_1_s_t_series, feat_lpc_series, periods, batch_size: int, e2e:bool=False):
        """
        Args:
            s_t_1_s_t_series ::(Chunk, T_s,       IO=2) - Input s_{t-1} series and Output s_t series, partially accessible
            feat_lpc_series  ::(B,     T_f, Feat+Order) - Acoustic feature series and LP coefficient series, partially accessible
            periods          ::(B,     T_f,          1) - Pitch period series, partially accessible
            batch_size - Batch size
            e2e - Whether to use End-to-End mode
        """
        self.batch_size = batch_size
        self.e2e = e2e

        assert s_t_1_s_t_series.shape[0] == feat_lpc_series.shape[0], f"s_t_1_s_t_series.shape[0] {s_t_1_s_t_series.shape[0]} should be equal to feat_lpc_series.shape[0] {feat_lpc_series.shape[0]}"
        assert feat_lpc_series.shape[0] == periods.shape[0], f"feat_lpc_series.shape[0] {feat_lpc_series.shape[0]} should be equal to periods.shape[0] {periods.shape[0]}"

        # Item (chunk) selection for chipping-less batching
        self.nb_batches = s_t_1_s_t_series.shape[0] // batch_size
        self._num_item = self.nb_batches * batch_size
        self._s_t_1_s_t_series = s_t_1_s_t_series[:self._num_item]
        self._feat_lpc_series =   feat_lpc_series[:self._num_item]
        self._pitch_period_series =       periods[:self._num_item]

        # Item index shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """Item index shuffle at epoch end."""
        self._chunk_indices = np.arange(self._num_item)
        np.random.shuffle(self._chunk_indices)

    def __getitem__(self, index):
        """
        Prepare a batch.

        Returns:
            inputs - Batch of input chunks
                s_t_1_noisy_series  ::(B, T_s,       1) - Lagged/Delayed sample series s_{t-1} (waveform), with noise
                feat_series         ::(B, T_f,    Feat) - Acoustic feature series
                pitch_series        ::(B, T_f,       1) - Pitch series
                lpcoeff_series      ::(B, T_f-4, Order) - Linear Prediction coefficient series, for default (non-E2E) input
            targets - Batch of target chunks
                s_t_clean_series    ::(B, T_s,       1) - Target sample series s_t (waveform), without noise
                lpcoeff_series      ::(B, T_f-4, Order) - Linear Prediction coefficient series, for E2E regularization
        """
        # Batching (index is shuffled at every epoch end)
        idx_start =  index   * self.batch_size
        idx_end   = (index+1)* self.batch_size
        indices_item = self._chunk_indices[idx_start:idx_end]
        s_t_1_s_t_series = self._s_t_1_s_t_series[indices_item]
        feat_lpc_series =   self._feat_lpc_series[indices_item]
        pitch_series =  self._pitch_period_series[indices_item]

        ## I/O sample series :: (B, T_s, 1)
        s_t_1_noisy_series = s_t_1_s_t_series[: , :, 0:1]
        s_t_clean_series   = s_t_1_s_t_series[: , :, 1:2]

        # Acoustic Feature series and LP coefficient series
        ORDER_LPC: int = 16
        feat_series    = feat_lpc_series[:, :,    :-ORDER_LPC]
        # todo: Is this compatible with look-ahead setup?
        lpcoeff_series = feat_lpc_series[:, 2:-2, -ORDER_LPC:]

        # Total I/O
        ## Inputs - In default, explicit LP coefficient input
        inputs = [s_t_1_noisy_series, feat_series, pitch_series]
        inputs += [lpcoeff_series] if not self.e2e else []
        ## Outputs - In E2E, LP coefficients (RCs) target for regularization
        targets = [s_t_clean_series]
        targets += []              if not self.e2e else [lpc2rc(lpcoeff_series)]

        return (inputs, targets)

    def __len__(self):
        return self.nb_batches
