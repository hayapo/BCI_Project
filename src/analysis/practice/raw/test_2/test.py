import time

import matplotlib
import numpy as np

import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

import mne
from mne.time_frequency import tfr_morlet

# ファイルメタデータ
exp_type: str = 'practice'
test_flag: bool = True
test_num: int = 2
subject_total: int = 3
subject_num:int = 2

def main():
    pathName = f'result/test_{test_num}/subject_{subject_num}/practice/'
    fileName = f'subject_{subject_num}_step_{9}.csv'
    data = DataFilter.read_file(pathName+fileName)

    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

    # Creating MNE objects from brainflow data arrays
    ch_types = ['eeg'] * len(eeg_channels)
    ch_names = BoardShim.get_eeg_names(BoardIds.CYTON_BOARD.value)
    sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    raw = mne.io.RawArray(eeg_data, info) 
    raw.info['bads']= ['Fp1', 'Fp2']
    orig_raw = raw.copy()

    # set up and fit the ICA
    ica = mne.preprocessing.ICA(n_components=6, random_state=30, max_iter=100)
    raw.load_data()
    ica.fit(raw)
    ica.apply(raw)
    
    # its time to plot something!
    # raw.plot_psd(fmax=20)
    # raw.plot(start=6.0, duration=0.1, n_channels=8, highpass=3,lowpass=40, remove_dc=True)
    frequencies = np.arange(7, 30, 3)

    
    plt.show()

if __name__ == '__main__':
    main()