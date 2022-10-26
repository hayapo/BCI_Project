import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from lib import filter_func
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

board_id = BoardIds.CYTON_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)
FS: int = 250

# データ読み込み
measure_date: str = '2022-10-14'
subject_num: int = 2
exp_type: str = 'actual'
test_flag: bool = True

if test_flag:
  pathName = f'../../result/test/{measure_date}/subject_{subject_num}/{exp_type}/'
else:
  pathName = f'../../result/{measure_date}/subject_{subject_num}/{exp_type}/'

fig = plt.figure()
fig.suptitle("STFT: Practice", fontsize=20)

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

ax1.set_title('Cz')
ax2.set_title('C3')
ax3.set_title('C4')
ax4.set_title('Fz')
ax5.set_title('F3')
ax6.set_title('F4')

cols = ['Cz', 'C3', 'C4','Fz', 'F3', 'F4']

df_sum = pd.DataFrame(index=range(6*FS),columns=cols)
df_sum.fillna(0,inplace=True)

bpf_Fp = np.array([5, 20])
bpf_Fs = np.array([1, 250])

steps:int = 20

fileName = f'subject_{subject_num}_step_{1}.csv'
data = DataFilter.read_file(pathName+fileName)
df = pd.DataFrame(np.transpose(data))

stft_start:int = FS * (3 + 5 - 1) - 548
stft_end:int = stft_start + (FS * 6 + 548)

Cz_f_sum = np.zeros(63)
Cz_t_sum = np.zeros(17)
Cz_Sxx_sum = np.zeros((63,17))

C3_f_sum = np.zeros(63)
C3_t_sum = np.zeros(17)
C3_Sxx_sum = np.zeros((63,17))

C4_f_sum = np.zeros(63)
C4_t_sum = np.zeros(17)
C4_Sxx_sum = np.zeros((63,17))

Fz_f_sum = np.zeros(63)
Fz_t_sum = np.zeros(17)
Fz_Sxx_sum = np.zeros((63,17))

F3_f_sum = np.zeros(63)
F3_t_sum = np.zeros(17)
F3_Sxx_sum = np.zeros((63,17))

F4_f_sum = np.zeros(63)
F4_t_sum = np.zeros(17)
F4_Sxx_sum = np.zeros((63,17))

data_Cz = df[3]
data_C3 = df[4]
data_C4 = df[5]
data_Fz = df[6]
data_F3 = df[7]
data_F4 = df[8]

Cz_notch_filtered = filter_func.notchfilter(data_Cz, FS)
Cz_filtered = filter_func.bandpass(Cz_notch_filtered, FS, bpf_Fp, bpf_Fs, 3, 40)

C3_notch_filtered = filter_func.notchfilter(data_C3, FS)
C3_filtered = filter_func.bandpass(C3_notch_filtered, FS, bpf_Fp, bpf_Fs, 3, 40)

C4_notch_filtered = filter_func.notchfilter(data_C4, FS)
C4_filtered = filter_func.bandpass(C4_notch_filtered, FS, bpf_Fp, bpf_Fs, 3, 40)

Fz_notch_filtered = filter_func.notchfilter(data_Fz, FS)
Fz_filtered = filter_func.bandpass(Fz_notch_filtered, FS, bpf_Fp, bpf_Fs, 3, 40)

F3_notch_filtered = filter_func.notchfilter(data_F3, FS)
F3_filtered = filter_func.bandpass(F3_notch_filtered, FS, bpf_Fp, bpf_Fs, 3, 40)

F4_notch_filtered = filter_func.notchfilter(data_F4, FS)
F4_filtered = filter_func.bandpass(F4_notch_filtered, FS, bpf_Fp, bpf_Fs, 3, 40)

window = signal.windows.hamming(125)
nfft = 150
n = len(Cz_filtered)
t = n / FS

f_Cz, t_Cz, Sxx_Cz = signal.stft(Cz_filtered[stft_start:stft_end], FS, window=window, noverlap=8, nperseg=125)
f_C3, t_C3, Sxx_C3 = signal.stft(C3_filtered[stft_start:stft_end], FS, window=window, noverlap=8, nperseg=125)
f_C4, t_C4, Sxx_C4 = signal.stft(C4_filtered[stft_start:stft_end], FS, window=window, noverlap=8, nperseg=125)
f_Fz, t_Fz, Sxx_Fz = signal.stft(Fz_filtered[stft_start:stft_end], FS, window=window, noverlap=8, nperseg=125)
f_F3, t_F3, Sxx_F3 = signal.stft(F3_filtered[stft_start:stft_end], FS, window=window, noverlap=8, nperseg=125)
f_F4, t_F4, Sxx_F4 = signal.stft(F4_filtered[stft_start:stft_end], FS, window=window, noverlap=8, nperseg=125)

ax1.pcolormesh(t_Cz, f_Cz, 10*np.log(np.abs(Sxx_Cz)))
ax2.pcolormesh(t_C3, f_C3, 10*np.log(np.abs(Sxx_C3)))
ax3.pcolormesh(t_C4, f_C4, 10*np.log(np.abs(Sxx_C4)))
ax4.pcolormesh(t_Fz, f_Fz, 10*np.log(np.abs(Sxx_Fz)))
ax5.pcolormesh(t_F3, f_F3, 10*np.log(np.abs(Sxx_F3)))
ax6.pcolormesh(t_F4, f_F4, 10*np.log(np.abs(Sxx_F4)))

ax1.set_ylim(0, 25)
ax2.set_ylim(0, 25)
ax3.set_ylim(0, 25)
ax4.set_ylim(0, 25)
ax5.set_ylim(0, 25)
ax6.set_ylim(0, 25)

plt.show()