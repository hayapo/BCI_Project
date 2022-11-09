import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
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
exp_type: str = 'practice'
test_flag: bool = True

if test_flag:
  pathName = f'../../result/test/{measure_date}/subject_{subject_num}/{exp_type}/'
else:
  pathName = f'../../result/{measure_date}/subject_{subject_num}/{exp_type}/'

fig = plt.figure()
fig.suptitle("Subject1: STFT(Practice)", fontsize=20)

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

bpf_Fp = np.array([3, 20])
bpf_Fs = np.array([1, 250])

Cz_f_sum = np.zeros(63)
Cz_t_sum = np.zeros(34)
Cz_Sxx_sum = np.zeros((63,34))

C3_f_sum = np.zeros(63)
C3_t_sum = np.zeros(34)
C3_Sxx_sum = np.zeros((63,34))

C4_f_sum = np.zeros(63)
C4_t_sum = np.zeros(34)
C4_Sxx_sum = np.zeros((63,34))

Fz_f_sum = np.zeros(63)
Fz_t_sum = np.zeros(34)
Fz_Sxx_sum = np.zeros((63,34))

F3_f_sum = np.zeros(63)
F3_t_sum = np.zeros(34)
F3_Sxx_sum = np.zeros((63,34))

F4_f_sum = np.zeros(63)
F4_t_sum = np.zeros(34)
F4_Sxx_sum = np.zeros((63,34))

steps:int = 10

for i in range(steps):
  fileName = f'subject_{subject_num}_step_{i+1}.csv'
  data = DataFilter.read_file(pathName+fileName)
  df = pd.DataFrame(np.transpose(data))

  stft_start:int = FS * (3 + 5 - 1) - 548
  stft_end:int = stft_start + (FS * 6 + 548)

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

  f_Cz, t_Cz, Sxx_Cz = signal.stft(Cz_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
  f_C3, t_C3, Sxx_C3 = signal.stft(C3_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
  f_C4, t_C4, Sxx_C4 = signal.stft(C4_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
  f_Fz, t_Fz, Sxx_Fz = signal.stft(Fz_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
  f_F3, t_F3, Sxx_F3 = signal.stft(F3_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
  f_F4, t_F4, Sxx_F4 = signal.stft(F4_filtered[stft_start:stft_end], FS, window=window, nperseg=125)

  Cz_f_sum += f_Cz
  Cz_t_sum += t_Cz
  Cz_Sxx_sum += 10*np.log(np.abs(Sxx_Cz))

  C3_f_sum += f_C3
  C3_t_sum += t_C3
  C3_Sxx_sum += 10*np.log(np.abs(Sxx_C3))

  C4_f_sum += f_C4
  C4_t_sum += t_C4
  C4_Sxx_sum += 10*np.log(np.abs(Sxx_C4))

  Fz_f_sum += f_Fz
  Fz_t_sum += t_Fz
  Fz_Sxx_sum += 10*np.log(np.abs(Sxx_Fz))

  F3_f_sum += f_F3
  F3_t_sum += t_F3
  F3_Sxx_sum += 10*np.log(np.abs(Sxx_F3))

  F4_f_sum += f_F4
  F4_t_sum += t_F4
  F4_Sxx_sum += 10*np.log(np.abs(Sxx_F4))

Cz_Sxx_ave = Cz_Sxx_sum/10
C3_Sxx_ave = C3_Sxx_sum/10
C4_Sxx_ave = C4_Sxx_sum/10
Fz_Sxx_ave = Fz_Sxx_sum/10
F3_Sxx_ave = F3_Sxx_sum/10
F4_Sxx_ave = F4_Sxx_sum/10

Cz_Sxx_min = Cz_Sxx_ave.min()
Cz_Sxx_max = Cz_Sxx_ave.max()

C3_Sxx_min = C3_Sxx_ave.min()
C3_Sxx_max = C3_Sxx_ave.max()

C4_Sxx_min = C4_Sxx_ave.min()
C4_Sxx_max = C4_Sxx_ave.max()

Fz_Sxx_min = Fz_Sxx_ave.min()
Fz_Sxx_max = Fz_Sxx_ave.max()

F3_Sxx_min = F3_Sxx_ave.min()
F3_Sxx_max = F3_Sxx_ave.max()

F4_Sxx_min = F4_Sxx_ave.min()
F4_Sxx_max = F4_Sxx_ave.max()

Sxx_max_ave = (Cz_Sxx_max + C3_Sxx_max + C4_Sxx_max + Fz_Sxx_max + F3_Sxx_max + F4_Sxx_max) // 6
Sxx_min_ave = (Cz_Sxx_min + C3_Sxx_min + C4_Sxx_min + Fz_Sxx_min + F3_Sxx_min + F4_Sxx_min) // 6

ax1.pcolormesh(Cz_t_sum/10, Cz_f_sum/10, Cz_Sxx_ave, cmap='jet')
ax2.pcolormesh(C3_t_sum/10, C3_f_sum/10, C3_Sxx_ave, cmap='jet')
ax3.pcolormesh(C4_t_sum/10, C4_f_sum/10, C4_Sxx_ave, cmap='jet')
ax4.pcolormesh(Fz_t_sum/10, Fz_f_sum/10, Fz_Sxx_ave, cmap='jet')
ax5.pcolormesh(F3_t_sum/10, F3_f_sum/10, F3_Sxx_ave, cmap='jet')
ax6.pcolormesh(F4_t_sum/10, F4_f_sum/10, F4_Sxx_ave, cmap='jet')

ax1.axvline(x=3.192, ymin=0, ymax=125, color='magenta', linewidth=3)
ax2.axvline(x=3.192, ymin=0, ymax=125, color='magenta', linewidth=3)
ax3.axvline(x=3.192, ymin=0, ymax=125, color='magenta', linewidth=3)
ax4.axvline(x=3.192, ymin=0, ymax=125, color='magenta', linewidth=3)
ax5.axvline(x=3.192, ymin=0, ymax=125, color='magenta', linewidth=3)
ax6.axvline(x=3.192, ymin=0, ymax=125, color='magenta', linewidth=3)

ax1.set_xlim(0, 8.192)
ax2.set_xlim(0, 8.192)
ax3.set_xlim(0, 8.192)
ax4.set_xlim(0, 8.192)
ax5.set_xlim(0, 8.192)
ax6.set_xlim(0, 8.192)

ax1.set_ylim(3, 20)
ax2.set_ylim(3, 20)
ax3.set_ylim(3, 20)
ax4.set_ylim(3, 20)
ax5.set_ylim(3, 20)
ax6.set_ylim(3, 20)

vmin, vmax = Sxx_min_ave, Sxx_max_ave

axpos = ax6.get_position()
cbar_ax = fig.add_axes([0.87, axpos.y0 + 0.04, 0.02, axpos.height*2])
norm = colors.Normalize(vmin,vmax)
mappable = ScalarMappable(cmap='jet',norm=norm)
mappable._A = []
fig.colorbar(mappable, cax=cbar_ax)

plt.subplots_adjust(right=0.85)
plt.subplots_adjust(wspace=0.1)

plt.show()