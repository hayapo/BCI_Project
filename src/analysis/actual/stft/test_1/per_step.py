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
WAIT_SECOND_ACTUAL: list[int] = [8, 6, 6, 7, 5, 7, 7, 5, 5, 7, 9, 9, 9, 8, 6, 5, 6, 8, 9, 8]

# データ読み込み
measure_date: str = '2022-11-07'
subject_num: int = 6
exp_type: str = 'actual'
test_flag: bool = True

if test_flag:
  pathName = f'../../result/test/{measure_date}/subject_{subject_num}/{exp_type}/'
else:
  pathName = f'../../result/{measure_date}/subject_{subject_num}/{exp_type}/'

fig = plt.figure()
fig.suptitle("Subject 6: STFT(Actual) Step 16", fontsize=20)

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

Cz_f_sum = np.zeros(251)
Cz_t_sum = np.zeros(10)
Cz_Sxx_sum = np.zeros((251,10))

C3_f_sum = np.zeros(251)
C3_t_sum = np.zeros(10)
C3_Sxx_sum = np.zeros((251,10))

C4_f_sum = np.zeros(251)
C4_t_sum = np.zeros(10)
C4_Sxx_sum = np.zeros((251,10))

Fz_f_sum = np.zeros(251)
Fz_t_sum = np.zeros(10)
Fz_Sxx_sum = np.zeros((251,10))

F3_f_sum = np.zeros(251)
F3_t_sum = np.zeros(10)
F3_Sxx_sum = np.zeros((251,10))

F4_f_sum = np.zeros(251)
F4_t_sum = np.zeros(10)
F4_Sxx_sum = np.zeros((251,10))

steps:int = 20

fileName = f'subject_{subject_num}_step_{16}.csv'
data = DataFilter.read_file(pathName+fileName)
df = pd.DataFrame(np.transpose(data))

stft_start:int = FS * (3 + WAIT_SECOND_ACTUAL[15] - 1) - 548
stft_end:int = stft_start +  FS * 6 + 548

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
#nfft = 250
n = len(Cz_filtered)
t = n / FS

f_Cz, t_Cz, Sxx_Cz = signal.stft(Cz_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
f_C3, t_C3, Sxx_C3 = signal.stft(C3_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
f_C4, t_C4, Sxx_C4 = signal.stft(C4_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
f_Fz, t_Fz, Sxx_Fz = signal.stft(Fz_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
f_F3, t_F3, Sxx_F3 = signal.stft(F3_filtered[stft_start:stft_end], FS, window=window, nperseg=125)
f_F4, t_F4, Sxx_F4 = signal.stft(F4_filtered[stft_start:stft_end], FS, window=window, nperseg=125)

Cz_Sxx = 10*np.log(np.abs(Sxx_Cz))
C3_Sxx = 10*np.log(np.abs(Sxx_C3))
C4_Sxx = 10*np.log(np.abs(Sxx_C4))
Fz_Sxx = 10*np.log(np.abs(Sxx_Fz))
F3_Sxx = 10*np.log(np.abs(Sxx_F3))
F4_Sxx = 10*np.log(np.abs(Sxx_F4))

Cz_Sxx_min = Cz_Sxx.min()
Cz_Sxx_max = Cz_Sxx.max()

C3_Sxx_min = C3_Sxx.min()
C3_Sxx_max = C3_Sxx.max()

C4_Sxx_min = C4_Sxx.min()
C4_Sxx_max = C4_Sxx.max()

Fz_Sxx_min = Fz_Sxx.min()
Fz_Sxx_max = Fz_Sxx.max()

F3_Sxx_min = F3_Sxx.min()
F3_Sxx_max = F3_Sxx.max()

F4_Sxx_min = F4_Sxx.min()
F4_Sxx_max = F4_Sxx.max()

Sxx_max_ave = (Cz_Sxx_max + C3_Sxx_max + C4_Sxx_max + Fz_Sxx_max + F3_Sxx_max + F4_Sxx_max) // 6
Sxx_min_ave = (Cz_Sxx_min + C3_Sxx_min + C4_Sxx_min + Fz_Sxx_min + F3_Sxx_min + F4_Sxx_min) // 6

ax1.pcolormesh(t_Cz, f_Cz, Cz_Sxx, cmap='jet')
ax2.pcolormesh(t_C3, f_C3, C3_Sxx, cmap='jet')
ax3.pcolormesh(t_C4, f_C4, C4_Sxx, cmap='jet')
ax4.pcolormesh(t_Fz, f_Fz, Fz_Sxx, cmap='jet')
ax5.pcolormesh(t_F3, f_F3, F3_Sxx, cmap='jet')
ax6.pcolormesh(t_F4, f_F4, F4_Sxx, cmap='jet')

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