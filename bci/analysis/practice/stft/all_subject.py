import sys
sys.path.append('../../../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
from scipy import signal
from lib import filter_func, fig_setup
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

board_id = BoardIds.CYTON_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)
FS: int = 250
nfft = 250
window = signal.windows.hamming(nfft)
nperseg = nfft
channels = ['Cz', 'C3', 'C4','Fz', 'F3', 'F4']

# データ読み込み
subject_num: int = 5
exp_type: str = 'practice'
test_flag: bool = True
test_num: int = 1

fig, axes = fig_setup.setup_stft(fig_title=f'STFT(Practice): All Subjects', channels=channels)

bpf_Fp = np.array([3, 20])
bpf_Fs = np.array([1, 250])


f_sum = np.zeros((6, 126))
t_sum = np.zeros((6, 18))
Sxx_sum = np.zeros((6, 126, 18))

'''
TODO: 
  - STFTのプロットおかしい気がするので、直す
    - OpenBCI_GUIのスペクトログラムのソースコード読む
    - Waveletも実装する
'''

for i in range(5):
  pathName = f'../../../result/subject_{i+1}/practice/'
  if test_flag:
    pathName = f'../../../result/test_{test_num}/subject_{i+1}/practice/'

  for j in range(10):
    fileName = f'subject_{i+1}_step_{j+1}.csv'
    data = DataFilter.read_file(pathName+fileName)
    df = pd.DataFrame(np.transpose(data))

    stft_start:int = FS * (3 + 5 - 1) - 548
    stft_end:int = stft_start + (FS * 6 + 548)

    df_all_ch = df\
      .iloc[:, 3:9]\
      .rename(columns={3:'Cz',4:'C3',5:'C4',6:'Fz',7:'F3',8:'F4'})

    for num, ch in enumerate(channels):
      col = num // 3
      row = num % 3

      df_notch_filtered = filter_func\
        .notchfilter(df_all_ch[ch], FS)
      df_filtered = filter_func\
        .bandpass(df_notch_filtered, FS, bpf_Fp, bpf_Fs, 3, 40)

      f, t, Sxx = signal.stft(df_filtered[stft_start:stft_end], FS, window=window, nperseg=nperseg)

      f_sum[num] += f
      t_sum[num] += t
      Sxx_sum[num] += 10 * np.log(np.abs(Sxx))

Sxx_min_sum, Sxx_max_sum = 0, 0

for i in range(6):
  t_ave = t_sum[i]/50
  f_ave = f_sum[i]/50
  Sxx_ave = Sxx_sum[i]/50
  #print(np.shape(t_ave), np.shape(f_ave), np.shape(Sxx_ave))

  Sxx_min_sum += Sxx_ave[i].min()
  Sxx_max_sum += Sxx_ave[i].max()
  
  col = i // 3
  row = i % 3

  axes[col, row].pcolormesh(t_ave, f_ave, Sxx_ave, cmap='jet')
  #axes[col, row].axvline(x=3.192, ymin=0, ymax=125, color='magenta', linewidth=3)

vmin, vmax = Sxx_min_sum/6, Sxx_max_sum/6

fig_setup.adjust_stft(vmin, vmax, fig, axes)

plt.show()