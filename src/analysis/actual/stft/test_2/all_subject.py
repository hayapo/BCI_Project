import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from lib import filter_func, fig_setup, calc_diff
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

board_id = BoardIds.CYTON_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)
FS: int = 250
nfft = 125
window = signal.windows.hamming(nfft)
nperseg = nfft
channels = ['Cz', 'C3', 'C4','Fz', 'F3', 'F4']

# ファイルメタデータ
exp_type: str = 'actual'
test_flag: bool = True
test_num: int = 2
subject_total: int = 3
# to_data_dir: str = '../../../..'
# result_dir: str = os.path.join(to_data_dir, 'result')

# フィルタ関連の変数
bpf_Fp = np.array([3, 20])
bpf_Fs = np.array([1, 250])

# 合計をもっておく変数
f_sum = np.zeros((6, 63))
t_sum = np.zeros((6, 17))
Sxx_sum = np.zeros((6, 63, 17))

# 図のセットアップ
fig, axes = fig_setup.setup_stft(fig_title="STFT(noFB): All Subject", channels=channels)

# df_sum = pd.DataFrame(index=range(4*FS), columns=channels)
# df_sum.fillna(0, inplace=True)

# 各被験者の各試行のスタート誤差を計算する
time_diffs, which_fast = calc_diff.timeDiff(test_num, exp_type, subject_total, 'result')

for i in range(subject_total):
  pathName = f'result/subject_{i+1}/practice/'
  if test_flag:
    pathName = f'result/test_{test_num}/subject_{i+1}/practice/'

  for j in range(10):
    fileName = f'subject_{i+1}_step_{j+1}.csv'
    data = DataFilter.read_file(pathName+fileName)
    df = pd.DataFrame(np.transpose(data))

    # justified start
    justified_start = math.floor(time_diffs[i][j] * FS) 

    df_all_ch = df\
      .iloc[:, 3:9]\
      .rename(columns={3:'Cz',4:'C3',5:'C4',6:'Fz',7:'F3',8:'F4'})

    for num, ch in enumerate(channels):
      col = num // 3
      row = num % 3

      df_notch_filtered = \
        filter_func.notchfilter(df_all_ch[ch], FS)

      plt_start: int = FS * (3 + 3 - 1) + justified_start
      plt_end: int = plt_start + FS * 4

      df_filtered = \
        filter_func.bandpass(df_notch_filtered, FS, bpf_Fp, bpf_Fs, 3, 40)[plt_start:plt_end]
      
      f, t, Sxx = signal.stft(df_filtered, FS, window=window, nperseg=nperseg)

      f_sum[num] += f
      t_sum[num] += t
      Sxx_sum[num] += 10 * np.log(pow(np.abs(Sxx),2))

Sxx_min_sum, Sxx_max_sum = 0, 0
 
for i in range(6):
  t_ave = t_sum[i]/30
  f_ave = f_sum[i]/30
  Sxx_ave = Sxx_sum[i]/30

  Sxx_min_sum += Sxx_ave[i].min()
  Sxx_max_sum += Sxx_ave[i].max()
  
  col: int = i // 3
  row: int = i % 3

  axes[col, row].pcolormesh(t_ave, f_ave, Sxx_ave, cmap='jet')
  #axes[col, row].axvline(x=3.192, ymin=0, ymax=125, color='magenta', linewidth=3)

  axes[col, row].axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)

vmin, vmax = Sxx_min_sum/6, Sxx_max_sum/6

fig_setup.adjust_stft(vmin, vmax, fig, axes)

plt.show()