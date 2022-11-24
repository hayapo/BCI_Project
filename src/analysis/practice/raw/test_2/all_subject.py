import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib import filter_func, fig_setup, calc_diff
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

board_id = BoardIds.CYTON_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)
FS: int = 250
channels = ['Cz', 'C3', 'C4','Fz', 'F3', 'F4']

# ファイルメタデータ
exp_type: str = 'practice'
test_flag: bool = True
test_num: int = 2
subject_total: int = 3
# to_data_dir: str = '../../../..'
# result_dir: str = os.path.join(to_data_dir, 'result')

# フィルタ関連の変数
bpf_Fp = np.array([3, 20])
bpf_Fs = np.array([1, 250])

# 図のセットアップ
fig, axes = fig_setup.setup_raw(fig_title="Raw(Practice): All Subject", channels=channels)

df_sum = pd.DataFrame(index=range(4*FS), columns=channels)
df_sum.fillna(0, inplace=True)

# 各被験者の各試行のスタート誤差を計算する
time_diffs, which_fast = calc_diff.timeDiff(test_num, exp_type, subject_total, 'result')
# time_diffs[i] + 
for i in range(subject_total):
  pathName = f'result/subject_{i+1}/practice/'
  if test_flag:
    pathName = f'result/test_{test_num}/subject_{i+1}/practice/'

  for j in range(10):
    fileName = f'subject_{i+1}_step_{j+1}.csv'
    data = DataFilter.read_file(pathName+fileName)
    df = pd.DataFrame(np.transpose(data))
    
    plt_start: int = FS * (3 + 5 - 1)
    plt_end: int = plt_start + FS * 4
    
    df_all_ch = df\
      .iloc[:, 3:9]\
      .rename(columns={3:'Cz',4:'C3',5:'C4',6:'Fz',7:'F3',8:'F4'})

    for num, ch in enumerate(channels):
      col = num // 3
      row = num % 3

      df_notch_filtered = \
        filter_func.notchfilter(df_all_ch[ch], FS)
      df_filtered = \
        filter_func.bandpass(df_notch_filtered, FS, bpf_Fp, bpf_Fs, 3, 40)[plt_start:plt_end]
      
      n = len(df_filtered)
      t = n // FS
      x = np.linspace(0, t, n)

      axes[col, row].plot(x, df_filtered, color='lightgray')
      df_sum[ch] = df_sum[ch] + df_filtered

for num, ch in enumerate(channels):
  col = num // 3
  row = num % 3

  axes[col, row].set_ylim(-50, 50)
  df_mean = df_sum[ch].div(subject_total * 10)
  axes[col, row].plot(x, df_mean, color='steelblue')
  #axes[col, row].axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)

plt.show()
