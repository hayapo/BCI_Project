import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from lib import filter_func
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

board_id = BoardIds.CYTON_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)
FS: int = 250

# データ読み込み
measure_date: str = '2022-11-04'
subject_num: int = 3
exp_type: str = 'practice'
test_flag: bool = True

if test_flag:
  pathName = f'../../result/test/{measure_date}/subject_{subject_num}/{exp_type}/'
else:
  pathName = f'../../result/{measure_date}/subject_{subject_num}/{exp_type}/'

# フィルタ関連の変数
bpf_Fp = np.array([3, 20])
bpf_Fs = np.array([1, 250])

fig = plt.figure()
fig.suptitle("Suject 2: Raw(Practice)", fontsize=20)
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

ax1.set_xlabel('Time[s]', fontsize=15)
ax2.set_xlabel('Time[s]', fontsize=15)
ax3.set_xlabel('Time[s]', fontsize=15)
ax4.set_xlabel('Time[s]', fontsize=15)
ax5.set_xlabel('Time[s]', fontsize=15)
ax6.set_xlabel('Time[s]', fontsize=15)

ax1.set_ylabel('μV', fontsize=15)
ax2.set_ylabel('μV', fontsize=15)
ax3.set_ylabel('μV', fontsize=15)
ax4.set_ylabel('μV', fontsize=15)
ax5.set_ylabel('μV', fontsize=15)
ax6.set_ylabel('μV', fontsize=15)

cols = ['Cz', 'C3', 'C4','Fz', 'F3', 'F4']
df_sum = pd.DataFrame(index=range(6*FS),columns=cols)
df_sum.fillna(0,inplace=True)

steps:int = 10

for i in range(steps):

  fileName = f'subject_{subject_num}_step_{i+1}.csv'
  data = DataFilter.read_file(pathName+fileName)
  df = pd.DataFrame(np.transpose(data))

  plt_start:int = FS * (3 + 5 - 1)
  plt_end:int = plt_start + FS * 6

  data_Cz = df[3]
  data_C3 = df[4]
  data_C4 = df[5]
  data_Fz = df[6]
  data_F3 = df[7]
  data_F4 = df[8]

  Cz_notch_filtered = filter_func.notchfilter(data_Cz, FS)
  Cz_filtered = filter_func.bandpass(data_Cz, FS, bpf_Fp, bpf_Fs, 3, 40)[plt_start:plt_end]

  C3_notch_filtered = filter_func.notchfilter(data_C3, FS)
  C3_filtered = filter_func.bandpass(data_C3, FS, bpf_Fp, bpf_Fs, 3, 40)[plt_start:plt_end]

  C4_notch_filtered = filter_func.notchfilter(data_C4, FS)
  C4_filtered = filter_func.bandpass(data_C4, FS, bpf_Fp, bpf_Fs, 3, 40)[plt_start:plt_end]

  Fz_notch_filtered = filter_func.notchfilter(data_Fz, FS)
  Fz_filtered = filter_func.bandpass(data_Fz, FS, bpf_Fp, bpf_Fs, 3, 40)[plt_start:plt_end]

  F3_notch_filtered = filter_func.notchfilter(data_F3, FS)
  F3_filtered = filter_func.bandpass(data_F3, FS, bpf_Fp, bpf_Fs, 3, 40)[plt_start:plt_end]
  
  F4_notch_filtered = filter_func.notchfilter(data_F4, FS)
  F4_filtered = filter_func.bandpass(data_F4, FS, bpf_Fp, bpf_Fs, 3, 40)[plt_start:plt_end]

  n = len(Cz_filtered)
  t = n // FS
  x = np.linspace(0, t, n)

  ax1.plot(x, Cz_filtered, color='lightgray')
  ax2.plot(x, C3_filtered, color='lightgray')
  ax3.plot(x, C4_filtered, color='lightgray')
  ax4.plot(x, Fz_filtered, color='lightgray')
  ax5.plot(x, F3_filtered, color='lightgray')
  ax6.plot(x, F4_filtered, color='lightgray')

  df_sum['Cz'] = df_sum['Cz'] + Cz_filtered
  df_sum['C3'] = df_sum['C3'] + C3_filtered
  df_sum['C4'] = df_sum['C4'] + C4_filtered
  df_sum['Fz'] = df_sum['Fz'] + Fz_filtered
  df_sum['F3'] = df_sum['F3'] + F3_filtered
  df_sum['F4'] = df_sum['F4'] + F4_filtered

ax1.plot(x, df_sum['Cz'].div(10), color='steelblue')
ax2.plot(x, df_sum['C3'].div(10), color='steelblue')
ax3.plot(x, df_sum['C4'].div(10), color='steelblue')
ax4.plot(x, df_sum['Fz'].div(10), color='steelblue')
ax5.plot(x, df_sum['F3'].div(10), color='steelblue')
ax6.plot(x, df_sum['F4'].div(10), color='steelblue')

ax1.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax2.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax3.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax4.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax5.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax6.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)

plt.show()