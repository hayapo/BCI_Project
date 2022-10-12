from turtle import color
import brainflow
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
measure_date: str = '2022-10-11'
subject_num: int = 1
exp_type: str = 'practice'
test_flag: bool = True

if test_flag:
  pathName = f'../../result/test/{measure_date}/subject_{subject_num}/{exp_type}/'
else:
  pathName = f'../../result/{measure_date}/subject_{subject_num}/{exp_type}/'


bsf_Fp = np.array([5, 80])
bsf_Fs = np.array([50, 60])

bpf_Fp = np.array([8, 13])
bpf_Fs = np.array([5, 120])


fig = plt.figure()
fig.suptitle("Raw Data Plot", fontsize=20)
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

ax1.set_xlabel('Time [s]', fontsize=15)
ax2.set_xlabel('Time [s]', fontsize=15)
ax3.set_xlabel('Time [s]', fontsize=15)
ax4.set_xlabel('Time [s]', fontsize=15)
ax5.set_xlabel('Time [s]', fontsize=15)
ax6.set_xlabel('Time [s]', fontsize=15)

ax1.set_ylabel('μV', fontsize=15)
ax2.set_ylabel('μV', fontsize=15)
ax3.set_ylabel('μV', fontsize=15)
ax4.set_ylabel('μV', fontsize=15)
ax5.set_ylabel('μV', fontsize=15)
ax6.set_ylabel('μV', fontsize=15)

cols = ['Cz', 'C3', 'C4','Fz', 'F3', 'F4']
df_sum = pd.DataFrame(index=range(10*FS),columns=cols)
df_sum.fillna(0,inplace=True)

steps:int = 10

for i in range(steps):

  fileName = f'subject_{subject_num}_step_{i+1}.csv'
  data = DataFilter.read_file(pathName+fileName)
  df = pd.DataFrame(np.transpose(data))

  data_Cz = df[3]
  data_C3 = df[4]
  data_C4 = df[5]
  data_Fz = df[6]
  data_F3 = df[7]
  data_F4 = df[8]

  data_filtered1 = filter_func.bandpass(data_Cz, FS, bpf_Fp, bpf_Fs, 3, 40)
  data_filtered2 = filter_func.bandpass(data_C3, FS, bpf_Fp, bpf_Fs, 3, 40)
  data_filtered3 = filter_func.bandpass(data_C4, FS, bpf_Fp, bpf_Fs, 3, 40)
  data_filtered4 = filter_func.bandpass(data_Fz, FS, bpf_Fp, bpf_Fs, 3, 40)
  data_filtered5 = filter_func.bandpass(data_F3, FS, bpf_Fp, bpf_Fs, 3, 40)
  data_filtered6 = filter_func.bandpass(data_C4, FS, bpf_Fp, bpf_Fs, 3, 40)

  plt_start:int = FS * 3
  plt_end:int = FS * (3 + 10)

  n = len(data_filtered1[plt_start:plt_end])
  t = (n // FS) + 1
  x = np.linspace(0, t, n)

  df_sum['Cz'] = df_sum['Cz'] + data_filtered1[plt_start:plt_end]
  df_sum['C4'] = df_sum['C4'] + data_filtered2[plt_start:plt_end]
  df_sum['C3'] = df_sum['C3'] + data_filtered3[plt_start:plt_end]
  df_sum['Fz'] = df_sum['Fz'] + data_filtered4[plt_start:plt_end]
  df_sum['F3'] = df_sum['F3'] + data_filtered5[plt_start:plt_end]
  df_sum['F4'] = df_sum['F4'] + data_filtered6[plt_start:plt_end]

  ax1.plot(x, data_filtered1[plt_start:plt_end], color='lightgray')
  ax2.plot(x, data_filtered2[plt_start:plt_end], color='lightgray')
  ax3.plot(x, data_filtered3[plt_start:plt_end], color='lightgray')
  ax4.plot(x, data_filtered4[plt_start:plt_end], color='lightgray')
  ax5.plot(x, data_filtered5[plt_start:plt_end], color='lightgray')
  ax6.plot(x, data_filtered6[plt_start:plt_end], color='lightgray')

ax1.plot(x, df_sum['Cz'].div(10), color='steelblue')
ax2.plot(x, df_sum['C3'].div(10), color='steelblue')
ax3.plot(x, df_sum['C4'].div(10), color='steelblue')
ax4.plot(x, df_sum['Fz'].div(10), color='steelblue')
ax5.plot(x, df_sum['F3'].div(10), color='steelblue')
ax6.plot(x, df_sum['F4'].div(10), color='steelblue')

pprint(df_sum)

plt.show()