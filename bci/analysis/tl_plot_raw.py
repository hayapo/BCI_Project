from turtle import color
import brainflow
import sys
sys.path.append('../')
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
  pathName = f'../result/test/{measure_date}/subject_{subject_num}/{exp_type}/'
else:
  pathName = f'../result/{measure_date}/subject_{subject_num}/{exp_type}/'

# step: int = 1
# fileName = f'subject_{subject_num}_step_{step}.csv'

# data = DataFilter.read_file(pathName+fileName)

# df = pd.DataFrame(np.transpose(data))

# pprint(df)

'''
TODO:全ステップの平均を取ってプロットするようにする
'''

# Recoding Channels
# data_Cz = df[3]
# data_C3 = df[4]
# data_C4 = df[5]
# data_Fz = df[6]
# data_F3 = df[7]
# data_F4 = df[8]

# # データ数
# N: int = len(df)

# # 計測時間
# T:int  = ( N // FS ) + 1

# # x軸の作成
# x = np.linspace(0, T, N)

bsf_Fp = np.array([5, 80])
bsf_Fs = np.array([50, 60])

bpf_Fp = np.array([8, 13])
bpf_Fs = np.array([5, 120])


fig = plt.figure()
fig.suptitle("Raw Data Plot", fontsize=20)
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.set_title('C3')
ax2.set_title('C4')
ax3.set_title('F3')
ax4.set_title('F4')

ax1.set_xlabel('Time [s]', fontsize=15)
ax2.set_xlabel('Time [s]', fontsize=15)
ax3.set_xlabel('Time [s]', fontsize=15)
ax4.set_xlabel('Time [s]', fontsize=15)

ax1.set_ylabel('μV', fontsize=15)
ax2.set_ylabel('μV', fontsize=15)
ax3.set_ylabel('μV', fontsize=15)
ax4.set_ylabel('μV', fontsize=15)

cols = ['C3', 'C4', 'F3', 'F4']
df_sum = pd.DataFrame(index=range(2501),columns=cols)
df_sum.fillna(0,inplace=True)

if exp_type == 'practice':
  steps:int = 10
else:
  steps:int = 20

for i in range(steps):

  fileName = f'subject_{subject_num}_step_{i+1}.csv'
  data = DataFilter.read_file(pathName+fileName)
  df = pd.DataFrame(np.transpose(data))

  # データ数
  N:int = len(df)

  except_n = (N - 2500) // 2

  # 計測時間
  T:int = (N // FS) + 1

  # x軸の作成
  x = np.linspace(0, T, N)

  data_Cz = df[3].iloc[except_n:N-except_n]
  n = len(data_Cz)
  t = (n // FS) + 1
  print(n)
  x_cz = np.linspace(0, t, n)

  data_C3 = df[4].iloc[except_n:N-except_n]
  n = len(data_C3)
  t = (n // FS) + 1
  x_c3 = np.linspace(0, t, n)

  data_C4 = df[5].iloc[except_n:N-except_n]
  n = len(data_C4)
  t = (n // FS) + 1
  x_c4 = np.linspace(0, t, n)

  data_Fz = df[6].iloc[except_n:N-except_n]
  n = len(data_Fz)
  t = (n // FS) + 1
  x_fz = np.linspace(0, t, n)

  data_F3 = df[7].iloc[except_n:N-except_n]
  n = len(data_F3)
  t = (n // FS) + 1
  x_f3 = np.linspace(0, t, n)

  data_F4 = df[8].iloc[except_n:N-except_n]
  n = len(data_F4)
  t = (n // FS) + 1
  x_f4 = np.linspace(0, t, n)

  df_sum['C3'] = df_sum['C3']
  df_sum['C4'] = df_sum['C4']
  df_sum['F3'] = df_sum['F3']
  df_sum['F4'] = df_sum['F4']

  data_filtered1 = filter_func.bandpass(data_C3, FS, bpf_Fp, bpf_Fs, 3, 40)
  data_filtered2 = filter_func.bandpass(data_C4, FS, bpf_Fp, bpf_Fs, 3, 40)
  data_filtered3 = filter_func.bandpass(data_F3, FS, bpf_Fp, bpf_Fs, 3, 40)
  data_filtered4 = filter_func.bandpass(data_F4, FS, bpf_Fp, bpf_Fs, 3, 40)

  ax1.plot(data_filtered1, color='lightgray')
  ax2.plot(data_filtered2, color='lightgray')
  ax3.plot(data_filtered3, color='lightgray')
  ax4.plot(data_filtered4, color='lightgray')

ax1.plot(df_sum['C3'], color='blue')
ax2.plot(df_sum['C4'], color='blue')
ax3.plot(df_sum['F3'], color='blue')
ax4.plot(df_sum['F4'], color='blue')

pprint(df_sum)

plt.show()