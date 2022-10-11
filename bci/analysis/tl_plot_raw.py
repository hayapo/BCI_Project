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
subject_num: str = 'subject_1'
exp_type: str = 'practice'
test_flag: bool = True

if test_flag:
  pathName = f'../result/test/{measure_date}/{subject_num}/{exp_type}/'
else:
  pathName = f'../result/{measure_date}/{subject_num}/{exp_type}/'

step: int = 1
fileName = f'{subject_num}_step_{step}.csv'

data = DataFilter.read_file(pathName+fileName)

df = pd.DataFrame(np.transpose(data))

pprint(df)

'''
TODO:全ステップの平均を取ってプロットするようにする
'''

# Recoding Channels
data_Cz = df[3]
data_C3 = df[4]
data_C4 = df[5]
data_Fz = df[6]
data_F3 = df[7]
data_F4 = df[8]

# データ数
N: int = len(df)

# 計測時間
T:int  = ( N // FS ) + 1

# x軸の作成
x = np.linspace(0, T, N)

bsf_Fp = np.array([5, 80])
bsf_Fs = np.array([50, 60])

bpf_Fp = np.array([8, 13])
bpf_Fs = np.array([5, 120])

data_filtered1 = filter_func.bandpass(data_C3, FS, bpf_Fp, bpf_Fs, 3, 40)
data_filtered2 = filter_func.bandpass(data_C4, FS, bpf_Fp, bpf_Fs, 3, 40)
data_filtered3 = filter_func.bandpass(data_F3, FS, bpf_Fp, bpf_Fs, 3, 40)
data_filtered4 = filter_func.bandpass(data_F4, FS, bpf_Fp, bpf_Fs, 3, 40)

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

ax1.plot(x, data_filtered1)
ax2.plot(x, data_filtered2)
ax3.plot(x, data_filtered3)
ax4.plot(x, data_filtered4)
plt.show()