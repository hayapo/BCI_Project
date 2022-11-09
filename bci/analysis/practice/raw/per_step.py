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

FS: int = 250

board_id = BoardIds.CYTON_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)

# データ読み込み
subject_num: int = 6
exp_type: str = 'practice'
test_flag: bool = True
test_num: int = 1

if test_flag:
  pathName = f'../../result/test_{test_num}/subject_{subject_num}/{exp_type}/'
else:
  pathName = f'../../result/subject_{subject_num}/{exp_type}/'

# フィルタ関連の変数
bpf_Fp = np.array([3, 20])
bpf_Fs = np.array([1, 250])

fig = plt.figure()
fig.suptitle("Suject 6: Raw(Actual) Step 16", fontsize=20)
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
df_sum = pd.DataFrame(index=range(6*FS),columns=cols)
df_sum.fillna(0,inplace=True)

steps:int = 20

fileName = f'subject_{subject_num}_step_{10}.csv'
data = DataFilter.read_file(pathName+fileName)
df = pd.DataFrame(np.transpose(data))

plt_start:int = FS * (3 + 5 - 1)
plt_end:int = plt_start +  FS * 6

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

ax1.plot(x, Cz_filtered, color='steelblue')
ax2.plot(x, C3_filtered, color='steelblue')
ax3.plot(x, C4_filtered, color='steelblue')
ax4.plot(x, Fz_filtered, color='steelblue')
ax5.plot(x, F3_filtered, color='steelblue')
ax6.plot(x, F4_filtered, color='steelblue')

ax1.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax2.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax3.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax4.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax5.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)
ax6.axvline(x=1, ymin=0, ymax=125, color='magenta', linewidth=2)

plt.show()