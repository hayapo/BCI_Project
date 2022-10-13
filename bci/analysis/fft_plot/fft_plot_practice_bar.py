import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
from pprint import pprint
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
fig.suptitle("Practice Step (Raw Data)", fontsize=20)

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

ax1.set_xlim([0,20])
ax2.set_xlim([0,20])
ax3.set_xlim([0,20])
ax4.set_xlim([0,20])
ax5.set_xlim([0,20])
ax6.set_xlim([0,20])

ax1.set_ylim([0,1.1])
ax2.set_ylim([0,1.1])
ax3.set_ylim([0,1.1])
ax4.set_ylim([0,1.1])
ax5.set_ylim([0,1.1])
ax6.set_ylim([0,1.1])

ax1.set_yticks(np.arange(0,1.1,0.1))

cols = ['Cz', 'C3', 'C4','Fz', 'F3', 'F4']
df_sum = pd.DataFrame(index=range(6*FS),columns=cols)
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

  Cz_filtered = filter_func.bandstop(data_Cz, FS, bsf_Fp, bsf_Fs, 3, 40)
  C3_filtered = filter_func.bandstop(data_C3, FS, bsf_Fp, bsf_Fs, 3, 40)
  C4_filtered = filter_func.bandstop(data_C4, FS, bsf_Fp, bsf_Fs, 3, 40)
  Fz_filtered = filter_func.bandstop(data_Fz, FS, bsf_Fp, bsf_Fs, 3, 40)
  F3_filtered = filter_func.bandstop(data_F3, FS, bsf_Fp, bsf_Fs, 3, 40)
  F4_filtered = filter_func.bandstop(data_C4, FS, bsf_Fp, bsf_Fs, 3, 40)

  plt_start:int = FS * (3 + 5 - 1)
  plt_end:int = plt_start + FS * 6

  n = len(Cz_filtered[plt_start:plt_end])
  t = n // FS
  x = np.linspace(0, t, n)

  Cz_yf = fft(np.array(Cz_filtered[plt_start:plt_end]))
  Cz_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
  Cz_freqList = fftfreq(n, d=1.0/ FS)
  Cz_amplitude = np.abs(Cz_yf)/(n/2)
  Cz_amp_max = np.max(Cz_amplitude)

  C3_yf = fft(np.array(C3_filtered[plt_start:plt_end]))
  C3_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
  C3_freqList = fftfreq(n, d=1.0/ FS)
  C3_amplitude = np.abs(C3_yf)/(n/2)
  C3_amp_max = np.max(C3_amplitude)

  C4_yf = fft(np.array(C4_filtered[plt_start:plt_end]))
  C4_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
  C4_freqList = fftfreq(n, d=1.0/ FS)
  C4_amplitude = np.abs(C4_yf)/(n/2)
  C4_amp_max = np.max(C4_amplitude)

  Fz_yf = fft(np.array(Fz_filtered[plt_start:plt_end]))
  Fz_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
  Fz_freqList = fftfreq(n, d=1.0/ FS)
  Fz_amplitude = np.abs(Fz_yf)/(n/2)
  Fz_amp_max = np.max(Fz_amplitude)

  F3_yf = fft(np.array(F3_filtered[plt_start:plt_end]))
  F3_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
  F3_freqList = fftfreq(n, d=1.0/ FS)
  F3_amplitude = np.abs(F3_yf)/(n/2)
  F3_amp_max = np.max(F3_amplitude)

  F4_yf = fft(np.array(F4_filtered[plt_start:plt_end]))
  F4_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
  F4_freqList = fftfreq(n, d=1.0/ FS)
  F4_amplitude = np.abs(F4_yf)/(n/2)
  F4_amp_max = np.max(F4_amplitude)

  ax1.bar(Cz_freqList, Cz_amplitude/Cz_amp_max, color='lightgray')
  ax2.bar(C3_freqList, C3_amplitude/C3_amp_max, color='lightgray')
  ax3.bar(C4_freqList, C4_amplitude/C4_amp_max, color='lightgray')
  ax4.bar(Fz_freqList, Fz_amplitude/Fz_amp_max, color='lightgray')
  ax5.bar(F3_freqList, F3_amplitude/F3_amp_max, color='lightgray')
  ax6.bar(F4_freqList, F4_amplitude/F4_amp_max, color='lightgray')


  df_sum['Cz'] = df_sum['Cz'] + Cz_filtered[plt_start:plt_end]
  df_sum['C3'] = df_sum['C3'] + C3_filtered[plt_start:plt_end]
  df_sum['C4'] = df_sum['C4'] + C4_filtered[plt_start:plt_end]
  df_sum['Fz'] = df_sum['Fz'] + Fz_filtered[plt_start:plt_end]
  df_sum['F3'] = df_sum['F3'] + F3_filtered[plt_start:plt_end]
  df_sum['F4'] = df_sum['F4'] + F4_filtered[plt_start:plt_end]

Cz_sum_yf = fft(np.array(df_sum['Cz'].div(10)))
Cz_sum_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
Cz_sum_freqList = fftfreq(n, d=1.0/ FS)
Cz_sum_amplitude = np.abs(Cz_sum_yf)/(n/2)
Cz_sum_amp_max = np.max(Cz_sum_amplitude)


C3_sum_yf = fft(np.array(df_sum['C3'].div(10)))
C3_sum_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
C3_sum_freqList = fftfreq(n, d=1.0/ FS)
C3_sum_amplitude = np.abs(C3_sum_yf)/(n/2)
C3_sum_amp_max = np.max(C3_sum_amplitude)

C4_sum_yf = fft(np.array(df_sum['C4'].div(10)))
C4_sum_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
C4_sum_freqList = fftfreq(n, d=1.0/ FS)
C4_sum_amplitude = np.abs(C4_sum_yf)/(n/2)
C4_sum_amp_max = np.max(C4_sum_amplitude)

Fz_sum_yf = fft(np.array(df_sum['Fz'].div(10)))
Fz_sum_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
Fz_sum_freqList = fftfreq(n, d=1.0/ FS)
Fz_sum_amplitude = np.abs(Fz_sum_yf)/(n/2)
Fz_sum_amp_max = np.max(Fz_sum_amplitude)

F3_sum_yf = fft(np.array(df_sum['F3'].div(10)))
F3_sum_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
F3_sum_freqList = fftfreq(n, d=1.0/ FS)
F3_sum_amplitude = np.abs(F3_sum_yf)/(n/2)
F3_sum_amp_max = np.max(F3_sum_amplitude)

F4_sum_yf = fft(np.array(df_sum['F4'].div(10)))
F4_sum_xf = np.linspace(0.0, 1.0/(2.0/FS), n//2)
F4_sum_freqList = fftfreq(n, d=1.0/ FS)
F4_sum_amplitude = np.abs(F4_sum_yf)/(n/2)
F4_sum_amp_max = np.max(F4_sum_amplitude)

ax1.bar(Cz_sum_freqList, Cz_sum_amplitude/Cz_sum_amp_max, color='steelblue')
ax2.bar(C3_sum_freqList, C3_sum_amplitude/C3_sum_amp_max, color='steelblue')
ax3.bar(C4_sum_freqList, C4_sum_amplitude/C4_sum_amp_max, color='steelblue')
ax4.bar(Fz_sum_freqList, Fz_sum_amplitude/Fz_sum_amp_max, color='steelblue')
ax5.bar(F3_sum_freqList, F3_sum_amplitude/F3_sum_amp_max, color='steelblue')
ax6.bar(F4_sum_freqList, F4_sum_amplitude/F4_sum_amp_max, color='steelblue')

plt.show()