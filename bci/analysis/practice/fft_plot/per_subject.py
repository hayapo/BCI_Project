import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
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



fig = plt.figure()
fig.suptitle("Subject 1: FFT(Practice)", fontsize=20)
fig.supxlabel("frequency [Hz]", fontsize=15)
fig.supylabel("Power Spectrum [$μV^{2}$]", fontsize=15)

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

ax1.axis([2.5, 20, 0, 20])
ax2.axis([2.5, 20, 0, 20])
ax3.axis([2.5, 20, 0, 20])
ax4.axis([2.5, 20, 0, 20])
ax5.axis([2.5, 20, 0, 20])
ax6.axis([2.5, 20, 0, 20])

cols = ['Cz', 'C3', 'C4','Fz', 'F3', 'F4']
df_sum = pd.DataFrame(index=range(2048),columns=cols)
df_sum.fillna(0,inplace=True)

bpf_Fp = np.array([3, 20])
bpf_Fs = np.array([1, 250])

steps:int = 10

for i in range(steps):

  fileName = f'subject_{subject_num}_step_{i+1}.csv'
  data = DataFilter.read_file(pathName+fileName)
  df = pd.DataFrame(np.transpose(data))

  fft_start:int = FS * (3 + 5 - 1) - 548
  fft_end:int = fft_start + (FS * 6 + 548)

  data_Cz = df[3]
  data_C3 = df[4]
  data_C4 = df[5]
  data_Fz = df[6]
  data_F3 = df[7]
  data_F4 = df[8]

  #Cz_notch_filtered = filter_func.notchfilter(data_Cz, FS)
  Cz_filtered = filter_func.bandpass(data_Cz, FS, bpf_Fp, bpf_Fs, 3, 40)

  #C3_notch_filtered = filter_func.notchfilter(data_C3, FS)
  C3_filtered = filter_func.bandpass(data_C3, FS, bpf_Fp, bpf_Fs, 3, 40)

  #C4_notch_filtered = filter_func.notchfilter(data_C4, FS)
  C4_filtered = filter_func.bandpass(data_C4, FS, bpf_Fp, bpf_Fs, 3, 40)

  #Fz_notch_filtered = filter_func.notchfilter(data_Fz, FS)
  Fz_filtered = filter_func.bandpass(data_Fz, FS, bpf_Fp, bpf_Fs, 3, 40)

  #F3_notch_filtered = filter_func.notchfilter(data_F3, FS)
  F3_filtered = filter_func.bandpass(data_F3, FS, bpf_Fp, bpf_Fs, 3, 40)
  
  #F4_notch_filtered = filter_func.notchfilter(data_F4, FS)
  F4_filtered = filter_func.bandpass(data_F4, FS, bpf_Fp, bpf_Fs, 3, 40)

  Cz_yf = fft(np.array(Cz_filtered[fft_start:fft_end]))
  n = len(Cz_yf)
  Cz_amplitude = np.abs(Cz_yf)/(n/2)
  Cz_power = pow(Cz_amplitude, 2)

  C3_yf = fft(np.array(C3_filtered[fft_start:fft_end]))
  C3_amplitude = np.abs(C3_yf)/(n/2)
  C3_power = pow(C3_amplitude, 2)

  C4_yf = fft(np.array(C4_filtered[fft_start:fft_end]))
  C4_amplitude = np.abs(C4_yf)/(n/2)
  C4_power = pow(C4_amplitude, 2)

  Fz_yf = fft(np.array(Fz_filtered[fft_start:fft_end]))
  Fz_amplitude = np.abs(Fz_yf)/(n/2)
  Fz_power = pow(Fz_amplitude, 2)

  F3_yf = fft(np.array(F3_filtered[fft_start:fft_end]))
  F3_amplitude = np.abs(F3_yf)/(n/2)
  F3_power = pow(F3_amplitude, 2)

  F4_yf = fft(np.array(F4_filtered[fft_start:fft_end]))
  F4_amplitude = np.abs(F4_yf)/(n/2)
  F4_power = pow(F4_amplitude, 2)

  x = np.linspace(0, FS, len(F4_power))

  ax1.plot(x, Cz_power, color='lightgray')
  ax2.plot(x, C3_power, color='lightgray')
  ax3.plot(x, C4_power, color='lightgray')
  ax4.plot(x, Fz_power, color='lightgray')
  ax5.plot(x, F3_power, color='lightgray')
  ax6.plot(x, F4_power, color='lightgray')

  df_sum['Cz'] = df_sum['Cz'] + Cz_power
  df_sum['C3'] = df_sum['C3'] + C3_power
  df_sum['C4'] = df_sum['C4'] + C4_power
  df_sum['Fz'] = df_sum['Fz'] + Fz_power
  df_sum['F3'] = df_sum['F3'] + F3_power
  df_sum['F4'] = df_sum['F4'] + F4_power

ax1.plot(x, df_sum['Cz'].div(10), color='steelblue')
ax2.plot(x, df_sum['C3'].div(10), color='steelblue')
ax3.plot(x, df_sum['C4'].div(10), color='steelblue')
ax4.plot(x, df_sum['Fz'].div(10), color='steelblue')
ax5.plot(x, df_sum['F3'].div(10), color='steelblue')
ax6.plot(x, df_sum['F4'].div(10), color='steelblue')

plt.show()