import sys
sys.path.append('../../')
from lib import filter_func
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

FS = 250
bpf_Fp = np.array([8, 13])
bpf_Fs = np.array([5, 250])

# データ読み込み
measure_date: str = '2022-10-11'
subject_num: int = 1
exp_type: str = 'practice'
test_flag: bool = True

if test_flag:
  pathName = f'../../result/test/{measure_date}/subject_{subject_num}/{exp_type}/'
else:
  pathName = f'../../result/{measure_date}/subject_{subject_num}/{exp_type}/'

fileName = f'subject_{subject_num}_step_{1}.csv'
data = DataFilter.read_file(pathName+fileName)
df = pd.DataFrame(np.transpose(data))
data_Cz = df[3]
plt_start:int = FS * (3 + 5 - 1)
plt_end:int = plt_start + FS * 6
Cz_filtered = filter_func.bandpass(data_Cz, FS, bpf_Fp, bpf_Fs, 3, 40)[plt_start:plt_end]

N = len(Cz_filtered)
T = 1/FS

yf = fft(np.array(Cz_filtered))
x_f = np.linspace(0.0, 1.0/T, N)
amplitude = np.abs(yf)

plt.plot(x_f, amplitude/np.max(amplitude), linestyle='-',label = "fft plot")

plt.axis([0, 125, 0, 1])
plt.xlabel("frequency [Hz]")
plt.ylabel("amplitude")

plt.show()