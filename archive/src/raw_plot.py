import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import sys
sys.path.append('../')
from lib import filter_func

fig = plt.figure()
fig.suptitle("Kinesthetic Image (walk)", fontsize=20)
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

path = '../GUI_data/2022-07-22_VR_KI/'
filename = 'run1_KI.txt'

df = pd.read_csv(path + filename, encoding="utf-8")
data1 = df[" EXG Channel 2"]
data1 = data1[1328:7690]

data2 = df[" EXG Channel 3"]
data2 = data2[1328:7690]

data3 = df[" EXG Channel 4"]
data3 = data3[1328:7690]

data4 = df[" EXG Channel 5"]
data4 = data4[1328:7690]

fs = 250
N = data1.shape[0]
t_0 = 1330 // 250
t = N // fs 
x = np.linspace(t_0, t, N)


bsf_Fp = np.array([5, 80])
bsf_Fs = np.array([50, 60])

bpf_Fp = np.array([8, 13])
bpf_Fs = np.array([5, 120])

data_filtered1 = filter_func.bandpass(data1, fs, bpf_Fp, bpf_Fs, 3, 40)
data_filtered2 = filter_func.bandpass(data2, fs, bpf_Fp, bpf_Fs, 3, 40)
data_filtered3 = filter_func.bandpass(data3, fs, bpf_Fp, bpf_Fs, 3, 40)
data_filtered4 = filter_func.bandpass(data4, fs, bpf_Fp, bpf_Fs, 3, 40)

ax1.set_title('F3')
ax2.set_title('F4')
ax3.set_title('C3')
ax4.set_title('C4')

ax1.set_xlabel('Time [s]', fontsize=15)
ax2.set_xlabel('Time [s]', fontsize=15)
ax3.set_xlabel('Time [s]', fontsize=15)
ax4.set_xlabel('Time [s]', fontsize=15)

ax1.set_ylabel('μV', fontsize=15)
ax2.set_ylabel('μV', fontsize=15)
ax3.set_ylabel('μV', fontsize=15)
ax4.set_ylabel('μV', fontsize=15)

# plt.xticks(np.arange(0, t+3, 1))

#print(t)
ax1.plot(x, data_filtered1)
ax2.plot(x, data_filtered2)
ax3.plot(x, data_filtered3)
ax4.plot(x, data_filtered4)
plt.show()