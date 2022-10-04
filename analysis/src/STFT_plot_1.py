import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
types = 'Wavelet'
ch_num = 2
task_num  = 3

path = '../GUI_data/2022-07-22_VR_KI/'
filename = 'run1_KI.txt'

df = pd.read_csv(path + filename, encoding="utf-8")
data1 = df[" EXG Channel 2"]
data1 = data1[:7690]

data2 = df[" EXG Channel 3"]
data2 = data2[:7690]

data3 = df[" EXG Channel 4"]
data3 = data3[:7690]

data4 = df[" EXG Channel 5"]
data4 = data4[:7690]

fs = 250

fig = plt.figure()
fig.suptitle("Kinesthetic Image (walk)", fontsize=20)
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

N = data1.shape[0]

duration = N // fs
amp = 2 * np.sqrt(2)

# バンドストップフィルタ
def bsf(x, samplerate, Fp, Fs, gpass, gstop):
    fn = samplerate / 2  # ナイキスト周波数
    wp = Fp / fn  # ナイキスト周波数で通過域端周波数を正規化
    ws = Fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
    n, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(n, Wn, "bandstop")  # フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
    return y

# バンドパスフィルタ
def bpf(x, samplerate, Fp, Fs, gpass, gstop):
    fn = samplerate / 2  # ナイキスト周波数
    wp = Fp / fn  # ナイキスト周波数で通過域端周波数を正規化
    ws = Fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
    n, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(n, Wn, "band")  # フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
    return y

bsf_Fp = np.array([5, 80])
bsf_Fs = np.array([50, 60])

bpf_Fp = np.array([8, 14])
bpf_Fs = np.array([5, 120])

data_filtered1 = bsf(data1, fs, bsf_Fp, bsf_Fs, 3, 40)
data_filtered2 = bsf(data2, fs, bsf_Fp, bsf_Fs, 3, 40)
data_filtered3 = bsf(data3, fs, bsf_Fp, bsf_Fs, 3, 40)
data_filtered4 = bsf(data4, fs, bsf_Fp, bsf_Fs, 3, 40)

window = signal.windows.hamming(125)
nfft = 150 

f_1, t_1, Sxx_1 = signal.spectrogram(data_filtered1, fs, window=window, noverlap=8)
f_2, t_2, Sxx_2 = signal.spectrogram(data_filtered2, fs, window=window, noverlap=8)
f_3, t_3, Sxx_3 = signal.spectrogram(data_filtered3, fs, window=window, noverlap=8)
f_4, t_4, Sxx_4 = signal.spectrogram(data_filtered4, fs, window=window, noverlap=8)

ax1.pcolormesh(t_1, f_1, 10*np.log(Sxx_1)) 
ax2.pcolormesh(t_2, f_2, 10*np.log(Sxx_2)) 
ax3.pcolormesh(t_3, f_3, 10*np.log(Sxx_3)) 
ax4.pcolormesh(t_4, f_4, 10*np.log(Sxx_4)) 

ax1.set_xlim(0, duration)
ax2.set_xlim(0, duration)
ax3.set_xlim(0, duration)
ax4.set_xlim(0, duration)

ax1.set_ylim(0, 25)
ax2.set_ylim(0, 25)
ax3.set_ylim(0, 25)
ax4.set_ylim(0, 25)

ax1.set_yticks(np.arange(0, 25, 1))
ax2.set_yticks(np.arange(0, 2, 1))
ax3.set_yticks(np.arange(0, 21, 1))
ax4.set_yticks(np.arange(0, 21, 1))

ax1.set_title('F3')
ax2.set_title('F4')
ax3.set_title('C3')
ax4.set_title('C4')

ax1.set_xlabel('Time [s]', fontsize=15)
ax2.set_xlabel('Time [s]', fontsize=15)
ax3.set_xlabel('Time [s]', fontsize=15)
ax4.set_xlabel('Time [s]', fontsize=15)

ax1.set_ylabel('Frequency [Hz]', fontsize=15)
ax2.set_ylabel('Frequency [Hz]', fontsize=15)
ax3.set_ylabel('Frequency [Hz]', fontsize=15)
ax4.set_ylabel('Frequency [Hz]', fontsize=15)

axs = plt.gcf().get_axes()

for ax in axs:
    ax.axvline(x=5.31, ymin=0, ymax=125,color='r')
    ax.axvline(x=15.51, ymin=0, ymax=125,color='r')
    ax.axvline(x=20.67, ymin=0, ymax=125,color='r')
    ax.axvline(x=30.76, ymin=0, ymax=125,color='r')
    
    ax.axvspan(0.0, 5.31, color = "white")
    ax.axvspan(15.451, 20.67, color = "white")

#ax1.set_ylim([7, 15])
#plt.title('Time-Frequency')
#cbar = plt.colorbar()  # カラーバー表示のため追加
#cbar.ax.set_ylabel("Intensity [dB]")  # カラーバーの名称表示のため追加
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')

plt.show()
