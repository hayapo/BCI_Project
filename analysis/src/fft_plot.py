from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '../GUI_data/2022-07-11_nonVR_KI/'
filename = 'KI_swim.txt'

df = pd.read_csv(path + filename, encoding="utf-8")
data = df[" EXG Channel 0"]

data = data[2000:]

fs = 250
N = data.shape[0]
T = 1/fs

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

data_filtered = bpf(data, fs, bpf_Fp, bpf_Fs, 3, 40)

yf = fft(np.array(data_filtered))
x_f = np.linspace(0.0, 1.0/(2.0*T), N//2)
freqList = fftfreq(N, d=1.0/ fs)
amplitude = np.abs(yf)/(N/2)

plt.plot(freqList, amplitude, linestyle='-',label = "fft plot")

plt.axis([0, 125, 0, 2])
plt.xlabel("frequency [Hz]")
plt.ylabel("amplitude")

plt.show()