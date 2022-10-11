import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

path = '../GUI_data/2022-07-11_nonVR_KI/'
filename = 'KI_normal.txt'

df = pd.read_csv(path + filename, encoding="utf-8")
data = df[" EXG Channel 0"]
#data = data[500:]

fs = 250

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

bsf_Fp = np.array([40, 70])
bsf_Fs = np.array([50, 60])

bpf_Fp = np.array([5, 50])
bpf_Fs = np.array([0, 60])

data_filtered = bsf(data, fs, bsf_Fp, bsf_Fs, 3, 40)

window = signal.windows.hamming(125)

f, t, Zxx = signal.stft(data_filtered, fs=fs, window=window, nperseg=125, noverlap=64, nfft=125)

plt.pcolormesh(t, f, np.abs(Zxx))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')

plt.show()