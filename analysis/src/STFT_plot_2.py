import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

path = '../GUI_data/2022-07-22_VR_KI/'
filename = 'run1_KI.txt'

df = pd.read_csv(path + filename, encoding="utf-8")
data = df[" EXG Channel 3"]
data = data[1327:7690]

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

bsf_Fp = np.array([5, 80])
bsf_Fs = np.array([50, 60])

bpf_Fp = np.array([8, 14])
bpf_Fs = np.array([5, 120])

data_filtered = bpf(data, fs, bpf_Fp, bpf_Fs, 3, 40)
N = data.shape[0]

amp = 2 * np.sqrt(2)

nfft = 120
Pxx1, freqs1, bins1, im1 = plt.specgram(data_filtered, Fs=fs, NFFT=nfft, noverlap=nfft//2, cmap='jet', mode='psd')

cbar = plt.colorbar()  # カラーバー表示のため追加

plt.ylim(0,20)
plt.yticks(np.arange(0, 20, 1))
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.show()
