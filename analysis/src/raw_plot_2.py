import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

path = '../GUI_data/2022-07-22_VR_KI/'
filename = 'run1_KI.txt'

df = pd.read_csv(path + filename, encoding="utf-8")
data1 = df[" EXG Channel 2"]
data1 = data1[1328:7690]

fs = 250
N = data1.shape[0]
t_0 = 1330 // 250
t = N // fs 
x = np.linspace(t_0, t, N)
T= 1 / fs

def highpass(x,samplerate, hp_fp, hp_fs, gpass, gstop):
  fn = samplerate / 2
  wp = hp_fp / fn
  ws = hp_fs / fn
  N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
  b, a = signal.butter(N, Wn, "high")           #フィルタ伝達関数の分子と分母を計算
  y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
  return y

def lowpass(x, samplerate, lp_fp, lp_fs, gpass, gstop):
  fn = samplerate / 2                           #ナイキスト周波数
  wp = lp_fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
  ws = lp_fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
  N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
  b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
  y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
  return y                                      #フィルタ後の信号を返す

hp_fp = 8       #通過域端周波数[Hz]
hp_fs = 1       #阻止域端周波数[Hz]

lp_fp = 14      #通過域端周波数[Hz]
lp_fs = 22      #阻止域端周波数[Hz]

gpass = 3       #通過域端最大損失[dB]
gstop = 40      #阻止域端最小損失[dB]

data_filt = highpass(data1, fs, hp_fp, hp_fs, gpass, gstop)
data_filt = lowpass(data_filt, fs, lp_fp, lp_fs, gpass, gstop)

yf = fft(np.array(data_filt))
x_f = np.linspace(0.0, 1.0/(2.0*T), N//2)
freqList = fftfreq(N, d=1.0/ fs)
amplitude = np.abs(yf)/(N/2)

plt.plot(freqList, amplitude, linestyle='-',label = "fft plot")

plt.axis([0, 125, 0, 2])
plt.xlabel("frequency [Hz]")
plt.ylabel("amplitude")

plt.show()