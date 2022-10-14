import numpy as np
from pandas import Series
from scipy import signal

def highpass(x,samplerate, hp_fp, hp_fs, gpass, gstop):
  '''
  ハイパスフィルタ \n
  samplerare で サンプリング周波数 \n
  bs_fp で 通過域端周波数 \n
  bs_fs で 阻止域端周波数 を設定 \n
  
  Example
  -----
  data_filt = highpass(data, fs, 10, 5, 3, 40) 
  '''
  fn = samplerate / 2                              #ナイキスト周波数
  wp = hp_fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
  ws = hp_fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
  N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
  b, a = signal.butter(N, Wn, "high")           #フィルタ伝達関数の分子と分母を計算
  y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
  return y

def lowpass(x, samplerate, lp_fp, lp_fs, gpass, gstop):
  '''
  ローパスフィルタ \n
  samplerare で サンプリング周波数 \n
  lp_fp で 通過域端周波数 を設定 \n
  lp_fs で 阻止域端周波数 を設定 \n
  
  '''
  fn = samplerate / 2                              #ナイキスト周波数
  wp = lp_fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
  ws = lp_fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
  N, Wn = signal.buttord(wp, ws, gpass, gstop)     #オーダーとバターワースの正規化周波数を計算
  b, a = signal.butter(N, Wn, "low")               #フィルタ伝達関数の分子と分母を計算
  y = signal.filtfilt(b, a, x)                     #信号に対してフィルタをかける
  return y                                         #フィルタ後の信号を返す

def bandstop(x: Series, samplerate: int , bs_fp: np.ndarray, bs_fs: np.ndarray, gpass: int, gstop: int) -> Series:
  '''
  バンドストップフィルタ \n
  samplerare で サンプリング周波数 \n
  bs_fp で 通過周波数域 \n
  bs_fs で 阻止周波数領域 を設定 \n
  
  '''
  fn = samplerate / 2  # ナイキスト周波数
  wp = bs_fp / fn  # ナイキスト周波数で通過域端周波数を正規化
  ws = bs_fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
  n, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
  b, a = signal.butter(n, Wn, "bandstop")  # フィルタ伝達関数の分子と分母を計算
  y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける

  return y

def bandpass(x: Series, samplerate: int , bp_fp: np.ndarray, bp_fs: np.ndarray, gpass: int, gstop: int) -> Series:
  '''
  バンドパスフィルタ \n
  samplerare で サンプリング周波数 \n
  bs_fp で 通過周波数域 \n
  bs_fs で 阻止周波数領域 を設定 \n
  
  '''
  fn = samplerate / 2  # ナイキスト周波数
  wp = bp_fp / fn  # ナイキスト周波数で通過域端周波数を正規化
  ws = bp_fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
  n, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
  b, a = signal.butter(n, Wn, "bandpass")  # フィルタ伝達関数の分子と分母を計算
  y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける

  return y

def notchfilter(x: Series, samplerate: int) -> Series:
  '''
  50-60Hzのノッチフィルタ\n
  '''
  ws = np.array([58.0, 62.0])

  fn = samplerate / 2  # ナイキスト周波数

  b, a = signal.butter(4, ws/fn, "bandstop")  # フィルタ伝達関数の分子と分母を計算
  y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける

  return y
