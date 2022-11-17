import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors

def init_fig(fig_title:str):
  fig, axes = plt.subplots(2, 3, squeeze=True)
  fig.suptitle(fig_title, fontsize=20)
  return fig, axes

def setup_raw(fig_title:str, channels:list[str]):
  fig, axes = init_fig(fig_title)

  for col in range(2):
    for row in range(3):
      ch: str = channels[3*col+row]
      axes[col, row].set_title(ch)
      axes[col, row].set_xlabel('Time [s]', fontsize=15)
      axes[col, row].set_ylabel('μV', fontsize=15)

  return fig, axes

def setup_fft(fig_title:str, channels:list[str]):
  fig, axes = init_fig(fig_title)
  fig.supxlabel("frequency [Hz]", fontsize=15)
  fig.supylabel("Power Spectrum [$μV^{2}$]", fontsize=15)

  for col in range(2):
    for row in range(3):
      ch: str = channels[3*col+row]
      axes[col, row].set_title(ch)
      axes[col, row].axis([2.5, 20, 0, 20])

  return fig, axes

def setup_stft(fig_title:str, channels:list[str]):
  fig, axes = init_fig(fig_title)
  fig.supxlabel("Time [s]", fontsize=15)
  fig.supylabel("frequency [Hz]", fontsize=15)

  for col in range(2):
    for row in range(3):
      ch: str = channels[3*col+row]
      axes[col, row].set_title(ch)
      axes[col, row].set_xlim(0, 8.192)
      axes[col, row].set_ylim(3, 20)
  
  return fig, axes

def adjust_stft(vmin, vmax, fig, axes):
  axpos = axes[1,2].get_position()
  cbar_ax = fig.add_axes([0.87, axpos.y0 + 0.04, 0.02, axpos.height*2])
  norm = colors.Normalize(vmin,vmax)
  mappable = ScalarMappable(cmap='jet',norm=norm)
  mappable._A = []
  fig.colorbar(mappable, cax=cbar_ax)

  plt.subplots_adjust(right=0.85)
  plt.subplots_adjust(wspace=0.1)