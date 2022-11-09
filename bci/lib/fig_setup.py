import matplotlib.pyplot as plt

def setup_raw(fig_title:str):
  fig = plt.figure()
  fig.suptitle(fig_title, fontsize=20)
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

  ax1.set_xlabel('Time[s]', fontsize=15)
  ax2.set_xlabel('Time[s]', fontsize=15)
  ax3.set_xlabel('Time[s]', fontsize=15)
  ax4.set_xlabel('Time[s]', fontsize=15)
  ax5.set_xlabel('Time[s]', fontsize=15)
  ax6.set_xlabel('Time[s]', fontsize=15)

  ax1.set_ylabel('μV', fontsize=15)
  ax2.set_ylabel('μV', fontsize=15)
  ax3.set_ylabel('μV', fontsize=15)
  ax4.set_ylabel('μV', fontsize=15)
  ax5.set_ylabel('μV', fontsize=15)
  ax6.set_ylabel('μV', fontsize=15)

  return fig, ax1, ax2, ax3, ax4, ax5, ax6