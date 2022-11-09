import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from lib import filter_func
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

board_id = BoardIds.CYTON_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)
FS: int = 250

# データ読み込み
subject_num: int = 3
exp_type: str = 'practice'
test_flag: bool = True
test_num: int = 1
