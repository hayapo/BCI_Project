import time
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def main():
    BoardShim.enable_dev_board_logger()
    # use synthetic board for demo
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    for i in range(10):
        time.sleep(1)
        board.insert_marker(i + 1)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    df = pd.DataFrame(np.transpose(data))
    print('Data From the Board')

    def find_index1(item):
      return df[31].where(df[31] == item).first_valid_index()
    print(df[31][250:270])


if __name__ == "__main__":
    main()