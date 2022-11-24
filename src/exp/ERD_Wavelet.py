
from signal import signal
from matplotlib.cbook import file_requires_unicode
from brainflow.data_filter import DataFilter,\
    NoiseTypes
from brainflow.board_shim import BoardShim, \
    BrainFlowInputParams, \
    BoardIds
import argparse
import logging
import socket
import time
import pywt
import os
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from threading import Thread

HOST = '127.0.0.1'
PORT = 50007

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cyton_board_ID = 0

def record_raw(data, ch, trial_amount, session):
    i = 0
    if (session == 'task'):
        while os.path.exists(f'data_raw/task/Wavelet/rawdata_task_ch{ch}_task{trial_amount+1}_{i}.csv'):
            i += 1
        filename = f'data_raw/task/Wavelet/rawdata_task_ch{ch}_task{trial_amount+1}_{i}.csv'
        pd.DataFrame(data).to_csv(filename)
    elif (session == 'rest'):
        while os.path.exists(f'data_raw/rest/Wavelet/rawdata_rest_ch{ch}_{i}.csv'):
            i += 1
        filename = f'data_raw/rest/Wavelet/rawdata_rest_ch{ch}_{i}.csv'
        pd.DataFrame(data).to_csv(filename)


def record_to_csv_task(TO_SAVE_LIST, ch, trial_amount):
    i = 0
    while os.path.exists(f'data/task/Wavelet/data_ch{ch}_task{trial_amount+1}_{i}.csv'):
        i += 1
    filename = f'data/task/Wavelet/task_data_ch{ch}_{i}.csv'
    pd.DataFrame(TO_SAVE_LIST).to_csv(filename)


def record_to_csv_rest(TO_SAVE_LIST, ch):
    i = 0
    while os.path.exists(f'data/rest/Wavelet/rest_data_ch{ch}_{i}.csv'):
        i += 1
    filename = f'data/rest/Wavelet/rest_data_ch{ch}_{i}.csv'
    pd.DataFrame(TO_SAVE_LIST).to_csv(filename)

def record_result(result_task):
    i = 0
    while os.path.exists(f'result/result_wavelet_{i}.csv'):
        i += 1
    filename = f'result/result_wavelet_{i}.csv'
    pd.DataFrame(result_task).to_csv(filename)

def CWT(data, fs):
    wavelet_type = 'morl'
    nq_f = fs / 2.0 # ナイキスト周波数
    freqs = np.linspace(8, 13, 10) # 解析したい周波数リスト
    dt = 1 / fs
    freqs_rate = freqs / fs
    scales = 1 / freqs_rate
    scales = scales[::-1] # 逆順にする
    frequencies_rate = pywt.scale2frequency(scale=scales, wavelet=wavelet_type)
    frequencies = frequencies_rate / dt

    cwtmatr, freq = pywt.cwt(data, scales=scales, wavelet=wavelet_type)
    return np.abs(cwtmatr) * np.abc(cwtmatr)

def calc_band_average(cwt):
    return np.mean(cwt)

## CH1 -> C3 CH2 -> C4

def rest_average(REST_DATA, board_shim):
    rest_data = REST_DATA
    board_shim = board_shim

    fs = BoardShim.get_sampling_rate(cyton_board_ID)
    eeg_channels = BoardShim.get_eeg_channels(cyton_board_ID)
    max_f = 13
    min_f = 8
    
    # Sampling Dots
    dt = 1/fs
    
    # TODO 絶対値の二乗を計算する->DONE
    for count, channel in enumerate(eeg_channels[0:2]):
        DataFilter.remove_environmental_noise(
            rest_data[channel], fs, NoiseTypes.FIFTY.value)
        N = rest_data.shape[0]
        #TODO fftのリストに電極位置の名前つける
        if count == 0:
            # sfft結果に対して絶対値の二乗を取って、パワースペクトルを得る。
            cwt = CWT(rest_data[channel], fs)
            record_raw(rest_data[channel], count,
                       trial_amount=0, session='rest')
            record_to_csv_rest(cwt, count)
            ch_1_power_rest = calc_band_average(cwt)
        elif count == 1:
            cwt = CWT(rest_data[channel], fs)
            record_raw(rest_data[channel], count,
                       trial_amount=0, session='rest')
            record_to_csv_rest(cwt, count)
            ch_2_power_rest = calc_band_average(cwt)
    else:
        return [ch_1_power_rest, ch_2_power_rest]

def eval_ERD(rest_power, task_power):
    ERD_flag = False
    ERD = (np.mean(task_power) - np.mean(rest_power)) / np.mean(rest_power)
    if ERD < 0:
        ERD_flag = True
    return ERD_flag

class calcBand:
    def __init__(self, board_shim, rest_list):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.board_dscr = board_shim.get_board_descr(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id) #250Hz
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.rest_power = rest_list

        self.bandpower()

    def bandpower(self):
        input(">>>>>>> PRESS Enter to start analysis")
        # バンドパスフィルタの幅を示すパラメータ
        fs = self.sampling_rate
        result_task = []
        for trial_amount in range(5):
            time.sleep(5)
            print(f'Trial {trial_amount+1}')
            
            first_detect = False
            flag = False
            t_end = time.time() + 5
            while time.time() < t_end:
                data = self.board_shim.get_current_board_data(self.num_points)  # current dataを取得
                for count, channel in enumerate(self.eeg_channels[0:2]):
                    DataFilter.remove_environmental_noise(data[channel], fs, NoiseTypes.FIFTY.value)
                    if count == 0:
                        # sfft結果に対して絶対値の二乗を取って、パワースペクトルを得る。
                        cwt_ch1 = CWT(data[channel], fs)
                        ch_1_power_task = calc_band_average(cwt_ch1)
                    elif count == 1:
                        cwt_ch2 = CWT(data[channel], fs)
                        ch_2_power_task = calc_band_average(cwt_ch2)
                task_power = [ch_1_power_task,ch_2_power_task]
                flag = eval_ERD(self.rest_power, task_power)
                if flag and not first_detect:
                    first_detect = True
                    result = '1'
                    client.sendto(result.encode('utf-8'), (HOST, PORT))
                    print("ERD Detected !!!!")
                time.sleep(1/self.sampling_rate)  # 0.004秒ずつfor文を回す
            else:
                if flag:
                    result_task.append('Success')
                else:
                    result_task.append('Failure')
                for i in range(2):
                    if i == 0:
                        record_raw(data[self.eeg_channels[0]],
                                   self.eeg_channels[0],
                                   trial_amount,
                                   'task')
                        record_to_csv_task(cwt_ch1, i, trial_amount)
                    elif i == 1:
                        record_raw(data[self.eeg_channels[1]],
                                   self.eeg_channels[1],
                                   trial_amount,
                                   'task')
                        record_to_csv_task(cwt_ch2, i, trial_amount)
        else:
            record_result(result_task)

def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    '''
    use docs to check which parameters are required for specific board, 
    e.g. for Cyton - set serial port
    '''
    parser.add_argument('--timeout',
                        type=int,
                        help='timeout for device discovery or connection',
                        required=False,
                        default=0)
    parser.add_argument('--ip-port',
                        type=int,
                        help='ip port',
                        required=False,
                        default=0)
    parser.add_argument('--ip-protocol',
                        type=int,
                        help='ip protocol, check IpProtocolType enum',
                        required=False,
                        default=0)
    parser.add_argument('--ip-address',
                        type=str,
                        help='ip address',
                        required=False,
                        default='')
    parser.add_argument('--serial-port',
                        type=str,
                        help='serial port',
                        required=False,
                        default='')
    parser.add_argument('--mac-address',
                        type=str,
                        help='mac address',
                        required=False, default='')
    parser.add_argument('--other-info',
                        type=str,
                        help='other info',
                        required=False,
                        default='')
    parser.add_argument('--streamer-params',
                        type=str,
                        help='streamer params',
                        required=False,
                        default='')
    parser.add_argument('--serial-number',
                        type=str,
                        help='serial number',
                        required=False, default='')
    parser.add_argument('--board-id',
                        type=int,
                        help='board id, check docs to get \
                              a list of supported boards',
                        required=False,
                        default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--file',
                        type=str,
                        help='file',
                        required=False,
                        default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    try:
        board_shim = BoardShim(args.board_id, params)
        board_shim.prepare_session()
        input("Press ENTER to start REST session")
        board_shim.start_stream(450000, args.streamer_params)
        print(">>>>>> Calculation Average Power of rest session(5 second)")
        time.sleep(5)
        rest_data = board_shim.get_board_data()
        board_shim.stop_stream()
        rest_list = rest_average(rest_data, board_shim)
        print(rest_list)
        time.sleep(2)
        input("Press ENTER to start TASK session")
        print("-----Starting Task Session-----")
        board_shim.start_stream(450000, args.streamer_params)
        calcband = calcBand(board_shim, rest_list)
        # calcband.bandpower()
    except KeyboardInterrupt:
        logging.warning('Keyboard Interrupt', exc_info=True)
    finally:
        logging.info('End')
        print("-----データ保存-----")
        #record_to_csv(calcband.to_save_list)
        time.sleep(2)
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()
