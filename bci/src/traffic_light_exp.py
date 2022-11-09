import time
import pathlib
import argparse
import keyboard
from ctypes import windll
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
from datetime import datetime, timedelta, timezone

# 練習時の待ち時間(赤→青)
WAIT_SECOND_PRACTICE: list[int] = [5] * 10
# 本番時の待ち時間(赤→青)のランダム生成配列
WAIT_SECOND_ACTUAL: list[int] = [8, 6, 6, 7, 5,
                                 7, 7, 5, 5, 7, 9, 9, 9, 8, 6, 5, 6, 8, 9, 8]

"""
Command Args:

BCI_Project/bci/src/ $ python traffic_light_exp.py --board-id -1

Channels:
1: Fp1
2: Fp2
3: Cz
4: C3
5: C4
6: Fz
7: F3
8: F4

"""


class SubjectData:
    def __init__(self, subject_num: int, exp_type: int, test_measurement_flag: bool):
        self.subject_num = subject_num
        self.dt_now_jst = datetime.now(timezone(timedelta(hours=9)))
        self.date_exp = self.dt_now_jst.date()
        self.exp_type = exp_type
        self.test_measurement_flag = test_measurement_flag

    def write_data(self, to_save_data, num_step: int):

        # ディレクトリがなかったら作る
        if (self.test_measurement_flag):
            result_dir = f'../result/test/{self.date_exp}/subject_{self.subject_num}/{self.exp_type}'
            pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
        else:
            result_dir = f'../result/{self.date_exp}/subject_{self.subject_num}/{self.exp_type}'
            pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

        # ステップごとのファイル名
        file_name = f'{result_dir}/subject_{self.subject_num}_step_{num_step+1}.csv'

        # データ保存
        DataFilter.write_file(to_save_data, file_name, 'w')


# TODO:Python側のデータ記録開始時刻と、Unity側のデータ記録開始時刻を記録するようにする
class ControlExp:
    def __init__(self, board: BoardShim, subject_num: int, test_measurement_flag: bool,debug_flag: bool):
        self.board = board
        self.subject_num = subject_num
        self.test_measurement_flag = test_measurement_flag
        self.debug_flag = debug_flag


    def control_exp(self, exp_type: str):

        save_data = SubjectData(
            subject_num=self.subject_num, exp_type=exp_type, test_measurement_flag=self.test_measurement_flag)

        total_duration: float = 0

        if (exp_type == 'practice'):
            wait_second_list: list = WAIT_SECOND_PRACTICE
        elif (exp_type == 'actual'):
            wait_second_list: list = WAIT_SECOND_ACTUAL

        for i in range(len(wait_second_list)):
            print("---------------")
            # 計測時間
            wait_second = wait_second_list[i] + 11

            # DataStream開始
            self.board.start_stream()

            # 計測開始時刻を出力
            time_start = datetime.now()
            print("Start Time:", time_start.strftime("%H:%M:%S:%f"))

            # 計測時間分Sleepする
            time.sleep(wait_second)

            # 計測時間終了後にデータをボードから取ってくる
            data = self.board.get_board_data()
            print(data)

            # 計測終了時刻
            time_end = datetime.now()

            # Streamの停止
            # stop_streamのあとにget_board_data()できる？
            self.board.stop_stream()

            if self.debug_flag:
                # データをCSVに書き込む
                save_data.write_data(to_save_data=data, num_step=i,)

            print(f'Step {i+1} Ended')
            # 計測終了時刻を出力
            print("Finished Time:", time_end.strftime("%H:%M:%S:%f"))
            # 計測開始から終了までのdurationを計算して出力
            step_duration = time_end - time_start
            print("Step Duration:", step_duration)

            total_duration += step_duration.total_seconds()

        print("++++++++++++++++++++++++++")
        # 全ステップの合計duration
        print("Total Duration:", total_duration)
        time_finished = datetime.now()
        # 計測終了時刻を出力
        print("All Steps Finished Time:", time_finished.strftime("%H:%M:%S:%f"))


def main():

    windll.winmm.timeBeginPeriod(1)

    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int,
                        help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str,
                        help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str,
                        help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str,
                        help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str,
                        help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str,
                        help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file',
                        required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=0)
    parser.add_argument('--preset', type=int, help='preset for streaming and playback boards',
                        required=False, default=0)
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
    params.master_board = args.master_board
    params.preset = args.preset

    board = BoardShim(args.board_id, params)
    board.prepare_session()
    
    subject_num: int = input("Enter Subjuct Number >>> ")
    test_measurement_flag: bool = bool(int(input(("Test Measurement? [1(Yes), 0(No)] >>> "))))
    debug_flag: bool = bool(int(input("If you wanna save a data? [1(Yes), 0(No)] >>> ")))
    exp_control = ControlExp(board=board, subject_num=subject_num, test_measurement_flag=test_measurement_flag, debug_flag=debug_flag)

    try:
        print(">>>>> Enter s key to start Practice measurement <<<<< ")
        while True:
            if keyboard.is_pressed("s"):
                print("wait 3 seconds ...")
                time.sleep(2.5)
                exp_control.control_exp(exp_type='practice')
                break

        print(">>>>> Enter s key to start Actual measurement <<<<< ")
        while True:
            if keyboard.is_pressed("s"):
                print("wait 3 seconds ...")
                time.sleep(2.5)
                exp_control.control_exp(exp_type='actual')
                break
    except KeyboardInterrupt:
        windll.winmm.timeEndPeriod(1)
    finally:
        windll.winmm.timeEndPeriod(1)

    board.release_session()


if __name__ == "__main__":
    main()
