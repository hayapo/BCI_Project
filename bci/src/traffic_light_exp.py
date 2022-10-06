import argparse
import pathlib
from datetime import datetime, timedelta, timezone
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
import keyboard

# 練習時の待ち時間(赤→青)
WAIT_SECOND_PRACTICE: list[int] = [5] * 10
# 実験ステップ数_練習
EXP_PRACTICE_STEPS: int = 10

# 本番時の待ち時間(赤→青)のランダム生成配列
WAIT_SECOND_ACTUAL: list[int] = [8, 6, 6, 7, 5, 7, 7, 5, 5, 7, 9, 9, 9, 8, 6, 5, 6, 8, 9, 8]
# 実験ステップ数_本番
EXP_ACTUAL_STEPS: int = len(WAIT_SECOND_ACTUAL)

"""
Command Args:

BCI_Project/bci/src/ $ python traffic_light_exp.py --board-id -1

"""

class SubjectData:
  def __init__(self, subject_num: int, exp_type: int):
    self.subject_num = subject_num
    self.dt_now_jst = datetime.now(timezone(timedelta(hours=9)))
    self.date_exp = self.dt_now_jst.date()
    self.exp_type = exp_type

  def write_data(self, to_save_data, num_step: int):

    #ディレクトリがなかったら作る
    result_dir = f'../result/{self.date_exp}/subject_{self.subject_num}/{self.exp_type}'
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

    #ステップごとのファイル名
    file_name = f'{result_dir}/subject_{self.subject_num}_step_{num_step+1}.csv'

    #データ保存
    DataFilter.write_file(to_save_data, file_name, 'w') 

  # def save_data_practice(self,to_save_data, num_step):
  #   dt_now_jst = datetime.now(timezone(timedelta(hours=9)))
  #   date_exp = dt_now_jst.date()

  #   #ディレクトリがなかったら作る
  #   dir_name = f'../result/{date_exp}/subject_{self.subject_num}/practice'
  #   pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

  #   #ステップごとのファイル名
  #   file_name = f'{dir_name}/subject_{self.subject_num}_step_{num_step+1}.csv'

  #   #データ保存
  #   DataFilter.write_file(to_save_data, file_name, 'w') 

  # def save_data_actual(self,to_save_data, num_step):
  #   dt_now_jst = datetime.now(timezone(timedelta(hours=9)))
  #   date_exp = dt_now_jst.date()

  #   #ディレクトリがなかったら作る
  #   dir_name = f'../result/{date_exp}/subject_{self.subject_num}/actual'
  #   pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

  #   #ステップごとのファイル名
  #   file_name = f'{dir_name}/subject_{self.subject_num}_step_{num_step+1}.csv'

  #   #データ保存
  #   DataFilter.write_file(to_save_data, file_name, 'w') 

class ControlExp:
  def __init__(self, board: BoardShim, subject_num: int ):
    self.subject_num = subject_num
    self.board = board
    
  def control_exp(self, exp_type: str):

    save_data = SubjectData(subject_num=self.subject_num, exp_type=exp_type)

    if (exp_type == 'practice'):
      wait_second_list: list = WAIT_SECOND_PRACTICE
    elif (exp_type == 'actual'):
      wait_second_list: list = WAIT_SECOND_ACTUAL

    for i in range(len(wait_second_list)):
      # 計測時間
      wait_second = wait_second_list[i] + 11

      # DataStream開始
      self.board.start_stream()

      # 計測時間分Sleepする
      time.sleep(wait_second)

      # 計測時間終了後にデータをボードから取ってくる
      data = self.board.get_board_data()
      print(data)

      # Streamの停止
      # stop_streamのあとにget_board_data()できる？
      self.board.stop_stream()

      # データをCSVに書き込む
      save_data.write_data(to_save_data=data, num_step=i)

# def control_practice(board: BoardShim, subject_num: int):
#   subjectData = SubjectData(subject_num)

#   for i in range(10):
#     wait_second = WAIT_SECOND_PRACTICE + 11
#     board.start_stream()
#     time.sleep(wait_second)
#     data = board.get_board_data()
#     print(data) # get all data and remove it from internal buffer
#     board.stop_stream()
#     subjectData.save_data_practice(data, i)

# def control_actual(board: BoardShim, subject_num: int):
#   subjectData = SubjectData(subject_num)

#   for i in range(EXP_ACTUAL_STEPS):
#     wait_second = WAIT_SECOND_ACTUAL[i] + 11
#     board.start_stream()
#     time.sleep(wait_second)
#     data = board.get_board_data()
#     print(data) # get all data and remove it from internal buffer
#     board.stop_stream()
#     subjectData.save_data_actual(data, i)

def main():
  BoardShim.enable_dev_board_logger()

  parser = argparse.ArgumentParser()
  # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
  parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                      default=0)
  parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
  parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                      default=0)
  parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
  parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
  parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
  parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
  parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
  parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                      required=True)
  parser.add_argument('--file', type=str, help='file', required=False, default='')
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
  exp_control = ControlExp(board=board, subject_num=subject_num)

  print(">>>>> Enter s key to start Practice measurement <<<<< ")
  while True:
    if keyboard.is_pressed("s"):
      time.sleep(3)
      exp_control.control_exp(exp_type='practice')
      break

  print(">>>>> Enter s key to start Actual measurement <<<<< ")
  while True:
    if keyboard.is_pressed("s"):
      time.sleep(3)
      exp_control.control_exp(exp_type='actual')
      break

  board.release_session()

if __name__ == "__main__":
    main()