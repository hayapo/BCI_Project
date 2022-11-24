import numpy as np
import random
import pathlib
import time
from datetime import datetime, timedelta, timezone
from lib import power_func

# with open('d.txt', 'w') as f:
#     for step,data in enumerate(l):
#         f.write(f'Step {step+1}: {data}\n')

# wait_second_list: list[int] = [5,6,7,8,9] * 2
# wait_second_list_save = []

# for i in range(10):
#   wait_second: int = random.choice(wait_second_list)
#   wait_second_list_save.append(wait_second)
#   print(wait_second)
#   wait_second_list.remove(wait_second)
#   print(f'list:{wait_second_list}')
#   print(f'list:{wait_second_list_save}')

# five_count = 0
# six_count = 0
# seven_count = 0
# eight_count = 0
# nine_count = 0

# for i in wait_second_list_save:
#   if i == 5:
#     five_count += 1
#   if i == 6:
#     six_count += 1
#   if i == 7:
#     seven_count += 1
#   if i == 8:
#     eight_count += 1
#   if i == 9:
#     nine_count += 1

# print(f'5: {five_count}')
# print(f'6: {six_count}')
# print(f'7: {seven_count}')
# print(f'8: {eight_count}')
# print(f'9: {nine_count}')

# wait_second_list_template = np.random.randint(5, 10, (10))

# print(wait_second_list_template)

# dt_now_jst_aware = datetime.now(
#     timezone(timedelta(hours=9))
# )
# date_exp = dt_now_jst_aware.date()

# subject_num = 1
# dir_name = f'../result/{date_exp}/subject_{subject_num}'
# pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
# WAIT_SECOND_PRACTICE: list[int] = [5] * 10
# WAIT_SECOND_ACTUAL: list[int] = [8, 6, 6, 7, 5, 7, 7, 5, 5, 7, 9, 9, 9, 8, 6, 5, 6, 8, 9, 8]
# wait_second = list(map(lambda x: x + 11, WAIT_SECOND_ACTUAL))
# print(wait_second)

# for i in wait_second:
#     print("----------------")
#     time_start = datetime.now()
#     print("start time:", time_start.strftime("%H:%M:%S:%f"))
    
#     time.sleep(i)

#     time_end = datetime.now()
#     print("end time:", time_end.strftime("%H:%M:%S:%f"))

#     print("duration:", time_end - time_start )
