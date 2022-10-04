import numpy as np
import random
import pathlib
from datetime import datetime, timedelta, timezone
wait_second_list: list[int] = [5,6,7,8,9] * 4
wait_second_list_save = []

# for i in range(20):
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

# wait_second_list_template = np.random.randint(5,10, (20))

# print(wait_second_list_template)

dt_now_jst_aware = datetime.now(
    timezone(timedelta(hours=9))
)
date_exp = dt_now_jst_aware.date()

subject_num = 1
dir_name = f'../result/{date_exp}/subject_{subject_num}'
pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)