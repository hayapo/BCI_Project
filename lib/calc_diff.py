from datetime import datetime, timedelta
import numpy as np
def timeDiff(test_num: int, exp_type: str, subject_total: int, result_dir: str) -> tuple[list[float], list[str]]:
    filename_python = 'step_start_python.txt'
    filename_unity = 'step_start_unity.txt'
    # (3,10)のほうが良いかも
    time_diffs: list[float] = []
    faster: list[str] = []

    time_fmt = '%H:%M:%S:%f'

    for i in range(subject_total):
        pathName = f'{result_dir}/test_{test_num}/subject_{i+1}/{exp_type}/'
        with open(pathName+filename_python) as f:
            tmp_python: list[str] = f.read().splitlines()

        with open(pathName+filename_unity) as f:
            tmp_unity: list[str] = f.read().splitlines()

        tmp_diffs = []
        tmp_faster = []
        for i in range(10):
            time_python = datetime.strptime(tmp_python[i], time_fmt)
            time_unity = datetime.strptime(tmp_unity[i], time_fmt)
            
            time_diff: timedelta = abs(time_python - time_unity)

            # 開始時刻差を計算(pythonより早いか遅いか)
            if time_python < time_unity:
                tmp_faster.append('python')
                tmp_diffs.append(time_diff.microseconds * 0.000001)
            elif time_unity < time_python:
                tmp_faster.append('unity')
                tmp_diffs.append(time_diff.microseconds * 0.000001 * -1)
        
        faster.append(tmp_faster)
        time_diffs.append(tmp_diffs)

    return time_diffs, faster