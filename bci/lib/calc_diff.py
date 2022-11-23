from datetime import datetime, timedelta

def calc_timeDiff(test_num: int, exp_type: str, subject_total: int) -> tuple[list[float], list[str]]:
    filename_python = 'step_start_python.txt'
    filename_unity = 'step_start_unity.txt'
    time_diffs: list[float] = []
    faster: list[str] = []
    lib_dir = '../../../..'

    time_fmt = '%H:%M:%S:%f'

    for i in range(subject_total):
        pathName = f'{lib_dir}/result/test_{test_num}/subject_{i+1}/{exp_type}/'
        with open(pathName+filename_python) as f:
            tmp_python: list[str] = f.read().splitlines()

        with open(pathName+filename_unity) as f:
            tmp_unity: list[str] = f.read().splitlines()

        for i in range(10):
            time_python = datetime.strptime(tmp_python[i],time_fmt)
            time_unity = datetime.strptime(tmp_unity[i],time_fmt)

            if time_python < time_unity:
                faster.append('python')
            else:
                faster.append('unity')

            time_diff: timedelta = abs(time_python - time_unity)
            time_diffs.append(time_diff.microseconds * 0.000001)

    return time_diffs, faster
