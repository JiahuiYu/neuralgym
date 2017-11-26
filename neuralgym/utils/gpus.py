""" gpu related utils """
import os
import re
import logging


logger = logging.getLogger()


def set_gpus(gpus):
    """Set environment variable CUDA_VISIBLE_DEVICES to a list of gpus.

    :param gpus: gpu id or a list of gpu ids
    :returns: None

    """
    if not isinstance(gpus, list):
        gpus = [gpus]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in gpus)
    logger.info('Set env: CUDA_VISIBLE_DEVICES={}.'.format(gpus))


def get_gpus(num=1, dedicated=True, verbose=True):
    """Auto-select gpus for running by setting CUDA_VISIBLE_DEVICES.

    :param num: gpu numbers to get
    :param dedicated: dedicated gpu or not, i.e. one process for one gpu
    :param verbose: display nvidia-smi info if true
    :returns: a list of selected gpu

    """
    ret = os.popen('nvidia-smi pmon -c 1').readlines()
    if ret == []:
        # error, get no gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info('Error reading GPU information, set no GPU.')
        return None
    """
    # gpu     pid  type    sm   mem   enc   dec   command
    # Idx       #   C/G     %     %     %     %   name
        0    7965     C    76    37     0     0   python
        1       -     -     -     -     -     -   -
        2       -     -     -     -     -     -   -
        3       -     -     -     -     -     -   -
    """
    if verbose:
        print(''.join(ret))
    gpus = {}  # gpu id: number of processes
    for line in ret[2:]:
        s = re.split(r'\s+', line)[1:-1]
        gpu_id = int(s[0])
        pid = s[1]
        if pid == '-':
            gpus[gpu_id] = 0
        else:
            if gpu_id in gpus:
                gpus[gpu_id] += 1
            else:
                gpus[gpu_id] = 1
    sorted_gpus = sorted(gpus.items(), key=lambda x: x[1])
    if len(sorted_gpus) < num:
        raise SystemError(
            'No enough gpus. {} v.s. {}.'.format(len(sorted_gpus), num))
    if dedicated and sorted_gpus[num-1][1] != 0:
        raise SystemError(
            'No enough gpus for dedicated usage.'
            ' [(gpu id: num of processes)]: {}'.format(sorted_gpus))
    selected_gpu_ids = [g[0] for g in sorted_gpus][:num]
    set_gpus(selected_gpu_ids)
    return selected_gpu_ids
