from nvitop import Device


def read_scp(file, sep=' '):
    content = dict()
    with open(file) as f:
        all = f.readlines()
        for line in all:
            line_split = line.strip().split(sep)
            wav_id = line_split[0]            
            content[wav_id] = ' '.join(line_split[1:])
    return content


def find_available_gpu(num=1, mem=8, interval=-1):
    # mem: xx G, -1 找没有计算任务的卡
    mem_G = 1024**3
    devices = Device.all()

    def _get_process_num(device, p_type='C'):
        cnt = 0
        processes = device.processes()
        for pid in processes:
            if processes[pid].type == p_type:
                cnt += 1
        return cnt

    def _get_valiable_device():
        device_dict = {}
        for device in devices:
            if mem == -1: # 找没有计算任务的卡
                if _get_process_num(device) == 0:
                    device_dict[str(device.index)] = mem
            else: # 找可用显存大于mem的卡
                mem_free = device.memory_free()/mem_G
                if mem_free > mem:
                    device_dict[str(device.index)] = mem_free
        return device_dict

    device_dict = _get_valiable_device()
    if interval > 1:
        while True:
            if len(device_dict) >= num:
                break
            rprint(f'Querying GPU per {interval} s ...')
            time.sleep(interval)
            device_dict = _get_valiable_device()
    sorted_keys = sorted(device_dict, key=device_dict.get, reverse=True)
    return ','.join(sorted_keys[:num]) if len(sorted_keys) >= num else ''