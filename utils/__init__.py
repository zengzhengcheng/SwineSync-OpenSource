import numpy as np


def getjilabel(xdata,yuzhi1=3,yuzhi2=4):
    if(not isinstance(xdata,np.ndarray)):
        xdata=np.array(xdata)
    usedata =xdata
    indexlist = [0] * len(usedata)
    for ii in range(0, len(usedata), 50):
        jj = min(len(usedata), ii + 100)
        xxdata = usedata[ii:jj]
        maxvalueindex = np.argmax(xxdata) + ii
        if (usedata[maxvalueindex] < yuzhi1):
            continue
        maxwindowd_datalist = usedata[
                              max(0, maxvalueindex - 20):min(len(usedata), maxvalueindex + 20)]
        minvalue = np.min(maxwindowd_datalist)
        if (usedata[maxvalueindex] - minvalue > yuzhi2):
            indexlist[maxvalueindex] = 1
    return np.array(indexlist)


def outclean(data, labels,input):
    # ==========================================
    # 【修复核心】：处理维度问题
    # 如果输入是 (Batch, 1, Length) -> 变为 (Batch, Length)
    # ==========================================
    if data.ndim == 3:
        data = data.squeeze(1)  # 去掉中间的 Channel 维度
    if labels.ndim == 3:
        labels = labels.squeeze(1)

    batch, length = labels.shape
    outdata = np.zeros_like(labels)
    for b in range(batch):
        # 经过上面的 squeeze，这里 sig 必然是 (15360,)
        sig = data[b]
        lab = labels[b].copy()

        # 1. 合并逻辑
        ones_idx = np.where(lab == 1)[0]
        if len(ones_idx) > 1:
            diffs = np.diff(ones_idx)
            # 这里的阈值 50 可以根据采样率调整，15360长度下通常是合适的
            gap_indices = np.where((diffs > 1) & (diffs < 50))[0]
            for g_idx in gap_indices:
                start_fill = ones_idx[g_idx] + 1
                end_fill = ones_idx[g_idx + 1]
                lab[start_fill:end_fill] = 1

        # 2. 寻找峰值
        padded_lab = np.pad(lab, (1, 1), 'constant', constant_values=0)
        diff_lab = np.diff(padded_lab)
        starts = np.where(diff_lab == 1)[0]
        ends = np.where(diff_lab == -1)[0]

        for start, end in zip(starts, ends):
            if end > start:
                segment = sig[start:end]
                if len(segment) > 0:
                    peak_relative = np.argmax(segment ** 2)  # 取平方后找极大值
                    outdata[b, start + peak_relative] = 1

    return outdata
