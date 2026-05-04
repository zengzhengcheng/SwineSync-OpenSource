import numpy as np

class RRIntervalCorrector:
    """
    RR间隔校正器，用于处理30秒心电数据
    """
    def __init__(self, sampling_rate=512):
        """
        初始化校正器

        Args:
            sampling_rate: 采样率，默认512Hz
        """
        self.sampling_rate = sampling_rate
        self.window_len = 30 * sampling_rate  # 30秒窗口长度

    def define_statistical_intervals(self):
        """
        定义基于统计分析的心率区间
        """
        return [
            {
                'interval_id': 1,
                'name': 'Very Fast (HR > 160 BPM)',
                'rr_range': (0, 375),
                'lower_bound_factor': 0.55,  # updated from excellent data IQR analysis (was 0.35)
                'upper_bound_factor': 1.17,  # updated from excellent data IQR analysis (was 1.24)
            },
            {
                'interval_id': 2,
                'name': 'Fast (HR 120-160 BPM)',
                'rr_range': (375, 500),
                'lower_bound_factor': 0.84,
                'upper_bound_factor': 1.11
            },
            {
                'interval_id': 3,
                'name': 'Normal (HR 75-120 BPM)',
                'rr_range': (500, 800),
                'lower_bound_factor': 0.79,
                'upper_bound_factor': 1.26
            },
            {
                'interval_id': 4,
                'name': 'Slow (HR 60-75 BPM)',
                'rr_range': (800, 1000),
                'lower_bound_factor': 0.93,
                'upper_bound_factor': 1.15
            },
            {
                'interval_id': 5,
                'name': 'Very Slow (HR < 60 BPM)',
                'rr_range': (1000, float('inf')),
                'lower_bound_factor': 0.87,
                'upper_bound_factor': 1.31
            }
        ]

    def find_matching_interval(self, mean_rr):
        """
        根据平均RR间隔找到匹配的心率区间
        """
        intervals = self.define_statistical_intervals()
        for interval in intervals:
            min_rr, max_rr = interval['rr_range']
            if min_rr <= mean_rr < max_rr:
                return interval
        return intervals[2]  # 默认返回正常区间

    def calculate_reference_interval(self, rr_intervals_ms):
        """
        两步迭代过滤，计算鲁棒参考RR间隔。

        第一步：过滤掉生理上不可能的极端值（< 200ms 对应 > 300BPM）
        第二步：用第一步的中位数再次过滤明显偏离的值（< 0.5x 或 > 2x），
                消除初始标注中批量假阳性/假阴性对参考值的污染。
        第三步：对最终有效集合用 Q40/中位数/Q60 决定是用均值还是中位数。
        """
        # --- 第一步：生理极限过滤 ---
        step1 = rr_intervals_ms[(rr_intervals_ms >= 200) & (rr_intervals_ms <= 3000)]
        if len(step1) < 2:
            step1 = rr_intervals_ms  # 无法过滤则回退

        if len(step1) < 3:
            return float(np.mean(step1)) if len(step1) > 0 else 600.0

        # --- 第二步：以初步中位数为中心，过滤明显离群的间隔 ---
        provisional_ref = float(np.median(step1))
        step2 = step1[(step1 >= provisional_ref * 0.5) & (step1 <= provisional_ref * 2.0)]
        if len(step2) < 2:
            step2 = step1  # 过滤后数据太少则回退

        # --- 第三步：稳健估计 ---
        q40 = np.percentile(step2, 40)
        median = np.median(step2)
        q60 = np.percentile(step2, 60)

        if max(q40, median, q60) - min(q40, median, q60) < 100:
            return float(np.mean(step2))
        else:
            return float(median)

    def adjust_r_wave_position(self, ecg, r_idx, window_size=20):
        """
        调整R波位置到局部绝对值最大点（兼容正负极性）。
        """
        window_start = max(0, r_idx - window_size)
        window_end = min(len(ecg), r_idx + window_size + 1)
        window_data = ecg[window_start:window_end]
        local_max_idx = np.argmax(window_data ** 2)
        return window_start + local_max_idx

    def handle_short_rr_interval(self, ecg, label, r1, r2,
                                   amp_ratio_threshold=0.6):
        """
        处理短RR间隔：保留绝对幅度最大的R波，移除其余。

        amp_ratio_threshold: 若次高峰幅度 / 最高峰幅度 >= 此值，
            认为两峰幅度相近、可能都是真实R波（自然HRV引起间隔偏短），
            则不删除任何峰，避免误删真实R波。
        """
        corrected_label = label.copy()

        between_r_indices = np.where(corrected_label[r1:r2 + 1] == 1)[0] + r1

        if len(between_r_indices) >= 2:
            amplitudes = []
            for r_idx in between_r_indices:
                window_start = max(0, r_idx - 5)
                window_end = min(len(ecg), r_idx + 6)
                window_data = ecg[window_start:window_end]
                max_amp = float(np.max(np.abs(window_data)))
                amplitudes.append(max_amp)

            amps = np.array(amplitudes)
            top_index = int(np.argmax(amps))
            max_amp = amps[top_index]

            # 若次高峰与最高峰幅度相近，说明很可能都是真实R波，不删
            other_amps = [a for i, a in enumerate(amps) if i != top_index]
            if len(other_amps) > 0 and max(other_amps) / (max_amp + 1e-9) >= amp_ratio_threshold:
                return corrected_label  # 保留全部，不处理

            keep_index = between_r_indices[top_index]
            for r_idx in between_r_indices:
                if r_idx != keep_index:
                    corrected_label[r_idx] = 0

        return corrected_label

    def _is_significant_peak(self, ecg, pos, reference_amplitude, min_ratio=0.4):
        """
        检查插入位置是否真的有显著的R波峰值。
        reference_amplitude: 本窗口内已知R波的中位幅度（绝对值）。
        min_ratio: 插入点幅度 / 参考幅度 的最小比例，低于此则认为该位置没有真实R波。
        """
        w_start = max(0, pos - 10)
        w_end = min(len(ecg), pos + 11)
        local_amp = float(np.max(np.abs(ecg[w_start:w_end])))
        return local_amp >= reference_amplitude * min_ratio

    def handle_long_rr_interval(self, ecg, label, r1, r2, reference_rr, reference_amplitude):
        """
        处理长RR间隔，递归检查并插入多个点直到间隔合理。
        reference_amplitude: 本窗口已知R波的幅度参考，用于验证插入点是否真实存在R波。
        """
        corrected_label = label.copy()

        current_rr = (r2 - r1) / self.sampling_rate * 1000

        if current_rr <= reference_rr * 1.3:
            return corrected_label

        num_peaks_to_insert = int(current_rr / reference_rr) - 1

        if num_peaks_to_insert > 0:
            interval = (r2 - r1) / (num_peaks_to_insert + 1)
            inserted_peaks = []

            for i in range(1, num_peaks_to_insert + 1):
                peak_pos = int(r1 + i * interval)
                min_gap_samples = int(self.sampling_rate * 0.15)
                if (peak_pos - r1) > min_gap_samples and (r2 - peak_pos) > min_gap_samples:
                    window_size = 20
                    window_start = max(0, peak_pos - window_size)
                    window_end = min(len(ecg), peak_pos + window_size + 1)
                    window_data = ecg[window_start:window_end]
                    if len(window_data) > 0:
                        local_max_idx = np.argmax(window_data ** 2)
                        actual_peak_pos = window_start + local_max_idx
                        # 验证该位置确实有显著的R波幅度，避免在平坦段插入假点
                        if self._is_significant_peak(ecg, actual_peak_pos, reference_amplitude):
                            inserted_peaks.append(actual_peak_pos)

            for peak in inserted_peaks:
                corrected_label[peak] = 1

            # 没有成功插入任何点（全部被振幅验证拒绝），停止递归，避免死循环
            if len(inserted_peaks) == 0:
                return corrected_label

            # 重新获取R波位置并递归检查子区间
            all_peaks = np.where(corrected_label == 1)[0]
            updated_heartindex = np.unique(np.concatenate([all_peaks, [r1, r2]]))

            r1_pos = np.where(updated_heartindex == r1)[0]
            r2_pos = np.where(updated_heartindex == r2)[0]
            if len(r1_pos) == 0 or len(r2_pos) == 0:
                return corrected_label
            r1_idx = r1_pos[0]
            r2_idx = r2_pos[0]

            for i in range(r1_idx, r2_idx):
                current_r = updated_heartindex[i]
                next_r = updated_heartindex[i + 1]
                corrected_label = self.handle_long_rr_interval(
                    ecg, corrected_label, current_r, next_r, reference_rr, reference_amplitude
                )

        elif current_rr > reference_rr * 1.5:
            mid_point = (r1 + r2) // 2
            min_gap_samples = int(self.sampling_rate * 0.15)
            if (mid_point - r1) > min_gap_samples and (r2 - mid_point) > min_gap_samples:
                window_size = 20
                window_start = max(0, mid_point - window_size)
                window_end = min(len(ecg), mid_point + window_size + 1)
                window_data = ecg[window_start:window_end]
                if len(window_data) > 0:
                    local_max_idx = np.argmax(window_data ** 2)
                    actual_peak_pos = window_start + local_max_idx
                    if self._is_significant_peak(ecg, actual_peak_pos, reference_amplitude):
                        corrected_label[actual_peak_pos] = 1
                        corrected_label = self.handle_long_rr_interval(
                            ecg, corrected_label, r1, actual_peak_pos, reference_rr, reference_amplitude
                        )
                        corrected_label = self.handle_long_rr_interval(
                            ecg, corrected_label, actual_peak_pos, r2, reference_rr, reference_amplitude
                        )

        return corrected_label

    def adjust_r_wave_positions(self, ecg, label):
        """
        调整所有R波位置到局部绝对值最大点。
        若目标位置已有标记（两峰吸附到同一点），则不移动原峰，保留两者。
        """
        corrected_label = label.copy()
        r_wave_indices = np.where(label == 1)[0]

        for r_idx in r_wave_indices:
            adjusted_idx = self.adjust_r_wave_position(ecg, r_idx)
            if adjusted_idx != r_idx:
                if corrected_label[adjusted_idx] == 1:
                    # 目标位置已有标记，不移动，保留原位置
                    pass
                else:
                    corrected_label[r_idx] = 0
                    corrected_label[adjusted_idx] = 1

        return corrected_label

    def _compute_reference_amplitude(self, ecg, heartindex):
        """
        计算窗口内已知R波的中位幅度（绝对值），作为插入验证的基准。
        """
        amps = []
        for idx in heartindex:
            w_start = max(0, idx - 5)
            w_end = min(len(ecg), idx + 6)
            amps.append(float(np.max(np.abs(ecg[w_start:w_end]))))
        if len(amps) == 0:
            return 0.0
        return float(np.median(amps))

    def correct(self, ecg, labels):
        """
        校正RR间隔

        Args:
            ecg: 30秒ECG信号
            labels: 30秒R波标签

        Returns:
            tuple: (校正后的标签, 校正信息)
        """
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        if not isinstance(ecg, np.ndarray):
            ecg = np.array(ecg, dtype=float)

        window_len = self.window_len

        assert len(labels) == window_len
        assert len(ecg) == window_len

        rough_mask = labels.copy()

        # 找到R波位置
        heartindex = np.where(rough_mask == 1)[0]

        if len(heartindex) < 3:
            return labels, {"message": "too few peaks", "peaks": heartindex}

        # 计算RR间隔（毫秒）
        rr_intervals_ms = np.diff(heartindex) / self.sampling_rate * 1000

        # 第一次参考RR计算（两步迭代过滤，对初始噪声鲁棒）
        reference_rr = self.calculate_reference_interval(rr_intervals_ms)

        # 心率区间与阈值
        matching_interval = self.find_matching_interval(reference_rr)
        lower_bound = reference_rr * matching_interval['lower_bound_factor']
        upper_bound = reference_rr * matching_interval['upper_bound_factor']

        # 计算R波幅度参考（用于验证插入点是否有真实R波）
        reference_amplitude = self._compute_reference_amplitude(ecg, heartindex)

        abnormal_indices = np.where(
            (rr_intervals_ms < lower_bound) | (rr_intervals_ms > upper_bound)
        )[0]

        corrected_label = rough_mask.copy()
        correction_info = []

        if len(abnormal_indices) > 0:
            # --- 第一轮：处理短RR间隔（从后向前，避免索引失效）---
            short_indices = [i for i in abnormal_indices if rr_intervals_ms[i] < lower_bound]
            for idx in reversed(short_indices):
                r1 = heartindex[idx]
                r2 = heartindex[idx + 1]
                corrected_label = self.handle_short_rr_interval(ecg, corrected_label, r1, r2)
                correction_info.append(
                    f'Short RR corrected: {rr_intervals_ms[idx]:.1f}ms (thr={lower_bound:.1f}ms)'
                )

            # --- 重新计算参考RR，再处理长RR间隔 ---
            updated_heartindex = np.where(corrected_label == 1)[0]
            if len(updated_heartindex) > 1:
                updated_rr_ms = np.diff(updated_heartindex) / self.sampling_rate * 1000
                if len(updated_heartindex) >= 3:
                    # 短间隔清理后峰值更干净，重算参考RR和阈值
                    reference_rr = self.calculate_reference_interval(updated_rr_ms)
                    matching_interval = self.find_matching_interval(reference_rr)
                    upper_bound = reference_rr * matching_interval['upper_bound_factor']
                    reference_amplitude = self._compute_reference_amplitude(ecg, updated_heartindex)
                long_indices = np.where(updated_rr_ms > upper_bound)[0]

                for idx in long_indices:
                    r1 = updated_heartindex[idx]
                    r2 = updated_heartindex[idx + 1]
                    corrected_label = self.handle_long_rr_interval(
                        ecg, corrected_label, r1, r2, reference_rr, reference_amplitude
                    )
                    correction_info.append(
                        f'Long RR processed: {updated_rr_ms[idx]:.1f}ms (thr={upper_bound:.1f}ms)'
                    )

        # 调整所有R波到局部绝对值最大点
        corrected_label = self.adjust_r_wave_positions(ecg, corrected_label)

        corrected_peaks = np.where(corrected_label == 1)[0]

        return corrected_label, {
            "message": "success",
            "corrections": correction_info,
            "peaks": corrected_peaks,
            "reference_rr": reference_rr
        }


# 测试函数
def test_rr_corrector():
    """
    测试RR间隔校正器
    """
    sampling_rate = 512
    duration = 30
    samples = sampling_rate * duration

    t = np.linspace(0, duration, samples)
    ecg = np.sin(2 * np.pi * 1 * t)

    r_peaks = np.arange(0, samples, int(sampling_rate * 0.8))
    labels = np.zeros(samples)
    labels[r_peaks] = 1

    labels[r_peaks[5]] = 0
    labels[r_peaks[10] + int(sampling_rate * 0.2)] = 1

    corrector = RRIntervalCorrector(sampling_rate)
    corrected_labels, info = corrector.correct(ecg, labels)

    original_peaks = np.where(labels == 1)[0]
    corrected_peaks = np.where(corrected_labels == 1)[0]

    print(f"Original peaks: {len(original_peaks)}")
    print(f"Corrected peaks: {len(corrected_peaks)}")
    print(f"Message: {info['message']}")
    print(f"Corrections: {len(info.get('corrections', []))}")
    print(f"Reference RR: {info.get('reference_rr', 0):.2f}ms")

    return True


if __name__ == '__main__':
    print("Testing RRIntervalCorrector...")
    test_rr_corrector()
    print("Test completed successfully!")
