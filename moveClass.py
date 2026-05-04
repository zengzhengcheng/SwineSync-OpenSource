import pandas as pd
from pathlib import Path
import numpy as np
import os
from datetime import datetime, timedelta
from tzlocal import get_localzone
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


def get_true_angle_diff(series):
    diff = series.diff()
    diff = np.where(diff > 180, diff - 360, diff)
    diff = np.where(diff < -180, diff + 360, diff)
    return diff


def parse_time(t_str):
    """将HH:MM:SS.fff转换为总秒数（保留毫秒精度）"""
    try:
        h, m, s = t_str.split(':', 2)
        s_parts = s.split('.', 1)
        hours = int(h)
        minutes = int(m)
        seconds = int(s_parts[0])
        milliseconds = int(s_parts[1]) if len(s_parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    except (ValueError, AttributeError, IndexError):
        return None


class MoveData:
    is_english = False

    def __init__(self, movefilepath, picturepath, savepath, onedir=False, is_english=False):
        self.oneDir = onedir
        self.is_english = is_english
        MoveData.is_english = is_english
        self.movefilepath = movefilepath
        self.picturepath = picturepath

        if savepath:
            self.savepath = os.path.dirname(savepath)
            full_filename = os.path.basename(savepath)
            self.saveFileName = os.path.splitext(full_filename)[0]
        else:
            self.savepath = os.getcwd()
            self.saveFileName = "movement_analysis"

        # A7：跨文件累计 zeroclan 联合插值行数
        self._n_zeroclan_imputed_total = 0

        self.getAllFile()
        self.getAllData()
        self.dataClean()
        self.calculateAngle()

    def getAllFile(self):
        angle_files = {}
        for path in Path(self.movefilepath).rglob('*'):
            if path.is_file() and path.name.startswith('Angle'):
                angle_files[path.name] = str(path.resolve())
        self.angle_files = angle_files
        print(self.angle_files)

    def getAllData(self):
        dflist = []
        total_files = len(self.angle_files)
        if MoveData.is_english:
            print(f"Processing {total_files} files...")
        else:
            print(f"正在处理 {total_files} 个文件...")

        for i, file in enumerate(self.angle_files):
            if MoveData.is_english:
                print(f"Processing file {i+1}/{total_files}: {file}")
            else:
                print(f"正在处理文件 {i+1}/{total_files}：{file}")
            df = self.readAngleTxt(file, self.angle_files[file])
            dflist.append(df)

        if MoveData.is_english:
            print("Combining all data...")
        else:
            print("正在合并所有数据...")

        combined_df = pd.concat(dflist)
        sorted_df = combined_df.sort_index()
        self.moveData = sorted_df

        # 跨天事件总览：n_cross_day == 0 = 单日内文件；== 1 = 自然 24:00 跨天；>= 2 = 异常。
        # 用户可对照 first_ts / last_ts (HH:MM:SS, reassign 前的原始字符串) + filename 中的日期，
        # 判断文件是否被错误跨天到了不该有它的日期 (例: 文件名标 2025-01-14 但 last_ts 落到 15:xx,
        # 配合 n_cross_day=1 → 被合并到 Jan 15, 可能造成 Jan 15 该小时段采样率异常飙高)。
        log = getattr(self, '_cross_day_log', [])
        if log:
            if MoveData.is_english:
                print(f"\n[Per-file cross-day summary]  (== 0 within-day | == 1 natural midnight | >= 2 ANOMALY)")
                print(f"  {'flag':>4}  {'n_cross':>7}  {'rows':>9}  {'first_ts':<16}  {'last_ts':<16}  file")
            else:
                print(f"\n[每文件跨天事件总览]  (== 0 单日内 | == 1 自然 24:00 跨天 | >= 2 异常)")
                print(f"  {'flag':>4}  {'n_cross':>7}  {'rows':>9}  {'first_ts':<16}  {'last_ts':<16}  file")
            for entry in log:
                flag = '  !!' if entry['n_cross_day'] >= 2 else '    '
                print(f"  {flag}  {entry['n_cross_day']:>7}  {entry['n_rows']:>9}  "
                      f"{entry['first_ts']:<16}  {entry['last_ts']:<16}  {entry['file']}")

    def readAngleTxt(self, filename, filepath):
        if "Angle" not in filename:
            return None

        rawdata = pd.read_csv(filepath, sep=r'\s+', header=None, names=["timestamp", "x", "y", "z", "t"])
        import re
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', filename.split(".")[0])
        if not date_match:
            raise ValueError(f"文件名中未找到有效日期: {filename}")
        date_str = date_match.group()
        year, month, day = map(int, date_str.split("-"))
        base_date = datetime(year, month, day)
        starttime = base_date

        total_seconds = pd.to_numeric(
            rawdata['timestamp'].apply(parse_time),
            errors='coerce'
        )
        rawdata['timedelta'] = pd.to_timedelta(total_seconds, unit='s')

        time_diff = rawdata['timedelta'].diff()
        current_time_seconds = rawdata['timedelta'].dt.total_seconds()
        is_early_morning = current_time_seconds <= 1800
        is_large_jump = time_diff < pd.Timedelta(minutes=-30)
        prev_time_seconds = current_time_seconds.shift(1)
        is_prev_late_night = prev_time_seconds >= 84600
        # 阈值历史: f21a6b4=−12h → 中间退化到 −4h → 0598a7c 进一步退化到 −2h
        # 恢复到 −12h: 自然跨天 dt ≈ −23h 必然 catch; 几小时级缓冲 dump / 回放
        # (−2 ~ −8h backward jump) 不再被误判成跨天.
        is_very_large_jump = time_diff < pd.Timedelta(hours=-12)

        is_first_row = pd.Series([True] + [False] * (len(rawdata) - 1), index=rawdata.index)
        normal_cross_day = is_large_jump & is_early_morning & (is_prev_late_night | is_first_row)
        long_break_cross_day = is_very_large_jump
        cross_day_mask = normal_cross_day | long_break_cross_day

        # 跨天异常检测：单文件正常最多 1 次（接近 24:00 的自然跨天）。
        # 若 >= 2，说明 detection logic 把"非真正跨天的 backward jump"也判进去了，
        # 会导致最终时间戳跨度膨胀（A1 实测：8.9 天数据被铺到 15.7 天 span）。
        n_cross_day = int(cross_day_mask.sum())

        # 全量记录每个文件的 cross-day 事件数，跑完后由 getAllData 打印汇总表，
        # 方便用户验证 detection 是否如预期触发（不只是 >= 2 的"异常文件"才有反馈）。
        # 此处 rawdata['timestamp'] 仍是原始 HH:MM:SS 字符串（reassign 在 cross_day cumsum 之后）
        self._cross_day_log = getattr(self, '_cross_day_log', [])
        self._cross_day_log.append({
            'file':         filename,
            'n_rows':       len(rawdata),
            'n_cross_day':  n_cross_day,
            'first_ts':     str(rawdata['timestamp'].iloc[0])  if len(rawdata) else 'EMPTY',
            'last_ts':      str(rawdata['timestamp'].iloc[-1]) if len(rawdata) else 'EMPTY',
        })

        if n_cross_day >= 2:
            if MoveData.is_english:
                print(f"\n[WARN] Cross-day anomaly in {filename}: {n_cross_day} cross-day events triggered (expected ≤ 1)")
                print("       Triggered rows (timestamp BEFORE day-offset reassignment):")
            else:
                print(f"\n[警告] {filename} 跨天异常：触发 {n_cross_day} 次（正常文件 ≤ 1 次，会膨胀时间跨度）")
                print("       触发行（reassign 前的原始 HH:MM:SS）：")
            triggers = np.where(cross_day_mask.values)[0]
            for tloc in triggers[:10]:
                cur_str  = str(rawdata['timestamp'].iloc[tloc])
                prev_str = str(rawdata['timestamp'].iloc[tloc - 1]) if tloc > 0 else 'BOF'
                td_val   = time_diff.iloc[tloc]
                td_str   = f"{td_val.total_seconds():+.2f}s" if pd.notna(td_val) else 'NaT'
                rule = 'normal_midnight' if normal_cross_day.iloc[tloc] else 'very_large_jump'
                print(f"         row {int(tloc):>7}: prev={prev_str:<16} cur={cur_str:<16} dt={td_str:>14}  [{rule}]")
            if len(triggers) > 10:
                print(f"         ... 共 {len(triggers)} 行触发，仅展示前 10 行")
            self._cross_day_warnings = getattr(self, '_cross_day_warnings', [])
            self._cross_day_warnings.append({
                'file': filename,
                'n_cross_day': n_cross_day,
            })

        cross_day = cross_day_mask.cumsum()

        rawdata['timestamp'] = starttime + rawdata['timedelta'] + pd.to_timedelta(cross_day, unit='D')
        try:
            local_timezone = get_localzone()
        except Exception:
            local_timezone = pytz.timezone('Asia/Shanghai')
            if MoveData.is_english:
                print("Warning: Using default timezone 'Asia/Shanghai'")
            else:
                print("Warning: 使用默认时区 'Asia/Shanghai'")

        rawdata['timestamp'] = (
            rawdata['timestamp']
            .dt.tz_localize(local_timezone)
            .dt.tz_localize(None)
        )

        rawdata = self.zeroclan(rawdata, ["x", "y", "z"])

        # 原始时间戳若为整秒精度（同一秒内有多条记录），按出现顺序均匀分配亚秒偏移
        # 避免后续去重时将同秒内的有效多次采样误删
        if rawdata['timestamp'].duplicated().any():
            rank_in_second = rawdata.groupby('timestamp').cumcount()
            total_in_second = rawdata.groupby('timestamp')['timestamp'].transform('count')
            rawdata['timestamp'] = rawdata['timestamp'] + pd.to_timedelta(
                rank_in_second / total_in_second, unit='s'
            )

        rawdata.set_index('timestamp', inplace=True)
        return rawdata[['x', 'y', 'z']]

    def dataClean(self):
        if MoveData.is_english:
            print(f"{'=' * 20} 1. Loading Data {'=' * 20}")
        else:
            print(f"{'=' * 20} 1. 加载数据 {'=' * 20}")

        df = self.moveData.copy()
        original_count = len(df)
        df = df[df.index.notnull()]

        for col in ['x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['x', 'y', 'z'], how='any', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        deleted_count = original_count - len(df)

        if MoveData.is_english:
            print(f"Data cleaning completed: Deleted {deleted_count} rows with invalid data.")
            print(f"Remaining valid data: {len(df)} rows.")
        else:
            print(f"数据清洗完成：删除了 {deleted_count} 行无效数据。")
            print(f"剩余有效数据：{len(df)} 行。")

        # 断点检测
        if MoveData.is_english:
            print(f"\n{'=' * 20} 2. Data Continuity Detection (Breakpoint > 3s) {'=' * 20}")
        else:
            print(f"\n{'=' * 20} 2. 数据连续性检测 (断点 > 3秒) {'=' * 20}")

        df['dt'] = df.index.to_series().diff().dt.total_seconds()
        gap_threshold = 3.0
        gaps = df[df['dt'] > gap_threshold].copy()

        if len(gaps) > 0:
            if MoveData.is_english:
                print(f"Found {len(gaps)} significant data breaks:")
                print(f"{'Break Start Time':<25} | {'Resume Time':<25} | {'Duration (s)'}")
                print("-" * 70)
            else:
                print(f"发现 {len(gaps)} 处明显的数据断裂：")
                print(f"{'断裂开始时间':<25} | {'数据恢复时间':<25} | {'中断时长(秒)'}")
                print("-" * 70)
            for end_time, row in gaps.iterrows():
                duration = row['dt']
                start_time = end_time - pd.Timedelta(seconds=duration)
                print(f"{str(start_time):<25} | {str(end_time):<25} | {duration:.2f} s")
        else:
            if MoveData.is_english:
                print("No breaks exceeding 3 seconds found.")
            else:
                print("未发现超过 3 秒的断裂。")

        # 采样率质量分析
        hz_per_second = df.resample('1S').size()
        quality_df = pd.DataFrame({'hz': hz_per_second})
        LOW_THRESH = 5
        HIGH_THRESH = 15
        quality_df['status'] = 'Normal'
        quality_df.loc[quality_df['hz'] == 0, 'status'] = 'Missing (0 Hz)'
        quality_df.loc[(quality_df['hz'] > 0) & (quality_df['hz'] < LOW_THRESH), 'status'] = 'Too Low (<5 Hz)'
        quality_df.loc[quality_df['hz'] > HIGH_THRESH, 'status'] = 'Too High (>15 Hz)'
        quality_df['date_hour'] = quality_df.index.floor('H')
        hourly_report = quality_df.groupby(['date_hour', 'status']).size().unstack(fill_value=0)

        problematic_hours = hourly_report[
            (hourly_report.get('Too High (>15 Hz)', 0) > 0) |
            (hourly_report.get('Too Low (<5 Hz)', 0) > 0)
        ].copy()
        problematic_hours['total_abnormal_seconds'] = (
            problematic_hours.get('Too High (>15 Hz)', 0) +
            problematic_hours.get('Too Low (<5 Hz)', 0)
        )
        problematic_hours = problematic_hours.sort_values('total_abnormal_seconds', ascending=False)

        if MoveData.is_english:
            print("\n====== Data Quality Overview (Based on samples per second) ======")
        else:
            print("\n====== 数据质量概览 (基于每秒采样数) ======")
        print(quality_df['status'].value_counts())

        if not problematic_hours.empty:
            if MoveData.is_english:
                print("\n====== Top 10 Time Periods with Most Severe Sampling Rate Abnormalities ======")
                print(f"{'Time Period':<25} | {'Too High (s)':<12} | {'Too Low (s)':<12} | {'Missing (s)':<12}")
            else:
                print("\n====== 采样率异常最为严重的 10 个小时段 ======")
                print(f"{'时间段':<25} | {'过高(秒数)':<12} | {'过低(秒数)':<12} | {'缺失(秒数)':<12}")
            print("-" * 70)
            for index, row in problematic_hours.head(10).iterrows():
                high_cnt = row.get('Too High (>15 Hz)', 0)
                low_cnt = row.get('Too Low (<5 Hz)', 0)
                miss_cnt = row.get('Missing (0 Hz)', 0)
                print(f"{str(index):<25} | {high_cnt:<12} | {low_cnt:<12} | {miss_cnt:<12}")

        self._plot_sampling_quality(quality_df, LOW_THRESH, HIGH_THRESH)

        self.data_quality_results = {
            'deleted_count': deleted_count,
            'remaining_count': len(df),
            'gaps': gaps,
            'quality_df': quality_df,
            'problematic_hours': problematic_hours,
            'hourly_report': hourly_report,
            'low_thresh': LOW_THRESH,
            'high_thresh': HIGH_THRESH,
            # A7：跨所有 readAngleTxt 调用累计的联合插值行数
            'n_zeroclan_imputed': int(getattr(self, '_n_zeroclan_imputed_total', 0)),
        }
        self.moveData = df

        # A8：朝向校准 metadata（夜间 23:00–04:00 的三轴中位数）
        self._calibrate_posture_zero()

    def _plot_sampling_quality(self, quality_df, LOW_THRESH, HIGH_THRESH):
        os.makedirs(self.savepath, exist_ok=True)
        plot_data = quality_df[quality_df['hz'] > 0]

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        colors = ['red' if hz > HIGH_THRESH else 'orange' if hz < LOW_THRESH else 'blue'
                  for hz in plot_data['hz']]
        axes[0].scatter(plot_data.index, plot_data['hz'], s=2, alpha=0.6, c=colors)
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Too High'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Too Low'),
        ]
        axes[0].legend(handles=custom_lines, loc='upper right')
        axes[0].axhline(y=10, color='green', linestyle='-', linewidth=1, label='Target (10Hz)')
        axes[0].axhline(y=HIGH_THRESH, color='red', linestyle='--', linewidth=0.8)
        axes[0].axhline(y=LOW_THRESH, color='orange', linestyle='--', linewidth=0.8)
        title = 'Sample Rate Fluctuation Over Time' if self.is_english else '全时段采样率波动图'
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Samples/Second (Hz)', fontsize=12)
        axes[0].grid(alpha=0.3, linestyle='--')

        bin_edges = np.arange(0, 31, 1)
        sns.histplot(plot_data['hz'], bins=bin_edges, kde=True, color='purple', alpha=0.7, ax=axes[1])
        axes[1].axvline(x=10, color='green', linestyle='--', linewidth=1.5, label='Target (10Hz)')
        axes[1].axvline(x=HIGH_THRESH, color='red', linestyle='--', linewidth=1)
        axes[1].axvline(x=LOW_THRESH, color='orange', linestyle='--', linewidth=1)
        title = 'Sampling Rate Distribution Histogram' if self.is_english else '采样率分布直方图'
        axes[1].set_title(title, fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Hz', fontsize=12)
        axes[1].set_ylabel('Count' if self.is_english else '数量', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(os.path.join(self.savepath, self.saveFileName + "_sampling_rate_distribution.png"), dpi=600)
        plt.close()

    def calculateAngle(self):
        """计算合成角速度，检测并清洗伪影"""
        df = self.moveData
        gap_threshold = 3.0

        # 计算真实角度差（处理±180度跳变）
        df['d_x'] = pd.Series(get_true_angle_diff(df['x']), index=df.index)
        df['d_y'] = pd.Series(get_true_angle_diff(df['y']), index=df.index)
        df['d_z'] = pd.Series(get_true_angle_diff(df['z']), index=df.index)

        # A4：断点处角度差无物理意义，置 NaN（不再归零，避免 1 s mean 把断点伪装成静息）
        invalid_mask = (df['dt'] > gap_threshold) | (df['dt'] <= 0.001) | (df['dt'].isna())
        df.loc[invalid_mask, ['d_x', 'd_y', 'd_z']] = np.nan

        # 合成角位移（°）
        df['Angle_Mag'] = np.sqrt(df['d_x']**2 + df['d_y']**2 + df['d_z']**2)
        df.loc[invalid_mask, 'Angle_Mag'] = np.nan

        # 合成角速度（°/s）
        df['Gyro_Mag'] = df['Angle_Mag'] / df['dt']
        df.loc[invalid_mask, 'Gyro_Mag'] = np.nan

        # 角加速度（Jerk）：角速度的逐点变化率，断点处置NaN避免伪峰
        jerk = df['Gyro_Mag'].diff().abs()
        jerk[df['dt'] > 0.5] = np.nan
        df['Jerk'] = jerk

        if MoveData.is_english:
            print(f"\n{'=' * 20} Statistical Analysis and Threshold Setting {'=' * 20}")
        else:
            print(f"\n{'=' * 20} 统计分析与阈值设定 {'=' * 20}")

        active_gyro = df[df['Gyro_Mag'] > 1.0]['Gyro_Mag']
        stats = active_gyro.describe(percentiles=[0.5, 0.75, 0.99, 0.999])
        ACTIVE_BASE = stats['75%']
        SUSPICIOUS_LIMIT = max(stats['99.9%'], 300.0)

        if MoveData.is_english:
            print(f"  - Median (P50): {stats['50%']:.2f} °/s")
            print(f"  - Active threshold (P75): {stats['75%']:.2f} °/s")
            print(f"  - Extreme threshold (P99.9): {stats['99.9%']:.2f} °/s")
            print(f"  1. Suspected abnormal high value: > {SUSPICIOUS_LIMIT:.2f} °/s")
            print(f"  2. Neighbor active threshold: > {ACTIVE_BASE:.2f} °/s")
        else:
            print(f"  - 中位数 (P50): {stats['50%']:.2f} °/s")
            print(f"  - 活跃线 (P75): {stats['75%']:.2f} °/s")
            print(f"  - 极值线 (P99.9): {stats['99.9%']:.2f} °/s")
            print(f"  1. 疑似异常高值线: > {SUSPICIOUS_LIMIT:.2f} °/s")
            print(f"  2. 邻居活跃达标线: > {ACTIVE_BASE:.2f} °/s")

        # 上下文异常检测
        df['Prev_Gyro'] = df['Gyro_Mag'].shift(1).fillna(0)
        df['Next_Gyro'] = df['Gyro_Mag'].shift(-1).fillna(0)
        df['Neighbor_Max'] = df[['Prev_Gyro', 'Next_Gyro']].max(axis=1)
        df['Isolation_Ratio'] = df['Gyro_Mag'] / (df['Neighbor_Max'] + 1.0)

        mask_real = (
            (df['Gyro_Mag'] > SUSPICIOUS_LIMIT) &
            ((df['Neighbor_Max'] > ACTIVE_BASE) | (df['Isolation_Ratio'] < 10.0))
        )
        mask_artifact = (
            (df['Gyro_Mag'] > SUSPICIOUS_LIMIT) &
            (df['Neighbor_Max'] < ACTIVE_BASE) &
            (df['Isolation_Ratio'] > 10.0)
        )

        df['Gyro_Status'] = 'Normal'
        df.loc[mask_real, 'Gyro_Status'] = 'Real_Activity'
        df.loc[mask_artifact, 'Gyro_Status'] = 'Artifact'

        n_real = mask_real.sum()
        n_artifact = mask_artifact.sum()

        if MoveData.is_english:
            print(f"Detection results:")
            print(f"  - Real intense movement: {n_real} points (retained)")
            print(f"  - Artifacts/Noise: {n_artifact} points (cleaned)")
        else:
            print(f"检测结果：")
            print(f"  - 真实剧烈运动点: {n_real} 个 (予以保留)")
            print(f"  - 伪影/噪点: {n_artifact} 个 (予以清洗)")

        if n_artifact > 0:
            label = "[Artifact Examples]:" if MoveData.is_english else "[伪影示例]:"
            print(f"\n{label}")
            print(df[mask_artifact][['Gyro_Mag', 'Neighbor_Max', 'Isolation_Ratio']].head(3))

        # A4b（大脑扩展决议）：伪影点统一置 NaN，与 §2.4 dt 断点同语义；
        # d_x/y/z 一并 NaN，保持 RotEnergy 类衍生量不被伪影残值污染
        df.loc[mask_artifact, ['d_x', 'd_y', 'd_z', 'Angle_Mag', 'Gyro_Mag']] = np.nan

        self.angle_results = {
            'n_real': n_real,
            'n_artifact': n_artifact,
            'n_artifact_set_to_nan': int(n_artifact),  # A4b：检测到即 NaN，二者数值相等
            'artifact_examples': df[mask_artifact][['Gyro_Mag', 'Neighbor_Max', 'Isolation_Ratio']].head(3),
            'real_examples': df[mask_real][['Gyro_Mag', 'Neighbor_Max', 'Isolation_Ratio']].head(3),
            'ACTIVE_BASE': ACTIVE_BASE,
            'SUSPICIOUS_LIMIT': SUSPICIOUS_LIMIT,
        }

        self.moveCleanData = df
        self.generate_secondly_summary()
        self.generate_advanced_visualizations()

    def generate_secondly_summary(self):
        """生成每秒数据汇总（1Hz），共含37列特征"""
        if MoveData.is_english:
            print(f"\n{'='*20} Generating Per-Second Data Summary (1Hz) {'='*20}")
        else:
            print(f"\n{'='*20} 生成每秒数据汇总 (1Hz) {'='*20}")

        df = self.moveCleanData.copy()

        # ==========================================
        # 步骤一：预计算派生列
        # ==========================================

        # A1：板载欧拉角 2 s 低通（旧名 Static_X/Y/Z 改为 Euler_X/Y/Z_static，
        # 与主线"基于加速度反算的几何 Roll/Pitch"做命名隔离）
        df['Euler_X_static'] = df['x'].rolling(window='2s', center=True).mean()
        df['Euler_Y_static'] = df['y'].rolling(window='2s', center=True).mean()
        df['Euler_Z_static'] = df['z'].rolling(window='2s', center=True).mean()
        df[['Euler_X_static', 'Euler_Y_static', 'Euler_Z_static']] = (
            df[['Euler_X_static', 'Euler_Y_static', 'Euler_Z_static']].bfill().ffill()
        )

        # A1：低通后的两轴姿态代理（不是几何 Roll/Pitch；下游论文引用走 Euler_X_lpf 命名）
        df['Euler_X_lpf'] = df['Euler_X_static']
        df['Euler_Y_lpf'] = df['Euler_Y_static']

        # 各轴绝对角位移（用于单轴均值特征）
        df['abs_d_x'] = df['d_x'].abs()
        df['abs_d_y'] = df['d_y'].abs()
        df['abs_d_z'] = df['d_z'].abs()

        # §3.1：旋转能量预计算列（dt 不固定，严格物理量需 Gyro_Mag² × dt）
        df['Gyro_Mag_Sq']            = df['Gyro_Mag'] ** 2
        df['Gyro_Energy_Increment']  = df['Gyro_Mag_Sq'] * df['dt']

        # A4：每秒非 NaN 样本数（与 Data_Count = 原始 x 计数 区分）
        df['valid_sample'] = df['Angle_Mag'].notna().astype(int)

        # ==========================================
        # 步骤二：数据驱动阈值
        # ==========================================
        active_angle = df['Angle_Mag'][df['Angle_Mag'] > 0]
        thresh_low  = active_angle.quantile(0.33) if len(active_angle) > 0 else 0.01
        thresh_high = active_angle.quantile(0.75) if len(active_angle) > 0 else 0.05

        # 侧卧阈值：|Euler_X_lpf| P90，限制在 20-60°之间（A1 配套，阈值含义不变）
        thresh_lying = float(np.clip(df['Euler_X_lpf'].abs().quantile(0.90), 20.0, 60.0))

        # 高活动阈值（用于Active_Fraction）：活跃Gyro_Mag的P75
        active_gyro = df[df['Gyro_Mag'] > 1.0]['Gyro_Mag']
        thresh_active = active_gyro.quantile(0.75) if len(active_gyro) > 0 else 10.0

        if MoveData.is_english:
            print(f"[Data-driven Thresholds]")
            print(f"  Activity - Rest/Low:   <= {thresh_low:.4f} °")
            print(f"  Activity - Low/High:   <= {thresh_high:.4f} °")
            print(f"  Posture  - Lying:      |Euler_X_lpf| > {thresh_lying:.1f} °")
            print(f"  Active fraction base:  Gyro > {thresh_active:.2f} °/s")
        else:
            print(f"[数据驱动阈值]")
            print(f"  活动强度 - 静息/低强度分界: <= {thresh_low:.4f} °")
            print(f"  活动强度 - 低强度/高强度分界: <= {thresh_high:.4f} °")
            print(f"  姿态 - 侧卧判断:  |Euler_X_lpf| > {thresh_lying:.1f} °")
            print(f"  高活动基线:  Gyro > {thresh_active:.2f} °/s")

        # ==========================================
        # 步骤三：标记点级行为状态
        # ==========================================
        # A4：断点处 Angle_Mag/Gyro_Mag 已为 NaN，is_xxx 也置 NaN（避免被 1 s mean 视为 0 静息）
        mask_valid = df['Angle_Mag'].notna()
        df['is_rest']     = (df['Angle_Mag'] <= thresh_low).astype(float).where(mask_valid)
        df['is_moderate'] = ((df['Angle_Mag'] > thresh_low) & (df['Angle_Mag'] <= thresh_high)).astype(float).where(mask_valid)
        df['is_vigorous'] = (df['Angle_Mag'] > thresh_high).astype(float).where(mask_valid)
        df['is_lying']    = (df['Euler_X_lpf'].abs() > thresh_lying).astype(float).where(mask_valid)
        df['is_active']   = (df['Gyro_Mag'] > thresh_active).astype(float).where(mask_valid)

        # ==========================================
        # 步骤四：向量化重采样聚合
        # ==========================================
        agg_logic = {
            'Angle_Mag':              ['mean', 'sum', 'max', 'min', 'std'],
            'Gyro_Mag':               ['mean', 'max'],
            'Gyro_Mag_Sq':            ['mean', 'max'],          # §3.1 RotEnergy_1s_mean / max
            'Gyro_Energy_Increment':  'sum',                    # §3.1 RotEnergy_1s_sum
            'x':                      'count',
            'd_x':                    'std',
            'd_y':                    'std',
            'd_z':                    'std',
            'abs_d_x':                'mean',
            'abs_d_y':                'mean',
            'abs_d_z':                'mean',
            'Euler_X_lpf':            ['mean', 'std'],          # A1：原 Roll
            'Euler_Y_lpf':            ['mean', 'std'],          # A1：原 Pitch
            'Jerk':                   'mean',
            'is_rest':                'mean',
            'is_moderate':            'mean',
            'is_vigorous':            'mean',
            'is_lying':               'mean',
            'is_active':              'mean',
            'valid_sample':           'sum',                    # A4：每秒非 NaN 样本数
        }

        df_1s = df.resample('1s').agg(agg_logic)
        df_1s.columns = ['_'.join(col).strip() for col in df_1s.columns.values]

        rename_dict = {
            'Angle_Mag_mean':              'Angle_Mean',
            'Angle_Mag_sum':               'Angle_Sum',
            'Angle_Mag_max':               'Angle_Max',
            'Angle_Mag_min':               'Angle_Min',
            'Angle_Mag_std':               'Angle_Std',
            'Gyro_Mag_mean':               'Gyro_Mean',
            'Gyro_Mag_max':                'Gyro_Max',
            'Gyro_Mag_Sq_mean':            'RotEnergy_1s_mean',     # §3.1
            'Gyro_Mag_Sq_max':             'RotEnergy_1s_max',      # §3.1
            'Gyro_Energy_Increment_sum':   'RotEnergy_1s_sum',      # §3.1
            'x_count':                     'Data_Count',
            'd_x_std':                     'dX_Std',
            'd_y_std':                     'dY_Std',
            'd_z_std':                     'dZ_Std',
            'abs_d_x_mean':                'dX_Mean',
            'abs_d_y_mean':                'dY_Mean',
            'abs_d_z_mean':                'dZ_Mean',
            'Euler_X_lpf_mean':            'Euler_X_lpf_Mean',      # A1
            'Euler_X_lpf_std':             'Euler_X_lpf_Std',       # A1
            'Euler_Y_lpf_mean':            'Euler_Y_lpf_Mean',      # A1
            'Euler_Y_lpf_std':             'Euler_Y_lpf_Std',       # A1
            'Jerk_mean':                   'Jerk_Mean',
            'is_rest_mean':                'Frac_Rest',
            'is_moderate_mean':            'Frac_Moderate',
            'is_vigorous_mean':            'Frac_Vigorous',
            'is_lying_mean':               'Frac_Lying',
            'is_active_mean':              'Active_Fraction',
            'valid_sample_sum':            'Valid_Sample_Count',    # A4
        }
        df_1s.rename(columns=rename_dict, inplace=True)

        # A1：旧列名作为 alias 保留一期（避免下游 extract_features_v2.py 立即崩）
        df_1s['Roll_Mean']  = df_1s['Euler_X_lpf_Mean']
        df_1s['Roll_Std']   = df_1s['Euler_X_lpf_Std']
        df_1s['Pitch_Mean'] = df_1s['Euler_Y_lpf_Mean']
        df_1s['Pitch_Std']  = df_1s['Euler_Y_lpf_Std']

        # 后处理：可从聚合列直接推导的特征
        df_1s['Angle_Range'] = df_1s['Angle_Max'] - df_1s['Angle_Min']
        df_1s['Axis_Dominance'] = df_1s[['dX_Std', 'dY_Std', 'dZ_Std']].values.argmax(axis=1)  # 0=X, 1=Y, 2=Z

        # ==========================================
        # A4b 前置过滤：把"全空 / 全 NaN 秒"在 per-second 循环 *之前* 丢弃。
        # 旧版在循环结束后才过滤,导致 1.35M 秒里 ~586k 全空秒也跑进 hot loop
        # (虽然每次都早退,但 los/his 索引 + 类型转换的开销仍累积).
        # 同时把"正在逐秒计算高阶特征,共 N 秒"的 N 直接显示成实际处理秒数,
        # 不再被 resample span (=数据跨度,含所有空隙) 误导.
        # ==========================================
        total_resample = len(df_1s)
        df_1s = df_1s[df_1s['Valid_Sample_Count'] > 0].copy()
        n_dropped_pre = total_resample - len(df_1s)
        if MoveData.is_english:
            print(f"  Pre-loop filter: kept {len(df_1s)}/{total_resample} secs "
                  f"(dropped {n_dropped_pre} empty / all-NaN secs before per-second feature loop)")
        else:
            print(f"  循环前过滤: 保留 {len(df_1s)}/{total_resample} 秒 "
                  f"(丢弃 {n_dropped_pre} 全空 / 全 NaN 秒,避免空跑)")

        # ==========================================
        # 步骤五：逐秒循环计算高阶特征（numpy 向量化版）
        #
        # A2：频谱源 d_z（带正负 → ZCR / FFT 物理有意义）+ Gyro_Mag 冗余对照（_GMag）
        # A3：fs = 该秒实际样本数（动态）
        # A4：NaN 断点 / 伪影样本不参与
        # CV / Skew / Kurt / Quartile / Autocorr 仍对 Angle_Mag
        #
        # 性能优化（数值与原版完全等价）：
        #   (a) 用 np.searchsorted 在 ts_ns 上批量找每秒边界，避免 df.loc[idx:end_time]
        #       千万次 O(log N) DatetimeIndex 调用 + Series 构造
        #   (b) Spectral_Entropy 与 Dominant_Frequency 共享同一次 fft（self._spec_pair）
        #       —— 8 次 FFT/秒 减半到 4 次/秒
        #   (c) 输出列预分配 np.zeros，避免 list.append 扩容
        #   (d) A4b 前置过滤已把 1.35M 秒砍到 ~766k,本循环只跑实际有数据的秒
        # ==========================================
        ts_ns      = df.index.values.astype('datetime64[ns]').astype('int64')
        amag_arr   = df['Angle_Mag'].values
        dz_arr     = df['d_z'].values
        gmag_arr   = df['Gyro_Mag'].values

        sec_starts_ns = df_1s.index.values.astype('datetime64[ns]').astype('int64')
        NS_PER_SEC = 1_000_000_000
        los = np.searchsorted(ts_ns, sec_starts_ns,                  side='left')
        his = np.searchsorted(ts_ns, sec_starts_ns + NS_PER_SEC,     side='left')

        total_seconds = len(df_1s)
        spec_dz   = np.zeros(total_seconds)
        zcr_dz    = np.zeros(total_seconds)
        domf_dz   = np.zeros(total_seconds)
        peak_dz   = np.zeros(total_seconds, dtype=np.int64)

        spec_gm   = np.zeros(total_seconds)
        zcr_gm    = np.zeros(total_seconds)
        domf_gm   = np.zeros(total_seconds)
        peak_gm   = np.zeros(total_seconds, dtype=np.int64)

        cv_arr        = np.zeros(total_seconds)
        skew_arr      = np.zeros(total_seconds)
        kurt_arr      = np.zeros(total_seconds)
        p25_arr       = np.zeros(total_seconds)
        p75_arr       = np.zeros(total_seconds)
        autocorr_arr  = np.zeros(total_seconds)

        if MoveData.is_english:
            print(f"Calculating per-second features for {total_seconds} seconds...")
        else:
            print(f"正在逐秒计算高阶特征，共 {total_seconds} 秒...")

        # 局部别名，缩短 attribute lookup（hot loop 微优化）
        spec_pair  = self._spec_pair
        peak_count = self.count_peaks
        isnan_     = np.isnan
        sign_      = np.sign
        diff_      = np.diff
        absolute_  = np.abs
        percentile_ = np.percentile
        dot_       = np.dot

        for i in range(total_seconds):
            if (i + 1) % 50000 == 0 or (i + 1) == total_seconds:
                if MoveData.is_english:
                    print(f"  {i+1}/{total_seconds} ({((i+1)/total_seconds)*100:.1f}%)")
                else:
                    print(f"  {i+1}/{total_seconds} ({((i+1)/total_seconds)*100:.1f}%)")

            lo = int(los[i])
            hi = int(his[i])
            if lo >= hi:
                continue   # 该秒无原始样本; 默认 0, 后续 Valid_Sample_Count > 0 过滤掉

            a_raw  = amag_arr[lo:hi]
            dz_raw = dz_arr[lo:hi]
            gm_raw = gmag_arr[lo:hi]

            a_seg  = a_raw[~isnan_(a_raw)]
            dz_seg = dz_raw[~isnan_(dz_raw)]
            gm_seg = gm_raw[~isnan_(gm_raw)]

            n_a   = len(a_seg)
            n_dz  = len(dz_seg)
            n_gm  = len(gm_seg)

            # d_z 主频谱（共享 FFT）
            fs_dz = max(1, n_dz) if n_dz > 0 else max(1, hi - lo)
            spec_dz[i], domf_dz[i] = spec_pair(dz_seg, fs_dz)
            if n_dz >= 2:
                zcr_dz[i] = float(absolute_(diff_(sign_(dz_seg))).sum() / 2.0)
            if n_dz >= 3:
                peak_dz[i] = peak_count(dz_seg)

            # Gyro_Mag 冗余对照（共享 FFT）
            fs_gm = max(1, n_gm) if n_gm > 0 else fs_dz
            spec_gm[i], domf_gm[i] = spec_pair(gm_seg, fs_gm)
            if n_gm >= 2:
                zcr_gm[i] = float(absolute_(diff_(sign_(gm_seg))).sum() / 2.0)
            if n_gm >= 3:
                peak_gm[i] = peak_count(gm_seg)

            # Angle_Mag 分布特征
            if n_a > 0:
                mean_a = a_seg.mean()
                if mean_a != 0:
                    cv_arr[i] = float(a_seg.std() / mean_a)
                p25_arr[i] = float(percentile_(a_seg, 25))
                p75_arr[i] = float(percentile_(a_seg, 75))
                if n_a >= 2:
                    s = pd.Series(a_seg)
                    autocorr_arr[i] = float(s.autocorr(lag=1))
                    if n_a >= 3:
                        skew_arr[i] = float(s.skew())
                        if n_a >= 4:
                            kurt_arr[i] = float(s.kurtosis())

        # 主频谱列（d_z 来源）
        df_1s['Spectral_Entropy_dz']     = spec_dz
        df_1s['Zero_Crossing_Rate_dz']   = zcr_dz
        df_1s['Dominant_Frequency_dz']   = domf_dz
        df_1s['Peak_Count_dz']           = peak_dz

        # 冗余对照（Gyro_Mag 来源）
        df_1s['Spectral_Entropy_GMag']   = spec_gm
        df_1s['Zero_Crossing_Rate_GMag'] = zcr_gm
        df_1s['Dominant_Frequency_GMag'] = domf_gm
        df_1s['Peak_Count_GMag']         = peak_gm

        # A2 alias：旧列名一期保留（值 = _dz 主版本），下游 extract_features_v2.py 暂不改
        df_1s['Spectral_Entropy']        = df_1s['Spectral_Entropy_dz']
        df_1s['Zero_Crossing_Rate']      = df_1s['Zero_Crossing_Rate_dz']
        df_1s['Dominant_Frequency']      = df_1s['Dominant_Frequency_dz']
        df_1s['Peak_Count']              = df_1s['Peak_Count_dz']

        df_1s['Coefficient_of_Variation'] = cv_arr
        df_1s['Skewness']                 = skew_arr
        df_1s['Kurtosis']                 = kurt_arr
        df_1s['Angle_P25']                = p25_arr
        df_1s['Angle_P75']                = p75_arr
        df_1s['Autocorr_Lag1']            = autocorr_arr

        # 注: A4b 过滤已前移到 per-second 循环之前,这里不再二次过滤 (重复 no-op)

        if MoveData.is_english:
            print(f"Summary complete. Shape: {df_1s.shape}")
            print(f"Columns ({len(df_1s.columns)}): {list(df_1s.columns)}")
        else:
            print(f"汇总完成。形状: {df_1s.shape}")
            print(f"列名 (共{len(df_1s.columns)}列): {list(df_1s.columns)}")

        self.finaldata = df_1s

    def calculate_spectral_entropy(self, signal):
        if len(signal) < 3:
            return 0
        fft_result = np.fft.fft(signal)
        power_spectrum = np.abs(fft_result) ** 2
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        return -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))

    def calculate_zero_crossing_rate(self, signal):
        if len(signal) < 2:
            return 0
        return np.sum(np.abs(np.diff(np.sign(signal)))) / 2

    def calculate_dominant_frequency(self, signal, fs=None):
        """A3：fs=None 时取 max(1, len(signal))（= 该秒实际样本数）作为采样率，
        避免对 5-15 Hz 浮动采样的样本固定按 10 Hz 估算导致频率刻度偏移。"""
        if len(signal) < 3:
            return 0
        if fs is None:
            fs = max(1, len(signal))
        fft_result = np.fft.fft(signal)
        power_spectrum = np.abs(fft_result) ** 2
        freqs = np.fft.fftfreq(len(signal), 1 / fs)
        positive_freqs = freqs[freqs > 0]
        positive_power = power_spectrum[freqs > 0]
        if len(positive_power) == 0:
            return 0
        return positive_freqs[np.argmax(positive_power)]

    def _spec_pair(self, signal, fs):
        """共享 FFT 一次返回 (Spectral_Entropy, Dominant_Frequency)。

        与 calculate_spectral_entropy + calculate_dominant_frequency 数值完全等价
        （同一份 np.fft.fft 结果），但只跑一次 FFT 而非两次——把 generate_secondly_summary
        per-second hot path 的 8 次 FFT/秒 减半到 4 次/秒（A1 实测从 ~1M FFT 减到 0.5M）。
        """
        n = len(signal)
        if n < 3:
            return 0.0, 0.0
        fft_result = np.fft.fft(signal)
        power = np.abs(fft_result) ** 2
        psum = power.sum()
        entropy = 0.0
        if psum > 0:
            p = power / psum
            entropy = float(-np.sum(p * np.log2(p + 1e-10)))
        freqs = np.fft.fftfreq(n, 1.0 / fs)
        pos_mask = freqs > 0
        if not pos_mask.any():
            return entropy, 0.0
        pos_power = power[pos_mask]
        return entropy, float(freqs[pos_mask][int(np.argmax(pos_power))])

    def calculate_cv(self, signal):
        if len(signal) == 0:
            return 0
        mean = np.mean(signal)
        if mean == 0:
            return 0
        return np.std(signal) / mean

    def count_peaks(self, signal, height_ratio=0.5):
        """统计局部峰值数量，峰值高度需超过信号最大值的height_ratio倍"""
        if len(signal) < 3:
            return 0
        threshold = np.max(signal) * height_ratio
        is_peak = (
            (signal[1:-1] > signal[:-2]) &
            (signal[1:-1] > signal[2:]) &
            (signal[1:-1] > threshold)
        )
        return int(np.sum(is_peak))

    def drawPicture(self, picturepath=None):
        if picturepath is None:
            picturepath = self.picturepath
        if not os.path.exists(picturepath):
            os.makedirs(picturepath)
        picturepath = os.path.join(picturepath, self.saveFileName)
        if not os.path.exists(picturepath):
            os.makedirs(picturepath)

        df_plot = self.moveCleanData.resample('60s').agg({
            'Angle_Mag': 'sum',
            'Gyro_Mag': 'mean'
        })

        for date, daily_df in df_plot.groupby(pd.Grouper(freq='D')):
            if daily_df.empty or daily_df['Angle_Mag'].sum() == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            x_hours = (daily_df.index - date).total_seconds() / 3600

            ax1.fill_between(x_hours, daily_df['Angle_Mag'], color='steelblue', alpha=0.6, label='运动总量')
            ax1.plot(x_hours, daily_df['Angle_Mag'], color='steelblue', linewidth=1)
            y_max = daily_df['Angle_Mag'].quantile(0.99) * 1.2
            ax1.set_ylim(0, y_max if y_max > 0 else 1)
            ax1.set_ylabel('运动幅度 (°/min)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{date.date()} 角位移运动总量 (Body Activity)', fontsize=14)
            ax1.grid(alpha=0.3, linestyle='--')
            ax1.legend(loc='upper right')

            ax2.plot(x_hours, daily_df['Gyro_Mag'], color='#ff7f0e', linewidth=1.2, label='旋转强度 (Gyro)')
            ax2.set_ylabel('旋转速度 (°/s)', fontsize=12, fontweight='bold')
            ax2.set_title(f'{date.date()} 姿态旋转强度 (Rotational Intensity)', fontsize=14)
            ax2.grid(alpha=0.3, linestyle='--')
            ax2.legend(loc='upper right')

            xticks = np.arange(0, 25, 2)
            xtick_labels = [f"{int(h):02d}:00" for h in xticks]
            plt.xlabel('当日时间 (Hour)', fontsize=12)
            plt.xticks(xticks, xtick_labels, fontsize=10)
            plt.xlim(0, 24)
            plt.tight_layout()

            save_name = f'daily_{date.date()}_dual_move.png'
            save_full_path = os.path.join(picturepath, save_name)
            plt.savefig(save_full_path, dpi=120)
            plt.close(fig)
            if MoveData.is_english:
                print(f"Saved: {save_full_path}")
            else:
                print(f"已保存: {save_full_path}")

    def saveData(self, filepath=None):
        self.finaldata.to_csv(filepath)
        base, ext = os.path.splitext(filepath)
        raw_filepath = base + "_raw_angle" + ext
        self.moveCleanData.to_csv(raw_filepath)
        if MoveData.is_english:
            print(f"Raw angle data saved: {raw_filepath}")
        else:
            print(f"原始角度数据已保存: {raw_filepath}")

    def generate_advanced_visualizations(self):
        os.makedirs(self.savepath, exist_ok=True)
        self._plot_sampling_rate_stability()
        self._plot_data_completeness_heatmap()
        self._plot_artifact_rejection_case_study()

    def _plot_sampling_rate_stability(self):
        if not hasattr(self, 'data_quality_results'):
            return
        quality_df = self.data_quality_results['quality_df']
        plot_data = quality_df[quality_df['hz'] > 0]

        plt.figure(figsize=(12, 6))
        sns.histplot(plot_data['hz'], bins=30, kde=True, color='purple', alpha=0.7)
        plt.axvline(x=10, color='green', linestyle='--', linewidth=1.5, label='Target (10Hz)')
        plt.title('Sampling Rate Stability Histogram', fontsize=14, fontweight='bold')
        plt.xlabel('Samples per Second (Hz)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.savepath, f"{self.saveFileName}_sampling_rate_stability.png"), dpi=600)
        plt.close()

    def _plot_data_completeness_heatmap(self):
        if not hasattr(self, 'data_quality_results'):
            return
        quality_df = self.data_quality_results['quality_df'].copy()
        quality_df['date'] = quality_df.index.date
        quality_df['hour'] = quality_df.index.hour
        daily_hourly = quality_df.groupby(['date', 'hour']).size().unstack(fill_value=0)
        completeness = daily_hourly / 3600 * 100
        completeness = completeness[completeness.sum(axis=1) > 0]
        if completeness.empty:
            return
        plt.figure(figsize=(14, max(len(completeness) * 0.8, 4)))
        sns.heatmap(completeness, cmap='YlGnBu', annot=True, fmt='.1f',
                    cbar_kws={'label': 'Completeness (%)'})
        plt.title('Data Completeness Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Date', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.savepath, f"{self.saveFileName}_data_completeness_heatmap.png"), dpi=600)
        plt.close()

    def _calibrate_posture_zero(self):
        """
        A8：取夜间静息段（23:00–04:00）的 x/y/z 中位数作为该文件的"零位姿态"，
        写入 self.data_quality_results['posture_zero']。

        论文 Methods 段引用的"传感器朝向假设 X = 脊柱方向 / Y = 横向 / Z = 垂直"
        由这三个数支撑跨猪一致性验证。不强制坐标变换。
        """
        empty = {'x': float('nan'), 'y': float('nan'), 'z': float('nan'), 'n_samples': 0}
        df = getattr(self, 'moveData', None)
        if df is None or len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
            self.data_quality_results['posture_zero'] = empty
            return

        hours = df.index.hour
        night_mask = (hours >= 23) | (hours < 4)
        night_df = df.loc[night_mask, ['x', 'y', 'z']]
        if len(night_df) == 0:
            self.data_quality_results['posture_zero'] = empty
            return

        self.data_quality_results['posture_zero'] = {
            'x': float(night_df['x'].median()),
            'y': float(night_df['y'].median()),
            'z': float(night_df['z'].median()),
            'n_samples': int(len(night_df)),
        }

    def _plot_artifact_rejection_case_study(self):
        df = self.moveCleanData
        artifact_indices = df[df['Gyro_Status'] == 'Artifact'].index
        if len(artifact_indices) == 0:
            return
        artifact_time = artifact_indices[0]
        window_start = artifact_time - pd.Timedelta(seconds=10)
        window_end = artifact_time + pd.Timedelta(seconds=10)
        window_data = df.loc[window_start:window_end].copy()

        plt.figure(figsize=(14, 6))
        plt.plot(window_data.index, window_data['Gyro_Mag'], 'r-', linewidth=2, label='Before Cleaning')
        cleaned_gyro = window_data['Gyro_Mag'].copy()
        cleaned_gyro[window_data['Gyro_Status'] == 'Artifact'] = 0
        plt.plot(window_data.index, cleaned_gyro, 'g-', linewidth=2, label='After Cleaning')
        artifact_in_window = window_data[window_data['Gyro_Status'] == 'Artifact']
        if not artifact_in_window.empty:
            plt.scatter(artifact_in_window.index, artifact_in_window['Gyro_Mag'],
                        color='red', s=100, marker='x', label='Artifact')
        plt.title('Artifact Rejection Case Study', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Angular Velocity (°/s)', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.savepath, f"{self.saveFileName}_artifact_rejection_case.png"), dpi=600)
        plt.close()

    def generate_report(self, report_path=None):
        if report_path is None:
            os.makedirs(self.savepath, exist_ok=True)
            report_path = os.path.join(self.savepath, f"{self.saveFileName}_report.md")

        markdown_content = f"""# 运动量数据分析报告 (3轴角度传感器)

## 基本信息
- 分析文件: {self.saveFileName}
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        if hasattr(self, 'data_quality_results'):
            results = self.data_quality_results
            markdown_content += f"""## 1. 数据清洗结果
- 原始数据行数: {results['remaining_count'] + results['deleted_count']}
- 删除无效数据: {results['deleted_count']} 行
- 剩余有效数据: {results['remaining_count']} 行

"""
            gaps = results['gaps']
            if len(gaps) > 0:
                markdown_content += f"""## 2. 数据连续性分析
发现 {len(gaps)} 处明显的数据断裂：

| 断裂开始时间 | 数据恢复时间 | 中断时长(秒) |
|------------|------------|------------|
"""
                for end_time, row in gaps.iterrows():
                    duration = row['dt']
                    start_time = end_time - pd.Timedelta(seconds=duration)
                    markdown_content += f"| {str(start_time)} | {str(end_time)} | {duration:.2f} |\n"
            else:
                markdown_content += "## 2. 数据连续性分析\n未发现超过 3 秒的断裂。\n\n"

            quality_df = results['quality_df']
            markdown_content += "## 3. 采样率质量分析\n\n### 整体统计\n"
            for status, count in quality_df['status'].value_counts().items():
                markdown_content += f"- {status}: {count} 秒\n"
            markdown_content += "\n"

        if hasattr(self, 'angle_results'):
            results = self.angle_results
            markdown_content += f"""## 4. 角速度分析

### 异常检测
- 真实剧烈运动点: {results['n_real']} 个
- 伪影/噪点: {results['n_artifact']} 个

### 判定标准
- 疑似异常高值线: > {results['SUSPICIOUS_LIMIT']:.2f} °/s
- 邻居活跃达标线: > {results['ACTIVE_BASE']:.2f} °/s

"""

        markdown_content += "## 5. 结论\n\n分析完成。\n"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        if MoveData.is_english:
            print(f"Report generated: {report_path}")
        else:
            print(f"报告已生成: {report_path}")
        return report_path

    def zeroclan(self, rawdata, namelist):
        """
        A7：联合判 0 + 连续长度 ≥ 3 才视为缺失再插值。

        旧版：每列单独判 == 0 → 站立时真实 0° 被误删。
        新版：仅当 namelist 中所有列同时为 0，且连续行数 ≥ 3 时，
              才用左右邻居均值 (prev+next)/2 填入；孤立 0、单轴 0 不动。
        计数累加到 self._n_zeroclan_imputed_total（供 dataClean 写入 quality_results）。
        """
        cols = rawdata[namelist]
        all_zero = (cols == 0).all(axis=1)

        if all_zero.any():
            diff_mask = all_zero.ne(all_zero.shift())
            run_id = diff_mask.cumsum()
            run_sizes = all_zero.groupby(run_id).transform('size')
            is_real_missing = all_zero & (run_sizes >= 3)
        else:
            is_real_missing = pd.Series(False, index=rawdata.index)

        n_imputed = int(is_real_missing.sum())
        if hasattr(self, '_n_zeroclan_imputed_total'):
            self._n_zeroclan_imputed_total += n_imputed

        if n_imputed == 0:
            return rawdata

        for name in namelist:
            col = rawdata[name].copy()
            prev_val = col.shift(1)
            next_val = col.shift(-1)
            avg = (prev_val + next_val) / 2.0
            col = col.where(~is_real_missing, avg)
            # boundary：第一行/最后一行没有完整邻居
            if is_real_missing.iloc[0] and not pd.isna(next_val.iloc[0]):
                col.iat[0] = next_val.iloc[0]
            if is_real_missing.iloc[-1] and not pd.isna(prev_val.iloc[-1]):
                col.iat[-1] = prev_val.iloc[-1]
            rawdata.loc[:, name] = col

        return rawdata


if __name__ == "__main__":
    data = MoveData(
        movefilepath=r"F:\your_angle_data_path",
        picturepath=r"C:\Users\zengz\Desktop\output_plots",
        savepath=r"C:\Users\zengz\Desktop\output\result.csv",
        is_english=True
    )
    data.saveData(r"C:\Users\zengz\Desktop\output\result.csv")
    data.generate_report()
    data.drawPicture()
