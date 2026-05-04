import pandas as pd
from pathlib import Path
import numpy as np
import os
from datetime import datetime, timedelta
# 获取系统当前时区（例如 Asia/Shanghai）
from math import dist
from tzlocal import get_localzone
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 尝试多种字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def get_true_angle_diff(series):
    diff = series.diff()
    # 向量化修正：跨越 +/- 180 度边界的情况
    diff = np.where(diff > 180, diff - 360, diff)
    diff = np.where(diff < -180, diff + 360, diff)
    return diff

def parse_time(t_str):
    """将HH:MM:SS.fff转换为总秒数（保留原始精度）"""
    # h, m, s = t_str.split(':')
    # seconds = int(h)*3600 + int(m)*60 + int(s)
    # return seconds
    try:
        h, m, s = t_str.split(':', 2)  # 最多分割两次，确保第三个部分包含剩余内容
        s_parts = s.split('.', 1)      # 分割秒和毫秒

        # 转换各部分为整数
        hours = int(h)
        minutes = int(m)
        seconds = int(s_parts[0])
        milliseconds = int(s_parts[1]) if len(s_parts) > 1 else 0

        # 计算总秒数
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    except (ValueError, AttributeError, IndexError):
        return None
class MoveDataNew:
    # Global class variable to control output language
    is_english = False
    
    def __init__(self, movefilepath,picturepath,savepath,onedir=False,is_english=False):

        self.oneDir=onedir
        self.is_english=is_english
        # 更新类变量，确保所有实例使用相同的语言设置
        MoveDataNew.is_english = is_english
        self.movefilepath = movefilepath
        self.picturepath=picturepath
        
        # 确保 savepath 和 saveFileName 总是被设置
        if savepath:
            self.savepath = os.path.dirname(savepath)
            full_filename = os.path.basename(savepath)
            self.saveFileName = os.path.splitext(full_filename)[0]  # 方法一：使用os.path
        else:
            # 如果没有提供 savepath，使用当前目录
            self.savepath = os.getcwd()
            self.saveFileName = "movement_analysis"
        
        self.getAllFile()
        self.getAllData()
        self.dataClean()
        self.calculateAngle()
        #self.dataClean()
    def getAllFile(self):
        # 创建空列表存储结果
        angle_files = {}
        # 递归遍历目
        if not self.oneDir:
            for path in Path(self.movefilepath).rglob('*'):
                if path.is_file() and "IMU" in path.name:
                    angle_files[path.name] = str(path.resolve())
        else:
            for path in Path(self.movefilepath).rglob('*'):
                if path.is_file() and "IMU" in path.name:
                    angle_files[path.name] = str(path.resolve())
        self.angle_files=angle_files
        print(self.angle_files)
    def readAngleTxt(self,filename,filepath):
        if("IMU" in filename):
            if("IMU" in filename):
                rawdata = pd.read_csv(filepath, sep=r',',skiprows=1, header=None, names=["timestamp", "x", "y", "z", "jx","jy","jz"])
            import re
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', filename.split(".")[0])
            if not date_match:
                raise ValueError(f"文件名中未找到有效日期: {filename}")
            date_str = date_match.group()
            year, month, day = map(int, date_str.split("-"))
            # 创建基础日期对象
            base_date = datetime(year, month, day)

            starttime=base_date
            # 向量化操作（比apply快10倍）
            total_seconds = pd.to_numeric(
                rawdata['timestamp'].apply(parse_time),
                errors='coerce'
            )
            rawdata['timedelta'] = pd.to_timedelta(total_seconds, unit='s')
            # 2. 自动检测跨天
            # 修复：考虑两种跨天情况
            # 1. 正常跨天（23:59:59 → 00:00:00）
            # 2. 长时间中断后恢复到第二天（如下午中断，第二天上午恢复）
            time_diff = rawdata['timedelta'].diff()
            
            # 提取当前时间的小时和分钟，转换为总秒数来判断是否在凌晨
            current_time_seconds = rawdata['timedelta'].dt.total_seconds()
            # 凌晨00:00:00到00:30:00对应0到1800秒
            is_early_morning = current_time_seconds <= 1800
            
            # 条件1：时间回退幅度足够大（超过30分钟），排除小幅乱序
            is_large_jump = time_diff < pd.Timedelta(minutes=-30)
            
            # 条件2：前一时刻在深夜（23:30:00之后），当前时刻在凌晨
            # 23:30:00 = 23*3600 + 30*60 = 84600秒
            prev_time_seconds = current_time_seconds.shift(1)
            is_prev_late_night = prev_time_seconds >= 84600  # 23:30:00之后
            
            # 条件3：时间回退幅度非常大（超过4小时），处理长时间中断的情况
            is_very_large_jump = time_diff < pd.Timedelta(hours=-2)
            
            # 综合判断：
            # 情况1：正常跨天（时间大幅回退 + 当前在凌晨 + 前一时刻在深夜）
            # 情况2：长时间中断（时间回退超过4小时，不管时间点）
            # 两种情况都视为跨天
            is_first_row = pd.Series([True] + [False] * (len(rawdata) - 1), index=rawdata.index)
            normal_cross_day = is_large_jump & is_early_morning & (is_prev_late_night | is_first_row)
            long_break_cross_day = is_very_large_jump
            cross_day_mask = normal_cross_day | long_break_cross_day
            cross_day = cross_day_mask.cumsum()
            # 3. 生成最终时间戳（处理跨天）
            rawdata['timestamp'] = starttime + rawdata['timedelta'] + pd.to_timedelta(cross_day, unit='D')
            try:
                local_timezone = get_localzone()
            except Exception:
                local_timezone = pytz.timezone('Asia/Shanghai')  # 自动获取系统时区需用第三方库，此处需手动指定示例
                if MoveDataNew.is_english:
                    print("Warning: Using default timezone 'Asia/Shanghai'")
                else:
                    print("Warning: 使用默认时区 'Asia/Shanghai'")
            rawdata['timestamp'] = (
                rawdata['timestamp']
                .dt.tz_localize(local_timezone)  # 自动附加本地时区
                .dt.tz_localize(None) #清楚时区信息，转换为本地时间
            )
            rawdata.drop(columns=rawdata.columns[-1], inplace=True)
            rawdata.set_index('timestamp', inplace=True)
            return rawdata
    def getAllData(self):
        dflist=[]
        total_files = len(self.angle_files)
        if MoveDataNew.is_english:
            print(f"Processing {total_files} files...")
        else:
            print(f"正在处理 {total_files} 个文件...")
        
        for i, file in enumerate(self.angle_files):
            if MoveDataNew.is_english:
                print(f"Processing file {i+1}/{total_files}: {file}")
            else:
                print(f"正在处理文件 {i+1}/{total_files}：{file}")
            df=self.readAngleTxt(file,self.angle_files[file])
            dflist.append(df)
        
        if MoveDataNew.is_english:
            print("Combining all data...")
        else:
            print("正在合并所有数据...")
        
        combined_df = pd.concat(dflist)
        # 按索引（时间戳）排序
        if MoveDataNew.is_english:
            print("Sorting data by timestamp...")
        else:
            print("正在按时间戳排序数据...")
        sorted_df = combined_df.sort_index()
        self.moveData=sorted_df
    def dataClean(self):
        # ==========================================
        # 1. 加载数据与预处理
        # ==========================================
        if MoveDataNew.is_english:
            print(f"{'=' * 20} 1. Loading Data {'=' * 20}")
        else:
            print(f"{'=' * 20} 1. 加载数据 {'=' * 20}")
        # 请确保文件路径正确
        # 假设文件名为 aaaall.csv
        df = self.moveData.copy()
        original_count = len(df)
        # 只保留索引不是 NaT (notnull) 的行
        df = df[df.index.notnull()]
        # 打印清理结果

        # 重命名列
        column_mapping = {'x': 'AccX', 'y': 'AccY', 'z': 'AccZ', 'jx': 'AngX', 'jy': 'AngY', 'jz': 'AngZ'}
        df.rename(columns=column_mapping, inplace=True)
        df.sort_index(inplace=True)  # 确保时间有序
        target_cols = ['AccX', 'AccY', 'AccZ', 'AngX', 'AngY', 'AngZ']
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=target_cols, how='any', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        deleted_count = original_count - len(df)
        if MoveDataNew.is_english:
            print(f"Data cleaning completed: Deleted {deleted_count} rows with invalid timestamps.")
            print(f"Remaining valid data: {len(df)} rows.")
        else:
            print(f"数据清洗完成：删除了 {deleted_count} 行无效时间戳数据。")
            print(f"剩余有效数据：{len(df)} 行。")

        # ==========================================
        # 2. 计算时间差与断点检测
        # ==========================================
        if MoveDataNew.is_english:
            print(f"\n{'=' * 20} 2. Data Continuity Detection (Breakpoint > 3s) {'=' * 20}")
        else:
            print(f"\n{'=' * 20} 2. 数据连续性检测 (断点 > 3秒) {'=' * 20}")

        # 计算相邻数据的时间差 (秒)
        df['dt'] = df.index.to_series().diff().dt.total_seconds()

        # 筛选断点 (时间差 > 3秒)
        gap_threshold = 3.0
        gaps = df[df['dt'] > gap_threshold].copy()

        if len(gaps) > 0:
            if MoveDataNew.is_english:
                print(f"Found {len(gaps)} significant data breaks:")
                print(f"{'Break Start Time':<25} | {'Data Resume Time':<25} | {'Break Duration (s)'}")
                print("-" * 70)
            else:
                print(f"发现 {len(gaps)} 处明显的数据断裂：")
                print(f"{'断裂开始时间':<25} | {'数据恢复时间':<25} | {'中断时长(秒)'}")
                print("-" * 70)

            for end_time, row in gaps.iterrows():
                duration = row['dt']
                # 开始时间 = 恢复时间 - 持续时长
                start_time = end_time - pd.Timedelta(seconds=duration)
                if MoveDataNew.is_english:
                    print(f"{str(start_time):<25} | {str(end_time):<25} | {duration:.2f} s")
                else:
                    print(f"{str(start_time):<25} | {str(end_time):<25} | {duration:.2f} s")
        else:
            if MoveDataNew.is_english:
                print("No breaks exceeding 3 seconds found.")
            else:
                print("未发现超过 3 秒的断裂。")
        hz_per_second = df.resample('1S').size()

        # 创建一个分析用的 DataFrame
        quality_df = pd.DataFrame({'hz': hz_per_second})

        # 标记异常类型
        # 设定阈值：正常范围定为 8-12 Hz
        # 低于 5 Hz 视为“数据过少/丢包” (不包含0，0是断连)
        # 高于 15 Hz 视为“数据过多/堆积”
        LOW_THRESH = 5
        HIGH_THRESH = 15

        quality_df['status'] = 'Normal'
        quality_df.loc[quality_df['hz'] == 0, 'status'] = 'Missing (0 Hz)'
        quality_df.loc[(quality_df['hz'] > 0) & (quality_df['hz'] < LOW_THRESH), 'status'] = 'Too Low (<5 Hz)'
        quality_df.loc[quality_df['hz'] > HIGH_THRESH, 'status'] = 'Too High (>15 Hz)'

        # ==========================================
        # 3. 统计异常时段（精确到小时）
        # ==========================================
        # 提取日期和小时信息用于分组
        quality_df['date_hour'] = quality_df.index.floor('H')

        # 统计每个小时内，各种异常状态出现的秒数
        hourly_report = quality_df.groupby(['date_hour', 'status']).size().unstack(fill_value=0)

        # 筛选出有问题的时段（即“过高”或“过少”不为0的时段）
        # 我们主要关注 'Too High' 和 'Too Low'
        problematic_hours = hourly_report[
            (hourly_report.get('Too High (>15 Hz)', 0) > 0) |
            (hourly_report.get('Too Low (<5 Hz)', 0) > 0)
            ].copy()

        # 按问题严重程度排序（数据过多的秒数 + 数据过少的秒数）
        problematic_hours['total_abnormal_seconds'] = (
                problematic_hours.get('Too High (>15 Hz)', 0) +
                problematic_hours.get('Too Low (<5 Hz)', 0)
        )
        problematic_hours = problematic_hours.sort_values('total_abnormal_seconds', ascending=False)

        # ==========================================
        # 4. 输出报告
        # ==========================================
        if MoveDataNew.is_english:
            print("\n====== Data Quality Overview (Based on samples per second) ======")
        else:
            print("\n====== 数据质量概览 (基于每秒采样数) ======")
        print(quality_df['status'].value_counts())

        if MoveDataNew.is_english:
            print("\n====== Top 10 Time Periods with Most Severe Sampling Rate Abnormalities ======")
        else:
            print("\n====== 采样率异常最为严重的 10 个小时段 ======")
        # 格式化输出
        if not problematic_hours.empty:
            if MoveDataNew.is_english:
                print(f"{'Time Period (Day Hour)':<25} | {'Too High (seconds)':<10} | {'Too Low (seconds)':<10} | {'Missing (seconds)':<10}")
            else:
                print(f"{'时间段 (Day Hour)':<25} | {'过高(秒数)':<10} | {'过低(秒数)':<10} | {'缺失(秒数)':<10}")
            print("-" * 65)
            for index, row in problematic_hours.head(10).iterrows():
                time_str = str(index)
                high_cnt = row.get('Too High (>15 Hz)', 0)
                low_cnt = row.get('Too Low (<5 Hz)', 0)
                miss_cnt = row.get('Missing (0 Hz)', 0)
                if MoveDataNew.is_english:
                    print(f"{time_str:<25} | {high_cnt:<10} | {low_cnt:<10} | {miss_cnt:<10}")
                else:
                    print(f"{time_str:<25} | {high_cnt:<10} | {low_cnt:<10} | {miss_cnt:<10}")
        else:
            if MoveDataNew.is_english:
                print("Congratulations! No obvious sampling rate abnormal periods found.")
            else:
                print("恭喜！未发现明显的采样率异常时段。")

        # ==========================================
        # 5. 可视化分析
        # ==========================================
        plt.figure(figsize=(16, 10))

        # 图1: 全局采样率散点图
        plt.subplot(2, 1, 1)
        # 过滤掉0值以便看清波动，且只画出一部分点避免卡顿
        plot_data = quality_df[quality_df['hz'] > 0]
        # 改进散点图：使用不同颜色区分不同范围的值
        colors = []
        for hz in plot_data['hz']:
            if hz > HIGH_THRESH:
                colors.append('red')
            elif hz < LOW_THRESH:
                colors.append('orange')
            else:
                colors.append('blue')
        scatter = plt.scatter(plot_data.index, plot_data['hz'], s=2, alpha=0.6, c=colors)
        # 添加图例
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Too High'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Too Low')
        ]
        plt.legend(handles=custom_lines, loc='upper right')
        plt.axhline(y=10, color='green', linestyle='-', linewidth=1, label='Target (10Hz)')
        plt.axhline(y=HIGH_THRESH, color='red', linestyle='--', linewidth=0.8, label='High Threshold')
        plt.axhline(y=LOW_THRESH, color='orange', linestyle='--', linewidth=0.8, label='Low Threshold')
        if self.is_english:
            plt.title('Sample Rate Fluctuation Over Time', fontsize=14, fontweight='bold')
        else:
            plt.title('全时段采样率波动图', fontsize=14, fontweight='bold')
        plt.ylabel('Samples per Second (Hz)', fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')

        # 图2: 采样率直方图分布
        plt.subplot(2, 1, 2)
        # 只看非零数据
        # 改进直方图：使用更美观的样式
        import numpy as np
        bin_edges = np.arange(0, 31, 1)
        sns.histplot(plot_data['hz'], bins=bin_edges, kde=True, color='purple', alpha=0.7)
        plt.axvline(x=10, color='green', linestyle='--', linewidth=1.5, label='Target (10Hz)')
        plt.axvline(x=HIGH_THRESH, color='red', linestyle='--', linewidth=1, label='High Threshold')
        plt.axvline(x=LOW_THRESH, color='orange', linestyle='--', linewidth=1, label='Low Threshold')
        if self.is_english:
            plt.title('Sampling Rate Distribution Histogram', fontsize=14, fontweight='bold')
            plt.xlabel('Samples per Second (Hz)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
        else:
            plt.title('采样率分布直方图', fontsize=14, fontweight='bold')
            plt.xlabel('Hz', fontsize=12)
            plt.ylabel('数量', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        # 设置 dpi 为 600
        if self.is_english:
            plt.savefig(os.path.join(self.savepath, self.saveFileName + "_sampling_rate_distribution.png"), dpi=600)
        else:
            plt.savefig(os.path.join(self.savepath, self.saveFileName + "采样评率分布图.png"), dpi=600)
        
        # 保存数据质量分析结果，用于生成报告
        self.data_quality_results = {
            'deleted_count': deleted_count,
            'remaining_count': len(df),
            'gaps': gaps,
            'quality_df': quality_df,
            'problematic_hours': problematic_hours,
            'hourly_report': hourly_report,
            'low_thresh': LOW_THRESH,
            'high_thresh': HIGH_THRESH
        }
        
        self.moveData=df
    def calculateAngle(self):
        df=self.moveData
        # 筛选断点 (时间差 > 3秒)
        gap_threshold = 3.0
        df['d_AngX'] = get_true_angle_diff(df['AngX'])
        df['d_AngY'] = get_true_angle_diff(df['AngY'])
        df['d_AngZ'] = get_true_angle_diff(df['AngZ'])

        # 3.2 计算合成角速度 (Gyro_Mag)
        # 先全部计算，不管 dt 大小
        df['Gyro_Mag'] = np.sqrt(df['d_AngX'] ** 2 + df['d_AngY'] ** 2 + df['d_AngZ'] ** 2) / df['dt']

        # 3.3 数据清洗 (Data Validating)
        # 对于断点处（dt > 3.0）或 重复时间戳（dt <= 0），其计算出的速度无物理意义
        # 强制归零，防止产生假的高频信号
        invalid_mask = (df['dt'] > gap_threshold) | (df['dt'] <= 0.001) | (df['dt'].isna())
        df.loc[invalid_mask, 'Gyro_Mag'] = 0.0

        if MoveDataNew.is_english:
            print("Angular velocity calculation completed. Auto-corrected velocity values at breakpoints.")
        else:
            print("角速度计算完成。已自动修正断点处的速度值。")

        # ==========================================
        # 4. 统计分析与阈值自适应
        # ==========================================
        if MoveDataNew.is_english:
            print(f"\n{'=' * 20} 4. Statistical Analysis and Threshold Setting {'=' * 20}")
        else:
            print(f"\n{'=' * 20} 4. 统计分析与阈值设定 {'=' * 20}")

        # 只统计非零的活跃数据
        active_gyro = df[df['Gyro_Mag'] > 1.0]['Gyro_Mag']
        stats = active_gyro.describe(percentiles=[0.5, 0.75, 0.99, 0.999])

        # 【自适应阈值】
        ACTIVE_BASE = stats['75%']
        SUSPICIOUS_LIMIT = max(stats['99.9%'], 300.0)  # 至少300，或者更高

        if MoveDataNew.is_english:
            print("Angular velocity distribution characteristics (only counting active points):")
            print(f"  - Median (P50): {stats['50%']:.2f} °/s")
            print(f"  - Active line (P75): {stats['75%']:.2f} °/s")
            print(f"  - Extreme line (P99.9): {stats['99.9%']:.2f} °/s")
            print(f"\n[Set Judgment Standards]")
            print(f"  1. Suspected abnormal high value line: > {SUSPICIOUS_LIMIT:.2f} °/s")
            print(f"  2. Neighbor active threshold: > {ACTIVE_BASE:.2f} °/s")
        else:
            print("角速度分布特征 (仅统计活跃点):")
            print(f"  - 中位数 (P50): {stats['50%']:.2f} °/s")
            print(f"  - 活跃线 (P75): {stats['75%']:.2f} °/s")
            print(f"  - 极值线 (P99.9): {stats['99.9%']:.2f} °/s")
            print(f"\n[设定判定标准]")
            print(f"  1. 疑似异常高值线: > {SUSPICIOUS_LIMIT:.2f} °/s")
            print(f"  2. 邻居活跃达标线: > {ACTIVE_BASE:.2f} °/s")

        # ==========================================
        # 5. 上下文异常检测
        # ==========================================
        if MoveDataNew.is_english:
            print(f"\n{'=' * 20} 5. Anomaly Classification Diagnosis {'=' * 20}")
        else:
            print(f"\n{'=' * 20} 5. 异常分类诊断 {'=' * 20}")

        # 获取前后邻居的值
        df['Prev_Gyro'] = df['Gyro_Mag'].shift(1).fillna(0)
        df['Next_Gyro'] = df['Gyro_Mag'].shift(-1).fillna(0)

        # 计算邻居的最大值
        df['Neighbor_Max'] = df[['Prev_Gyro', 'Next_Gyro']].max(axis=1)

        # 计算孤立比率 (当前值 / 邻居最大值)
        df['Isolation_Ratio'] = df['Gyro_Mag'] / (df['Neighbor_Max'] + 1.0)

        # --- 判定逻辑 ---

        # 逻辑 A: 真实剧烈运动 (Real High Activity)
        # 条件：(当前值 > 高值线) 且 (邻居也活跃 OR 并没有高出邻居太多倍)
        mask_real = (
                (df['Gyro_Mag'] > SUSPICIOUS_LIMIT) &
                ((df['Neighbor_Max'] > ACTIVE_BASE) | (df['Isolation_Ratio'] < 10.0))
        )

        # 逻辑 B: 伪影/噪点 (Artifact)
        # 条件：(当前值 > 高值线) 且 (邻居很弱 且 突变倍数很高)
        mask_artifact = (
                (df['Gyro_Mag'] > SUSPICIOUS_LIMIT) &
                (df['Neighbor_Max'] < ACTIVE_BASE) &
                (df['Isolation_Ratio'] > 10.0)
        )

        # 标记结果列
        df['Gyro_Status'] = 'Normal'
        df.loc[mask_real, 'Gyro_Status'] = 'Real_Activity'
        df.loc[mask_artifact, 'Gyro_Status'] = 'Artifact'

        n_real = mask_real.sum()
        n_artifact = mask_artifact.sum()

        if MoveDataNew.is_english:
            print(f"Detection results:")
            print(f"  - Real intense movement points: {n_real} (retained)")
            print(f"  - Artifacts/Noise:     {n_artifact} (suggested to remove when calculating VeDBA)")
        else:
            print(f"检测结果：")
            print(f"  - 真实剧烈运动点: {n_real} 个 (予以保留)")
            print(f"  - 伪影/噪点:     {n_artifact} 个 (后续计算VeDBA时建议剔除)")

        if n_artifact > 0:
            if MoveDataNew.is_english:
                print("\n[Artifact Examples] (Extremely high value but front and back are stationary):")
            else:
                print("\n[伪影示例] (数值极大但前后静止):")
            print(df[mask_artifact][['Gyro_Mag', 'Neighbor_Max', 'Isolation_Ratio']].head(3))

        if n_real > 0:
            if MoveDataNew.is_english:
                print("\n[Real Movement Examples] (High value and front/back have movement):")
            else:
                print("\n[真实运动示例] (数值大且前后有动作):")
            print(df[mask_real][['Gyro_Mag', 'Neighbor_Max', 'Isolation_Ratio']].head(3))
        if MoveDataNew.is_english:
            print(f"\n{'=' * 20} 6. Acceleration Processing and VeDBA Calculation {'=' * 20}")
        else:
            print(f"\n{'=' * 20} 6. 加速度处理与 VeDBA 计算 {'=' * 20}")

        # 6.1 计算合加速度 (用于检测失重/硬件错误)
        df['Acc_Mag'] = np.sqrt(df['AccX'] ** 2 + df['AccY'] ** 2 + df['AccZ'] ** 2)
        """# 1. 剔除索引中无效的时间戳 (NaT)
        df = df[df.index.notnull()]

        # 2. 确保索引是严格按时间排序的 (rolling的时间窗口依赖顺序)
        df.sort_index(inplace=True)

        # 3. 再次检查是否有重复时间戳 (如果有完全重复的时间点，rolling可能会困惑，建议去重)
        df = df[~df.index.duplicated(keep='first')]"""
        # 6.2 分离重力 (Static) 与 动态 (Dynamic)
        # 使用 '2s' 时间窗口，center=True 保证不发生相位滞后
        # 这种写法会自动处理时间断裂：如果断连超过2s，rolling不会强行计算
        df['StaticX'] = df['AccX'].rolling(window='2s', center=True).mean()
        df['StaticY'] = df['AccY'].rolling(window='2s', center=True).mean()
        df['StaticZ'] = df['AccZ'].rolling(window='2s', center=True).mean()

        # 填充因滚动产生的边缘NaN (前后填充)
        df[['StaticX', 'StaticY', 'StaticZ']] = df[['StaticX', 'StaticY', 'StaticZ']].bfill().ffill()

        # 计算动态加速度
        df['DynX'] = df['AccX'] - df['StaticX']
        df['DynY'] = df['AccY'] - df['StaticY']
        df['DynZ'] = df['AccZ'] - df['StaticZ']

        # 计算原始 VeDBA
        df['VeDBA'] = np.sqrt(df['DynX'] ** 2 + df['DynY'] ** 2 + df['DynZ'] ** 2)

        # ==========================================
        # 7. 基于统计与角速度的深度清洗
        # ==========================================
        if MoveDataNew.is_english:
            print(f"\n{'=' * 20} 7. Deep Data Cleaning {'=' * 20}")
        else:
            print(f"\n{'=' * 20} 7. 深度数据清洗 {'=' * 20}")

        # 7.1 统计分布以确定噪声基底
        # 选取 "正常状态" 下的数据 (排除掉刚才识别的伪影点)
        valid_acc_data = df[df['Gyro_Status'] != 'Artifact']['VeDBA']
        acc_stats = valid_acc_data.describe(percentiles=[0.5, 0.999])

        # 自适应阈值设定
        # 噪声底噪：中位数的 1.5 倍，且至少 0.01g
        NOISE_FLOOR = max(acc_stats['50%'] * 1.5, 0.01)
        # 生物极限：P99.9 或 3.0g，防止撞击产生离谱积分
        IMPACT_CEILING = min(acc_stats['99.9%'], 3.0)

        if MoveDataNew.is_english:
            print(f"[Adaptive Thresholds]")
            print(f"  - Ignore micro-motion noise: < {NOISE_FLOOR:.4f} g")
            print(f"  - Impact clipping upper limit: > {IMPACT_CEILING:.4f} g")
        else:
            print(f"[自适应阈值]")
            print(f"  - 忽略微动噪声: < {NOISE_FLOOR:.4f} g")
            print(f"  - 撞击削峰上限: > {IMPACT_CEILING:.4f} g")

        # --- 清洗步骤 ---
        df['VeDBA_Clean'] = df['VeDBA'].copy()

        # A. 联动清洗：如果角速度被判定为"伪影"，则该时刻的加速度也视为无效震动
        artifact_count = (df['Gyro_Status'] == 'Artifact').sum()
        df.loc[df['Gyro_Status'] == 'Artifact', 'VeDBA_Clean'] = 0
        if MoveDataNew.is_english:
            print(f"  -> Removed {artifact_count} angular velocity artifact point acceleration values.")
        else:
            print(f"  -> 已剔除 {artifact_count} 个角速度伪影点的加速度值。")

        # B. 失重清洗：如果合加速度 < 0.2g (传感器失效/掉包)，归零
        freefall_mask = df['Acc_Mag'] < 0.2
        df.loc[freefall_mask, 'VeDBA_Clean'] = 0
        if MoveDataNew.is_english:
            print(f"  -> Removed {freefall_mask.sum()} suspected weightlessness/failure points.")
        else:
            print(f"  -> 已剔除 {freefall_mask.sum()} 个疑似失重/失效点。")

        # C. 噪声归零：去除无意义的微颤
        df.loc[df['VeDBA_Clean'] < NOISE_FLOOR, 'VeDBA_Clean'] = 0

        # D. 撞击削峰：保留运动事件，但限制数值大小
        #    (例如：撞笼子产生 5g，记为 3g，算作剧烈运动，而不是删掉)
        df.loc[df['VeDBA_Clean'] > IMPACT_CEILING, 'VeDBA_Clean'] = IMPACT_CEILING
        
        # 保存角度和加速度分析结果，用于生成报告
        self.angle_accel_results = {
            'n_real': n_real,
            'n_artifact': n_artifact,
            'artifact_examples': df[mask_artifact][['Gyro_Mag', 'Neighbor_Max', 'Isolation_Ratio']].head(3),
            'real_examples': df[mask_real][['Gyro_Mag', 'Neighbor_Max', 'Isolation_Ratio']].head(3),
            'ACTIVE_BASE': ACTIVE_BASE,
            'SUSPICIOUS_LIMIT': SUSPICIOUS_LIMIT,
            'NOISE_FLOOR': NOISE_FLOOR,
            'IMPACT_CEILING': IMPACT_CEILING,
            'artifact_count': artifact_count,
            'freefall_count': freefall_mask.sum()
        }
        
        # 保存清洗后的数据，包含 Gyro_Status 列
        self.moveCleanData=df
        self.generate_secondly_summary()
        
        # 生成高级可视化图表
        self.generate_advanced_visualizations()
    def calculate_spectral_entropy(self, signal):
        """
        计算信号的频谱熵
        """
        if len(signal) < 3:
            return 0
        # 计算FFT
        fft_result = np.fft.fft(signal)
        # 计算功率谱
        power_spectrum = np.abs(fft_result) ** 2
        # 归一化
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        # 计算熵
        entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))
        return entropy
    
    def calculate_zero_crossing_rate(self, signal):
        """
        计算信号的过零率
        """
        if len(signal) < 2:
            return 0
        # 计算符号变化次数
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal)))) / 2
        return zero_crossings
    
    def calculate_dominant_frequency(self, signal, fs=10):
        """
        计算信号的主频强度
        fs: 采样频率
        """
        if len(signal) < 3:
            return 0
        # 计算FFT
        fft_result = np.fft.fft(signal)
        # 计算功率谱
        power_spectrum = np.abs(fft_result) ** 2
        # 计算频率轴
        freqs = np.fft.fftfreq(len(signal), 1/fs)
        # 只考虑正频率
        positive_freqs = freqs[freqs > 0]
        positive_power = power_spectrum[freqs > 0]
        if len(positive_power) == 0:
            return 0
        # 找到最大功率对应的频率
        dominant_freq = positive_freqs[np.argmax(positive_power)]
        return dominant_freq
    
    def calculate_cv(self, signal):
        """
        计算信号的变异系数
        """
        if len(signal) == 0:
            return 0
        mean = np.mean(signal)
        if mean == 0:
            return 0
        std = np.std(signal)
        cv = std / mean
        return cv

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

    def calculate_sample_entropy(self, signal, m=2, r_ratio=0.2):
        """样本熵：衡量时间序列的复杂度/不可预测性"""
        n = len(signal)
        if n < m + 2:
            return 0.0
        r = r_ratio * np.std(signal)
        if r == 0:
            return 0.0
        def _count_templates(tlen):
            cnt = 0
            for i in range(n - tlen):
                for j in range(i + 1, n - tlen + 1):
                    if np.max(np.abs(signal[i:i+tlen] - signal[j:j+tlen])) < r:
                        cnt += 1
            return cnt
        B = _count_templates(m)
        A = _count_templates(m + 1)
        if B == 0 or A == 0:
            return 0.0
        return float(-np.log(A / B))

    def calculate_band_powers(self, signal, fs=10):
        """FFT频段功率比：Low(0-1Hz), Loco(1-4Hz), High(4-10Hz)"""
        n = len(signal)
        if n < 3:
            return 0.0, 0.0, 0.0
        fft_vals = np.fft.fft(signal)
        power    = np.abs(fft_vals) ** 2
        freqs    = np.fft.fftfreq(n, 1.0 / fs)
        pos      = freqs > 0
        fp, pp   = freqs[pos], power[pos]
        total    = np.sum(pp)
        if total == 0:
            return 0.0, 0.0, 0.0
        p_low  = float(np.sum(pp[fp < 1.0])                        / total)
        p_loco = float(np.sum(pp[(fp >= 1.0) & (fp < 4.0)])        / total)
        p_high = float(np.sum(pp[fp >= 4.0])                       / total)
        return p_low, p_loco, p_high

    def generate_secondly_summary(self):
        """
        生成每秒数据汇总（1Hz），含姿态、行为状态比例、频谱特征共29列
        """
        if MoveDataNew.is_english:
            print(f"\n{'='*20} Generating Per-Second Data Summary (1Hz) {'='*20}")
        else:
            print(f"\n{'='*20} 生成每秒数据汇总 (1Hz) {'='*20}")

        df = self.moveCleanData.copy()

        # ==========================================
        # 步骤一：预计算派生列（基于10Hz原始数据）
        # ==========================================
        # Roll：绕脊柱轴旋转角（侧卧核心指标），静止站立≈0°，侧躺趋近±90°
        df['Roll'] = np.degrees(np.arctan2(
            df['StaticY'],
            np.sqrt(df['StaticX']**2 + df['StaticZ']**2)
        ))
        # Pitch：前后倾斜角（低头采食/拱地时正向增大）
        df['Pitch'] = np.degrees(np.arctan2(df['StaticX'], np.abs(df['StaticZ'])))
        # Jerk：VeDBA_Clean逐点变化率，跨断点（dt>0.5s）置NaN避免伪峰
        jerk = df['VeDBA_Clean'].diff().abs()
        jerk[df['dt'] > 0.5] = np.nan
        df['Jerk'] = jerk
        # 各轴绝对角速度（方向无意义，取绝对值用于单轴均值特征）
        df['AngY_abs'] = df['d_AngY'].abs()
        df['abs_d_AngX'] = df['d_AngX'].abs()
        df['abs_d_AngZ'] = df['d_AngZ'].abs()
        # 倾斜幅度（Roll²+Pitch² 的平方根）
        df['Tilt_Mag'] = np.sqrt(df['Roll']**2 + df['Pitch']**2)
        # 各轴动态加速度绝对值（用于 ODBA）
        df['DynX_abs'] = df['DynX'].abs()
        df['DynY_abs'] = df['DynY'].abs()
        df['DynZ_abs'] = df['DynZ'].abs()

        # ==========================================
        # 步骤二：数据驱动阈值（自动计算，打印供记录）
        # ==========================================
        vedba_active = df['VeDBA_Clean'][df['VeDBA_Clean'] > 0]
        thresh_low  = vedba_active.quantile(0.33) if len(vedba_active) > 0 else 0.01
        thresh_high = vedba_active.quantile(0.75) if len(vedba_active) > 0 else 0.05

        thresh_lying = float(np.clip(df['Roll'].abs().quantile(0.90), 40.0, 80.0))

        pitch_positive = df['Pitch'][df['Pitch'] > 0]
        thresh_headdown = pitch_positive.quantile(0.75) if len(pitch_positive) > 0 else 10.0

        active_gyro = df[df['Gyro_Mag'] > 1.0]['Gyro_Mag']
        thresh_active = active_gyro.quantile(0.75) if len(active_gyro) > 0 else 10.0

        if MoveDataNew.is_english:
            print(f"[Data-driven Thresholds - Back Sensor]")
            print(f"  Activity - Rest/Low boundary     : <= {thresh_low:.4f} g  (active VeDBA P33)")
            print(f"  Activity - Low/High boundary     : <= {thresh_high:.4f} g  (active VeDBA P75)")
            print(f"  Posture  - Lying detection       : |Roll| > {thresh_lying:.1f}°  (|Roll| P90, clipped 40-80°)")
            print(f"  Posture  - Head-down detection   : Pitch  > {thresh_headdown:.1f}°  (positive Pitch P75)")
            print(f"  Active fraction base             : Gyro > {thresh_active:.2f} °/s")
        else:
            print(f"[数据驱动阈值 - 背部传感器]")
            print(f"  活动强度 - 静息/低强度分界 : <= {thresh_low:.4f} g  (活跃VeDBA的P33)")
            print(f"  活动强度 - 低强度/高强度分界: <= {thresh_high:.4f} g  (活跃VeDBA的P75)")
            print(f"  姿态     - 侧卧判断        : |Roll| > {thresh_lying:.1f}°  (|Roll|的P90，限制40-80°)")
            print(f"  姿态     - 低头采食判断    : Pitch  > {thresh_headdown:.1f}°  (正Pitch的P75)")
            print(f"  高活动基线               : Gyro > {thresh_active:.2f} °/s")

        # ==========================================
        # 步骤三：标记行为状态（10Hz点级别，用于后续求比例）
        # ==========================================
        df['is_rest']     = (df['VeDBA_Clean'] <= thresh_low).astype(float)
        df['is_moderate'] = ((df['VeDBA_Clean'] > thresh_low) & (df['VeDBA_Clean'] <= thresh_high)).astype(float)
        df['is_vigorous'] = (df['VeDBA_Clean'] > thresh_high).astype(float)
        df['is_lying']    = (df['Roll'].abs() > thresh_lying).astype(float)
        df['is_headdown'] = (df['Pitch'] > thresh_headdown).astype(float)
        df['is_active']   = (df['Gyro_Mag'] > thresh_active).astype(float)

        # ==========================================
        # 步骤四：向量化重采样聚合
        # ==========================================
        agg_logic = {
            'VeDBA_Clean': ['mean', 'sum', 'max', 'min', 'std'],
            'Gyro_Mag':    ['mean', 'max'],
            'AccX':        'count',
            'DynX':        ['mean', 'std'],
            'DynY':        ['mean', 'std'],
            'DynZ':        ['mean', 'std'],
            'Acc_Mag':     ['mean', 'std'],
            'Roll':        ['mean', 'std'],
            'Pitch':       ['mean', 'std'],
            'd_AngX':      'std',
            'd_AngY':      'std',
            'd_AngZ':      'std',
            'AngY_abs':    'mean',
            'abs_d_AngX':  'mean',
            'abs_d_AngZ':  'mean',
            'Jerk':        'mean',
            'is_rest':     'mean',
            'is_moderate': 'mean',
            'is_vigorous': 'mean',
            'is_lying':    'mean',
            'is_headdown': 'mean',
            'is_active':   'mean',
            'Tilt_Mag':    ['mean', 'std'],
            'DynX_abs':    'mean',
            'DynY_abs':    'mean',
            'DynZ_abs':    'mean',
        }

        df_1s = df.resample('1s').agg(agg_logic)
        df_1s.columns = ['_'.join(col).strip() for col in df_1s.columns.values]

        rename_dict = {
            'VeDBA_Clean_mean': 'Acc_Mean',
            'VeDBA_Clean_sum':  'Acc_Sum',
            'VeDBA_Clean_max':  'Acc_Max',
            'VeDBA_Clean_min':  'Acc_Min',
            'VeDBA_Clean_std':  'Acc_Std',
            'Gyro_Mag_mean':    'Gyro_Mean',
            'Gyro_Mag_max':     'Gyro_Max',
            'AccX_count':       'Data_Count',
            'DynX_mean':        'DynX_Mean',
            'DynX_std':         'DynX_Std',
            'DynY_mean':        'DynY_Mean',
            'DynY_std':         'DynY_Std',
            'DynZ_mean':        'DynZ_Mean',
            'DynZ_std':         'DynZ_Std',
            'Acc_Mag_mean':     'AccMag_Mean',
            'Acc_Mag_std':      'AccMag_Std',
            'Roll_mean':        'Roll_Mean',
            'Roll_std':         'Roll_Std',
            'Pitch_mean':       'Pitch_Mean',
            'Pitch_std':        'Pitch_Std',
            'd_AngX_std':       'AngX_Std',
            'd_AngY_std':       'AngY_Std',
            'd_AngZ_std':       'AngZ_Std',
            'AngY_abs_mean':    'AngY_Rate',
            'abs_d_AngX_mean':  'AngX_Rate',
            'abs_d_AngZ_mean':  'AngZ_Rate',
            'Jerk_mean':        'Jerk_Mean',
            'is_rest_mean':     'Frac_Rest',
            'is_moderate_mean': 'Frac_Moderate',
            'is_vigorous_mean': 'Frac_Vigorous',
            'is_lying_mean':    'Frac_Lying',
            'is_headdown_mean': 'Frac_HeadDown',
            'is_active_mean':   'Active_Fraction',
            'Tilt_Mag_mean':    'Tilt_Mag_Mean',
            'Tilt_Mag_std':     'Tilt_Mag_Std',
            'DynX_abs_mean':    'DynX_Mean_abs',
            'DynY_abs_mean':    'DynY_Mean_abs',
            'DynZ_abs_mean':    'DynZ_Mean_abs',
        }
        df_1s.rename(columns=rename_dict, inplace=True)

        # 后处理：可从聚合列直接推导的特征
        df_1s['Acc_Range']      = df_1s['Acc_Max'] - df_1s['Acc_Min']
        df_1s['Axis_Dominance'] = df_1s[['AngX_Std', 'AngY_Std', 'AngZ_Std']].values.argmax(axis=1)  # 0=X,1=Y,2=Z
        # ODBA：各轴动态加速度绝对均值之和（线性能耗代理）
        df_1s['ODBA'] = df_1s['DynX_Mean_abs'] + df_1s['DynY_Mean_abs'] + df_1s['DynZ_Mean_abs']

        # ==========================================
        # 步骤五：逐秒循环特征（频谱 + 分布统计矩 + 时序）
        # ==========================================
        spectral_entropy_list    = []
        zero_crossing_rate_list  = []
        dominant_frequency_list  = []
        cv_list                  = []
        skewness_list            = []
        kurtosis_list            = []
        acc_p25_list             = []
        acc_p75_list             = []
        autocorr_lag1_list       = []
        peak_count_list          = []
        vedba_rms_list           = []
        corr_xy_list             = []
        corr_xz_list             = []
        corr_yz_list             = []
        power_low_list           = []
        power_loco_list          = []
        power_high_list          = []
        sample_entropy_list      = []

        total_seconds = len(df_1s)
        if MoveDataNew.is_english:
            print(f"Calculating per-second features for {total_seconds} seconds...")
        else:
            print(f"正在逐秒计算高阶特征，共 {total_seconds} 秒...")

        def _safe_corr(a, b):
            if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        for i, idx in enumerate(df_1s.index):
            if (i + 1) % 3600 == 0 or (i + 1) == total_seconds:
                if MoveDataNew.is_english:
                    print(f"  {i+1}/{total_seconds} ({((i+1)/total_seconds)*100:.1f}%)")
                else:
                    print(f"  {i+1}/{total_seconds} ({((i+1)/total_seconds)*100:.1f}%)")

            end_time = idx + pd.Timedelta(seconds=1) - pd.Timedelta(nanoseconds=1)
            vedba_data = df.loc[idx:end_time, 'VeDBA_Clean'].values
            dynx_data  = df.loc[idx:end_time, 'DynX'].values
            dyny_data  = df.loc[idx:end_time, 'DynY'].values
            dynz_data  = df.loc[idx:end_time, 'DynZ'].values

            actual_fs = 10
            spectral_entropy_list.append(self.calculate_spectral_entropy(vedba_data))
            zero_crossing_rate_list.append(self.calculate_zero_crossing_rate(vedba_data))
            dominant_frequency_list.append(self.calculate_dominant_frequency(vedba_data, fs=actual_fs))
            cv_list.append(self.calculate_cv(vedba_data))

            s = pd.Series(vedba_data)
            skewness_list.append(float(s.skew())      if len(vedba_data) >= 3 else 0.0)
            kurtosis_list.append(float(s.kurtosis())  if len(vedba_data) >= 4 else 0.0)
            acc_p25_list.append(float(np.percentile(vedba_data, 25)) if len(vedba_data) > 0 else 0.0)
            acc_p75_list.append(float(np.percentile(vedba_data, 75)) if len(vedba_data) > 0 else 0.0)
            autocorr_lag1_list.append(float(s.autocorr(lag=1)) if len(vedba_data) >= 2 else 0.0)
            peak_count_list.append(self.count_peaks(vedba_data))

            # VeDBA_RMS
            vedba_rms_list.append(float(np.sqrt(np.mean(vedba_data**2))) if len(vedba_data) > 0 else 0.0)

            # 跨轴相关系数
            corr_xy_list.append(_safe_corr(dynx_data, dyny_data))
            corr_xz_list.append(_safe_corr(dynx_data, dynz_data))
            corr_yz_list.append(_safe_corr(dyny_data, dynz_data))

            # 频段功率
            p_low, p_loco, p_high = self.calculate_band_powers(vedba_data, fs=actual_fs)
            power_low_list.append(p_low)
            power_loco_list.append(p_loco)
            power_high_list.append(p_high)

            # 样本熵
            sample_entropy_list.append(self.calculate_sample_entropy(vedba_data))

        df_1s['Spectral_Entropy']         = spectral_entropy_list
        df_1s['Zero_Crossing_Rate']       = zero_crossing_rate_list
        df_1s['Dominant_Frequency']       = dominant_frequency_list
        df_1s['Coefficient_of_Variation'] = cv_list
        df_1s['Skewness']                 = skewness_list
        df_1s['Kurtosis']                 = kurtosis_list
        df_1s['Acc_P25']                  = acc_p25_list
        df_1s['Acc_P75']                  = acc_p75_list
        df_1s['Autocorr_Lag1']            = autocorr_lag1_list
        df_1s['Peak_Count']               = peak_count_list
        df_1s['VeDBA_RMS']               = vedba_rms_list
        df_1s['Corr_XY']                 = corr_xy_list
        df_1s['Corr_XZ']                 = corr_xz_list
        df_1s['Corr_YZ']                 = corr_yz_list
        df_1s['Power_Low']               = power_low_list
        df_1s['Power_Loco']              = power_loco_list
        df_1s['Power_High']              = power_high_list
        df_1s['Sample_Entropy']          = sample_entropy_list

        # ==========================================
        # 步骤六：过滤无效时间点（无采样数据的秒）
        # ==========================================
        df_1s = df_1s[df_1s['Data_Count'] > 0].copy()

        if MoveDataNew.is_english:
            print(f"Summary complete. Shape: {df_1s.shape}")
            print(f"Columns: {list(df_1s.columns)}")
        else:
            print(f"汇总完成。形状: {df_1s.shape}")
            print(f"列名: {list(df_1s.columns)}")

        self.finaldata = df_1s

    def drawPicture(self,picturepath=None):
        if(picturepath==None):
            picturepath=self.picturepath
        if not os.path.exists(picturepath):
            os.makedirs(picturepath)
        picturepath=os.path.join(picturepath,self.saveFileName)
        if not os.path.exists(picturepath):
            os.makedirs(picturepath)

        df_plot = self.moveCleanData.resample('60s').agg({
            'VeDBA_Clean': 'sum',  # 运动总量
            'Gyro_Mag': 'mean'  # 平均转速
        })

        # 3. 字体配置
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        plt.rcParams['axes.unicode_minus'] = False

        # 4. 按天分组绘图
        groups = df_plot.groupby(pd.Grouper(freq='D'))

        for date, daily_df in groups:
            # 跳过空数据的天
            if daily_df.empty or daily_df['VeDBA_Clean'].sum() == 0:
                continue

            # --- 创建画布：2行1列，共享X轴 ---
            # figsize=(15, 10) 让图片变高一点，容纳两个图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

            # --- 数据准备：将时间标准化为 0-24 小时 ---
            start_of_day = date
            x_hours = (daily_df.index - start_of_day).total_seconds() / 3600

            # ==========================================
            # 子图 1: 加速度 (运动总量)
            # ==========================================
            # 使用填充图 (fill_between) 表现"量"的感觉
            ax1.fill_between(x_hours, daily_df['VeDBA_Clean'], color='steelblue', alpha=0.6, label='运动总量 (VeDBA)')
            ax1.plot(x_hours, daily_df['VeDBA_Clean'], color='steelblue', linewidth=1)

            # 设置 Y 轴 (根据之前需求限制为 100，或者自适应)
            ax1.set_ylim(0, 100)
            ax1.set_ylabel('运动幅度 (g/min)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{date.date()} 身体运动总量 (Body Activity)', fontsize=14)
            ax1.grid(alpha=0.3, linestyle='--')
            ax1.legend(loc='upper right')

            # 标记高活跃阈值 (例如 50)
            ax1.axhline(y=50, color='red', linestyle=':', alpha=0.5)

            # ==========================================
            # 子图 2: 角速度 (旋转强度)
            # ==========================================
            # 使用折线图表现"波动"和"尖峰"
            ax2.plot(x_hours, daily_df['Gyro_Mag'], color='#ff7f0e', linewidth=1.2, label='旋转强度 (Gyro)')

            # 这里的 Y 轴范围可以根据角速度的统计值设定，比如 0-200度/秒
            # 如果不设，matplotlib 会自动适配
            # ax2.set_ylim(0, 300)

            ax2.set_ylabel('旋转速度 (°/s)', fontsize=12, fontweight='bold')
            ax2.set_title(f'{date.date()} 姿态旋转强度 (Rotational Intensity)', fontsize=14)
            ax2.grid(alpha=0.3, linestyle='--')
            ax2.legend(loc='upper right')

            # ==========================================
            # 公共坐标轴设置 (X轴)
            # ==========================================
            xticks = np.arange(0, 25, 2)
            xtick_labels = [f"{int(h):02d}:00" for h in xticks]

            plt.xlabel('当日时间 (Hour)', fontsize=12)
            plt.xticks(xticks, xtick_labels, fontsize=10)
            plt.xlim(0, 24)  # 强制显示全天

            # 调整子图间距
            plt.tight_layout()

            # 5. 保存图片
            save_name = f'daily_{date.date()}_dual_move.png'
            save_full_path = os.path.join(picturepath, save_name)
            plt.savefig(save_full_path, dpi=120)
            plt.close(fig)
            if MoveDataNew.is_english:
                print(f"Saved: {save_full_path}")
            else:
                print(f"已保存: {save_full_path}")

            # 关闭画布
    def saveData(self,filepath=None):
        self.finaldata.to_csv(filepath)
        base, ext = os.path.splitext(filepath)
        raw_filepath = base + "rawimudata" + ext
        self.moveCleanData.to_csv(raw_filepath)
        if MoveDataNew.is_english:
            print(f"Raw IMU data saved: {raw_filepath}")
        else:
            print(f"原始IMU数据已保存: {raw_filepath}")
        
    def generate_advanced_visualizations(self):
        """
        Generate advanced visualizations for论文
        """
        # 确保目录存在
        os.makedirs(self.savepath, exist_ok=True)
        
        # 1. 采样率稳定性直方图
        self._plot_sampling_rate_stability()
        
        # 2. 数据缺失与连续性热力图
        self._plot_data_completeness_heatmap()
        
        # 3. 伪影剔除对比图
        self._plot_artifact_rejection_case_study()
        
    def _plot_sampling_rate_stability(self):
        """
        Plot sampling rate stability histogram
        """
        if not hasattr(self, 'data_quality_results'):
            return
        
        quality_df = self.data_quality_results['quality_df']
        plot_data = quality_df[quality_df['hz'] > 0]
        
        plt.figure(figsize=(12, 6))
        # 绘制直方图
        sns.histplot(plot_data['hz'], bins=30, kde=True, color='purple', alpha=0.7)
        # 添加目标线
        plt.axvline(x=10, color='green', linestyle='--', linewidth=1.5, label='Target (10Hz)')
        # 设置标题和标签
        plt.title('Sampling Rate Stability Histogram', fontsize=14, fontweight='bold')
        plt.xlabel('Samples per Second (Hz)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3, linestyle='--')
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(self.savepath, f"{self.saveFileName}_sampling_rate_stability.png"), dpi=600)
        plt.close()
        
    def _plot_data_completeness_heatmap(self):
        """
        Plot data completeness heatmap
        """
        if not hasattr(self, 'data_quality_results'):
            return
        
        quality_df = self.data_quality_results['quality_df']
        
        # 准备数据：按天和小时分组
        quality_df['date'] = quality_df.index.date
        quality_df['hour'] = quality_df.index.hour
        
        # 计算每天每小时的数据完整率
        daily_hourly = quality_df.groupby(['date', 'hour']).size().unstack(fill_value=0)
        
        # 计算完整率（假设每小时应该有3600秒的数据）
        completeness = daily_hourly / 3600 * 100
        
        # 只保留有数据的天
        completeness = completeness[completeness.sum(axis=1) > 0]
        
        if completeness.empty:
            return
        
        plt.figure(figsize=(14, len(completeness) * 0.8))
        # 绘制热力图
        sns.heatmap(completeness, cmap='YlGnBu', annot=True, fmt='.1f', cbar_kws={'label': 'Completeness (%)'})
        # 设置标题和标签
        plt.title('Data Completeness Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Date', fontsize=12)
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(self.savepath, f"{self.saveFileName}_data_completeness_heatmap.png"), dpi=600)
        plt.close()
        
    def _plot_artifact_rejection_case_study(self):
        """
        Plot artifact rejection case study
        """
        if not hasattr(self, 'angle_accel_results'):
            return
        
        df = self.moveCleanData
        
        # 找到伪影点
        artifact_mask = df['Gyro_Status'] == 'Artifact'
        artifact_indices = df[artifact_mask].index
        
        if len(artifact_indices) == 0:
            return
        
        # 选择第一个伪影点作为案例
        artifact_time = artifact_indices[0]
        # 选择前后10秒的时间窗口
        window_start = artifact_time - pd.Timedelta(seconds=10)
        window_end = artifact_time + pd.Timedelta(seconds=10)
        window_data = df.loc[window_start:window_end].copy()
        
        plt.figure(figsize=(14, 6))
        # 绘制清洗前的角速度
        plt.plot(window_data.index, window_data['Gyro_Mag'], 'r-', linewidth=2, label='Before Cleaning')
        # 绘制清洗后的角速度（伪影点被置为0）
        cleaned_gyro = window_data['Gyro_Mag'].copy()
        cleaned_gyro[window_data['Gyro_Status'] == 'Artifact'] = 0
        plt.plot(window_data.index, cleaned_gyro, 'g-', linewidth=2, label='After Cleaning')
        # 标记伪影点
        artifact_in_window = window_data[window_data['Gyro_Status'] == 'Artifact']
        if not artifact_in_window.empty:
            plt.scatter(artifact_in_window.index, artifact_in_window['Gyro_Mag'], color='red', s=100, marker='x', label='Artifact')
        # 设置标题和标签
        plt.title('Artifact Rejection Case Study', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Angular Velocity (°/s)', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3, linestyle='--')
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(self.savepath, f"{self.saveFileName}_artifact_rejection_case.png"), dpi=600)
        plt.close()
    
    def generate_report(self, report_path=None):
        """
        Generate a detailed Markdown report with all analysis results
        """
        if report_path is None:
            # 确保目录存在
            os.makedirs(self.savepath, exist_ok=True)
            report_path = os.path.join(self.savepath, f"{self.saveFileName}_report.md")
        
        # Start with report header
        markdown_content = f"""# 运动量数据分析报告

## 基本信息
- 分析文件: {self.saveFileName}
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # 1. 数据清洗结果
        if hasattr(self, 'data_quality_results'):
            results = self.data_quality_results
            markdown_content += f"""
## 1. 数据清洗结果
- 原始数据行数: {results['remaining_count'] + results['deleted_count']}
- 删除无效时间戳: {results['deleted_count']} 行
- 剩余有效数据: {results['remaining_count']} 行

"""
            
            # 2. 数据连续性分析
            gaps = results['gaps']
            if len(gaps) > 0:
                markdown_content += f"""
## 2. 数据连续性分析
发现 {len(gaps)} 处明显的数据断裂：

| 断裂开始时间 | 数据恢复时间 | 中断时长(秒) |
|------------|------------|------------|
"""
                for end_time, row in gaps.iterrows():
                    duration = row['dt']
                    start_time = end_time - pd.Timedelta(seconds=duration)
                    markdown_content += f"| {str(start_time)} | {str(end_time)} | {duration:.2f} |\n"
            else:
                markdown_content += f"""
## 2. 数据连续性分析
未发现超过 3 秒的断裂。

"""
            
            # 3. 采样率质量分析
            quality_df = results['quality_df']
            markdown_content += f"""
## 3. 采样率质量分析

### 整体统计
"""
            status_counts = quality_df['status'].value_counts()
            for status, count in status_counts.items():
                markdown_content += f"- {status}: {count} 秒\n"
            
            # 4. 详细异常分析（精确到分钟）
            markdown_content += f"""
### 详细异常分析（精确到分钟）

"""
            
            # 按分钟聚合异常数据
            quality_df['date_minute'] = quality_df.index.floor('T')
            minute_report = quality_df.groupby(['date_minute', 'status']).size().unstack(fill_value=0)
            
            # 找出有异常的分钟
            abnormal_minutes = minute_report[
                (minute_report.get('Too High (>15 Hz)', 0) > 0) |
                (minute_report.get('Too Low (<5 Hz)', 0) > 0)
            ]
            
            if not abnormal_minutes.empty:
                # 按小时分组，检查是否整小时异常
                abnormal_minutes['date_hour'] = abnormal_minutes.index.floor('H')
                hourly_minutes = abnormal_minutes.groupby('date_hour').size()
                
                for hour, minute_count in hourly_minutes.items():
                    # 检查是否整小时60分钟都异常
                    if minute_count == 60:
                        markdown_content += f"### 整小时异常: {hour.strftime('%Y-%m-%d %H:00')}\n"
                        markdown_content += "该小时内所有分钟都存在采样率异常\n\n"
                    else:
                        # 列出该小时内的异常分钟
                        hour_minutes = abnormal_minutes[abnormal_minutes['date_hour'] == hour]
                        markdown_content += f"### 小时异常: {hour.strftime('%Y-%m-%d %H:00')}\n"
                        markdown_content += "| 分钟 | 过高(秒) | 过低(秒) | 缺失(秒) |\n"
                        markdown_content += "|------|---------|---------|---------|\n"
                        for minute, row in hour_minutes.iterrows():
                            high = row.get('Too High (>15 Hz)', 0)
                            low = row.get('Too Low (<5 Hz)', 0)
                            missing = row.get('Missing (0 Hz)', 0)
                            if high > 0 or low > 0:
                                markdown_content += f"| {minute.strftime('%H:%M')} | {high} | {low} | {missing} |\n"
                        markdown_content += "\n"
            else:
                markdown_content += "未发现采样率异常时段。\n\n"
        
        # 5. 角速度与加速度分析
        if hasattr(self, 'angle_accel_results'):
            results = self.angle_accel_results
            markdown_content += f"""
## 4. 角速度与加速度分析

### 角速度异常检测
- 真实剧烈运动点: {results['n_real']} 个
- 伪影/噪点: {results['n_artifact']} 个

### 判定标准
- 疑似异常高值线: > {results['SUSPICIOUS_LIMIT']:.2f} °/s
- 邻居活跃达标线: > {results['ACTIVE_BASE']:.2f} °/s

### 加速度清洗
- 剔除角速度伪影点: {results['artifact_count']} 个
- 剔除疑似失重/失效点: {results['freefall_count']} 个
- 噪声底噪阈值: < {results['NOISE_FLOOR']:.4f} g
- 撞击削峰上限: > {results['IMPACT_CEILING']:.4f} g

"""
            
            # 伪影示例
            if results['n_artifact'] > 0:
                markdown_content += f"""
### 伪影示例

| Gyro_Mag | Neighbor_Max | Isolation_Ratio |
|----------|--------------|----------------|
"""
                for _, row in results['artifact_examples'].iterrows():
                    markdown_content += f"| {row['Gyro_Mag']:.2f} | {row['Neighbor_Max']:.2f} | {row['Isolation_Ratio']:.2f} |\n"
                markdown_content += "\n"
            
            # 真实运动示例
            if results['n_real'] > 0:
                markdown_content += f"""
### 真实运动示例

| Gyro_Mag | Neighbor_Max | Isolation_Ratio |
|----------|--------------|----------------|
"""
                for _, row in results['real_examples'].iterrows():
                    markdown_content += f"| {row['Gyro_Mag']:.2f} | {row['Neighbor_Max']:.2f} | {row['Isolation_Ratio']:.2f} |\n"
                markdown_content += "\n"
        
        # 6. 结论
        markdown_content += f"""
## 5. 结论

分析完成。报告包含了所有异常信息，精确到分钟级别。

"""
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        if MoveDataNew.is_english:
            print(f"Report generated: {report_path}")
        else:
            print(f"报告已生成: {report_path}")
        
        return report_path
    def zeroclan(self,rawdata,namelist):
        for name in namelist:
            # 步骤1: 复制列避免修改原始数据
            rawdata.loc[:,'x_temp'] = rawdata[name].copy()

            # 步骤2: 定位0值位置 & 计算前后平均值
            # 生成两个辅助列: 前移一位和后移一位的值
            rawdata.loc[:,'prev_val'] = rawdata['x_temp'].shift(1)
            rawdata.loc[:,'next_val'] = rawdata['x_temp'].shift(-1)

            # 计算前后非零值的平均值（忽略0值）
            avg_condition = (rawdata['x_temp'] == 0)  # 当前值为0
            rawdata.loc[avg_condition, 'x_temp'] = (rawdata['prev_val'] + rawdata['next_val']) / 2

            # 步骤3: 处理边界问题（首尾无前/后值）
            # 首项为0时：用后一项填充（若后一项非零）
            if rawdata['x_temp'].iloc[0] == 0 and not np.isnan(rawdata['next_val'].iloc[0]):
                rawdata['x_temp'].iloc[0] = rawdata['next_val'].iloc[0]

            # 末项为0时：用前一项填充（若前一项非零）
            if rawdata['x_temp'].iloc[-1] == 0 and not np.isnan(rawdata['prev_val'].iloc[-1]):
                rawdata['x_temp'].iloc[-1] = rawdata['prev_val'].iloc[-1]

            # 步骤4: 更新原列并清理临时列
            rawdata.loc[:,name] = rawdata['x_temp']
            rawdata = rawdata.drop(columns=['x_temp', 'prev_val', 'next_val'])
        return rawdata
if __name__=="__main__":
    data = MoveDataNew(movefilepath=r"F:\aaa甜菜粕\B1",
                            picturepath=r"C:\Users\zengz\Desktop\B1图",savepath=r"C:\Users\zengz\Desktop\B1图\B1.csv"
                            ,is_english=True)
    data.saveData(r"C:\Users\zengz\Desktop\B1图\B1.csv")
    data.generate_report()
    data.drawPicture()
    data.generate_advanced_visualizations()