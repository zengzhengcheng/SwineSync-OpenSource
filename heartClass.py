import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytz
from tzlocal import get_localzone


class HeartClass:
    def __init__(self, heartfilepath, picturepath, savepath=".\\", oneDir=False):
        self.heartfilepath = heartfilepath
        self.picturepath = picturepath
        self.savepath = savepath
        self.oneDir = oneDir
        self.heartdata = pd.DataFrame(columns=["label"])
        self.load_data()

    def drawPicture(self, picturepath=None):
        if picturepath is None:
            picturepath = self.picturepath

        picture_dir = Path(picturepath)
        picture_dir.mkdir(parents=True, exist_ok=True)

        if self.heartdata.empty:
            print("Warning: no heart-rate data available for plotting.")
            return

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        df = self.heartdata.resample("1min").sum()
        for date, daily_df in df.groupby(pd.Grouper(freq="D")):
            if daily_df.empty:
                continue

            plt.figure(figsize=(15, 6))
            time_only = daily_df.index - date
            full_day = pd.date_range(date, date + pd.Timedelta(days=1), freq="1h")
            hours_ticks = [(x - date).total_seconds() / 3600 for x in full_day]
            hour_labels = [x.strftime("%H:%M") for x in full_day]

            plt.xticks(hours_ticks, hour_labels)
            plt.plot(
                time_only.total_seconds() / 3600,
                daily_df["label"],
                color="steelblue",
                label="Daily data",
            )
            plt.ylim(0, 240)
            plt.xlabel("Time of day", fontsize=12)
            plt.ylabel("Heart rate", fontsize=12)
            plt.title(f"{date.date()} Heart-Rate Distribution", fontsize=14)
            plt.xlim(0, 24)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(picture_dir / f"daily_{date.date()}_heart.png", dpi=120)
            plt.close()

    def load_data(self):
        heart_dir = Path(self.heartfilepath)
        if not heart_dir.exists():
            raise FileNotFoundError(f"Heart-rate source folder not found: {heart_dir}")

        all_paths = [path for path in heart_dir.iterdir() if path.is_file()]
        csv_files = [
            path
            for path in all_paths
            if path.suffix.lower() == ".csv" and ("Raw" in path.name or "BMD" in path.name)
        ]

        if not csv_files:
            print("Warning: no heart-rate csv files matched the naming rule (Raw* or *BMD*).")
            self.heartdata = pd.DataFrame(columns=["label"])
            return

        try:
            local_timezone = get_localzone()
        except Exception:
            local_timezone = pytz.timezone("Asia/Shanghai")
            print("Warning: fallback timezone set to Asia/Shanghai.")

        data_frames = []
        for file_path in csv_files:
            print(file_path)
            data_frame, has_header = self._read_csv(file_path)
            data_frame["timestamp"] = data_frame["timestamp"].apply(self._clean_timestamp)

            try:
                data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])
            except Exception:
                print(f"Warning: failed to parse timestamps in {file_path}")
                continue

            target_label = 1 if has_header else 2
            data_frame = data_frame[data_frame["label"] == target_label]
            if data_frame.empty:
                continue

            data_frame.set_index("timestamp", inplace=True)
            if getattr(data_frame.index, "tz", None) is not None:
                data_frame.index = data_frame.index.tz_convert(local_timezone).tz_localize(None)

            data_frames.append(data_frame[["label"]])

        if not data_frames:
            print("Warning: heart-rate files were found, but no usable rows remained after filtering.")
            self.heartdata = pd.DataFrame(columns=["label"])
            return

        self.heartdata = pd.concat(data_frames).sort_index()

    def saveData(self, savepath):
        save_file = Path(savepath)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        self.heartdata.to_csv(save_file, header=True, encoding="utf-8-sig")

    @staticmethod
    def _clean_timestamp(value):
        if isinstance(value, str) and value.startswith("b'") and value.endswith("'"):
            return value[2:-1]
        return value

    @staticmethod
    def _read_csv(file_path):
        has_header = False
        try:
            data_frame = pd.read_csv(file_path)
            if "timestamp" in data_frame.columns and "label" in data_frame.columns:
                has_header = True
            else:
                data_frame = pd.read_csv(file_path, names=["timestamp", "label"])
        except Exception:
            data_frame = pd.read_csv(file_path, names=["timestamp", "label"])
        return data_frame, has_header
