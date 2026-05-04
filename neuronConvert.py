from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import concurrent
import subprocess
import traceback

import matplotlib.pyplot as plt
import numpy as np

from models.ECGModelInference import ECGModelInference
from neuronDataClass import NeuronDataClass
from neuronDataManager import NeuronDataManager
from utils import outclean


def check_cuda_installation():
    gpu_brands = ["NVIDIA", "AMD", "ATI", "GeForce", "Radeon"]

    def _check_output(output):
        return any(brand in output for brand in gpu_brands)

    for cmd in (
        ["wmic", "path", "win32_VideoController", "get", "name"],
        ["powershell", "-Command", "Get-WmiObject Win32_VideoController | Select-Object -ExpandProperty Name"],
    ):
        try:
            output = subprocess.check_output(
                cmd,
                shell=False,
                universal_newlines=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if _check_output(output):
                return True
        except Exception:
            continue
    return False


def debug_plot(sig, raw_lab, clean_lab):
    sig = sig.reshape(-1)
    raw_lab = raw_lab.reshape(-1)
    clean_lab = clean_lab.reshape(-1)
    plt.figure(figsize=(30, 12))
    plt.plot(sig, color="black", alpha=0.6, linewidth=0.8, label="ECG Signal")
    plt.fill_between(
        np.arange(len(sig)),
        sig.min(),
        sig.max(),
        where=(raw_lab == 1),
        color="orange",
        alpha=0.3,
        label="Model Raw Mask",
    )
    peaks = np.where(clean_lab == 1)[0]
    if len(peaks) > 0:
        plt.scatter(peaks, sig[peaks], color="red", s=50, zorder=5, marker="o", label="Detected R-Peak")
    plt.tight_layout()
    plt.savefig("testas.png")
    plt.show()


class NeuronConvert:
    def __init__(
        self,
        parent,
        neuronLength,
        inputshape=1,
        modelout=1,
        batchSize=4,
        cacheSize=2,
        usecache=False,
        cleanfile=False,
        correct=True,
        origin_resample=True,
    ):
        self.batchSize = batchSize
        self.cacheSize = cacheSize
        self.modelout = modelout
        self.parent = parent
        self.neuronLength = neuronLength
        self.cache = usecache
        self.cleanfile = cleanfile
        self.correct = correct
        self.origin_resample = origin_resample

        base_dir = Path(__file__).resolve().parent
        model_paths = {
            "tragan": str(base_dir / "models" / "tragan_model.onnx"),
            "trabase": str(base_dir / "models" / "trabase_model.onnx"),
            "traganbase": str(base_dir / "models" / "traganbase_model.onnx"),
        }
        self.ort_session = ECGModelInference(model_paths)

    def allData(self, filepaths, savepath):
        if not check_cuda_installation():
            print("No discrete GPU detected, using CPU execution.")

        providers = [
            NeuronDataClass(
                filepath,
                savepath,
                cache=self.cache,
                cleanfile=self.cleanfile,
                correct=self.correct,
                origin_resample=self.origin_resample,
            )
            for filepath in filepaths
        ]
        neuronDataManager = NeuronDataManager(providers, batchSize=self.batchSize, cacheSize=self.cacheSize)
        dataDict = neuronDataManager.get_next_data()
        while dataDict is not None:
            datas = []
            labels = []
            indexlists = []
            results = {}
            keys = []
            for key, value in dataDict.items():
                data = value["data"]
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                label = value["label"]
                indexlist = value["indexlist"]
                start = value["start"]
                swt = value["swt"]
                tradata = np.concatenate((data, swt), axis=0)
                keys.append(key)
                results[key] = [start]
                labels.append(label)
                datas.append(tradata)
                indexlists.append(indexlist)

            datas = np.array(datas)
            labels = np.array(labels)
            indexlists = np.array(indexlists)
            batch = len(keys)
            out, cleanecg = self(datas, labels, indexlists, batch)
            for index, key in enumerate(keys):
                results[key].append(out[index])
                results[key].append(cleanecg[index])
            neuronDataManager.send_output(results)
            dataDict = neuronDataManager.get_next_data()

    def __call__(self, input_data, label, indexlist, batch):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        label = label.reshape([batch, 1, self.neuronLength])
        data = input_data.reshape([batch, -1, self.neuronLength])
        indexlist = indexlist.reshape([batch, 1, self.neuronLength])
        data = data.astype(np.float32)
        cleanecg, out = self.ort_session.getout(data)
        outdata = outclean(cleanecg, out, input_data)
        del data, out, indexlist, label
        return outdata, cleanecg


class NeuronConvertToCache:
    def __init__(self, parent, batchSize=4, cacheSize=2, origin_resample=True):
        self.batchSize = batchSize
        self.cacheSize = cacheSize
        self.parent = parent
        self.cache = True
        self.origin_resample = origin_resample

    def allData(self, filepaths, savepath):
        providers = [
            NeuronDataClass(filepath, savepath, cache=self.cache, origin_resample=self.origin_resample)
            for filepath in filepaths
        ]
        if not providers:
            print("No files to process")
            return
        if len(providers) < self.batchSize:
            self.batchSize = len(providers)
        with ThreadPoolExecutor(max_workers=self.batchSize) as executor:
            future_to_index = {}
            for i, provider in enumerate(providers):
                future = executor.submit(provider.convertTocache)
                future_to_index[future] = i
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Task {index} failed: {exc}")
                    traceback.print_exc()
