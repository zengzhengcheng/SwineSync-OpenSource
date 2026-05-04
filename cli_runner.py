import json
import sys
from pathlib import Path

from folder_inspector import inspect_folder
from heartClass import HeartClass
from moveClass import MoveData
from moveClassNew import MoveDataNew
from neuronConvert import NeuronConvert, NeuronConvertToCache


def _algorithm_to_origin_resample(algorithm):
    return algorithm == "original_timestamps"


def run_task(task):
    task_type = task["task_type"]

    if task_type in {"convert", "convert_cache"}:
        config = task["config"]
        converter = NeuronConvert(
            None,
            5120 * 3,
            inputshape=1,
            modelout=1,
            batchSize=config["batch_size"],
            cacheSize=config["cache_size"],
            usecache=(task_type == "convert_cache"),
            cleanfile=False,
            correct=config["correct"],
            origin_resample=_algorithm_to_origin_resample(config["interpolation_algorithm"]),
        )
        converter.allData(task["selected_files"], task["output_dir"])
        return

    if task_type == "cache_only":
        config = task["config"]
        converter = NeuronConvertToCache(
            None,
            batchSize=config["batch_size"],
            cacheSize=config["cache_size"],
            origin_resample=_algorithm_to_origin_resample(config["interpolation_algorithm"]),
        )
        converter.allData(task["selected_files"], task["output_dir"])
        return

    if task_type == "heart_process":
        data = HeartClass(task["source_dir"], task["picture_dir"], task["save_path"])
        data.saveData(task["save_path"])
        data.drawPicture()
        return

    if task_type == "folder_inspect":
        for message in inspect_folder(task["source_dir"], task["language"]):
            print(message)
        return

    if task_type == "move_old":
        data = MoveData(
            task["source_dir"],
            task["picture_dir"],
            task["save_path"],
            is_english=task["language"] == "en",
        )
        data.saveData(task["save_path"])
        data.drawPicture()
        return

    if task_type == "move_new":
        data = MoveDataNew(
            task["source_dir"],
            task["picture_dir"],
            task["save_path"],
            is_english=task["language"] == "en",
        )
        data.saveData(task["save_path"])
        data.generate_report()
        data.drawPicture()
        return

    raise ValueError(f"Unsupported task type: {task_type}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python cli_runner.py <task_json_path>")

    task_path = Path(sys.argv[1]).resolve()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    run_task(task)


if __name__ == "__main__":
    main()
