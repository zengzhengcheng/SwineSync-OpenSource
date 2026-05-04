# SwineSync OpenSource User Guide

## 1. Overview

Software name:

- SwineSync OpenSource

The open-source release includes:

- Convert ECG
- Convert ECG (Use Cache)
- Build Cache Only
- Heart-Rate Processing
- Folder Inspection
- Movement Processing (Legacy)
- Movement Processing (New)

Notes:

- synthetic-data export has been removed from the open-source release
- synthetic data is not output by default

## 2. Folder Structure

- [main.py](./main.py): launch entry
- [gui.py](./gui.py): main GUI
- [cli_runner.py](./cli_runner.py): background task runner
- [config.json](./config.json): configuration file
- [config_manager.py](./config_manager.py): config read/write
- [translations.py](./translations.py): Chinese and English UI strings
- [dialogs.py](./dialogs.py): file confirmation dialogs
- [neuronConvert.py](./neuronConvert.py): ECG conversion orchestration
- [neuronDataClass.py](./neuronDataClass.py): ECG processing core
- [heartClass.py](./heartClass.py): heart-rate processing
- [folder_inspector.py](./folder_inspector.py): folder inspection
- [moveClass.py](./moveClass.py): legacy movement processing
- [moveClassNew.py](./moveClassNew.py): new movement processing
- [models](./models): model files and inference code
- [heartCorrect](./heartCorrect): RR-correction logic

## 3. Installation and Launch

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch:

```bash
python main.py
```

## 4. Front-End / Back-End Execution Model

This release separates the GUI from heavy processing.

1. The GUI handles parameter input, file selection, logs, and status display.
2. Heavy work runs in a separate Python subprocess.
3. Logs are streamed back to the GUI in real time.
4. The interface is much less likely to freeze than the original in-process approach.

## 5. Configuration File

Configuration file:

- [config.json](./config.json)

Main fields:

- `language`
- `batch_size`
- `cache_size`
- `interpolation_algorithm`
- `correct`

`interpolation_algorithm` supports:

- `original_timestamps`
- `uniform_grid`

Example:

```json
{
  "language": "zh",
  "batch_size": 2,
  "cache_size": 1,
  "interpolation_algorithm": "original_timestamps",
  "correct": true
}
```

## 6. Language Switching

Use the language toggle button in the GUI.

- The selection is saved into the config file automatically.
- It remains effective on the next launch.

## 7. Practical Notes and Hardware Guidance

1. The first conversion button is effectively the combination of the next two steps.
2. Unless you have 64 GB RAM and a 50-series-or-better GPU, splitting the workflow is usually recommended.
3. For cache generation, increasing `cache_size` increases concurrency.
4. A practical rule is about 6 files per 32 GB of free RAM.
5. For `Convert ECG (Use Cache)`, with 32 GB RAM and around 6 files, `cache_size=2` is usually enough.
6. During conversion, the software may warn if a page contains too few heartbeats.

## 8. ECG Features

### 8.1 Convert ECG

Purpose:

- Run the full ECG conversion workflow

Notes:

- This button is effectively “build cache first, then finish conversion”
- On lower-spec machines, using the next two buttons separately is safer

### 8.2 Convert ECG (Use Cache)

Purpose:

- Finish ECG conversion by using already generated cache files

Recommended when:

- cache files are already available
- you want lower peak resource usage

### 8.3 Build Cache Only

Purpose:

- Build cache without running the final conversion stage

Recommendations:

- Increasing `cache_size` raises cache-stage concurrency
- A practical rule is about 6 files per 32 GB of free RAM

File matching rule:

- file name starts with `Raw`
- or file name contains `BMD`

## 9. Heart-Rate Processing

Purpose:

- aggregate heart-rate data
- export CSV
- generate daily images

Workflow:

1. Click `Heart-Rate Processing`.
2. Select the folder that contains heart-rate data.
3. Select the picture output folder.
4. Select the result CSV save path.
5. Wait for the background task to finish.

Input recommendation:

- heart-rate source files should preferably be CSV
- file names should match `Raw*` or contain `BMD`

## 10. Folder Inspection

Purpose:

- check whether folder structure and file naming are reasonable

Workflow:

1. Click `Folder Inspection`.
2. Select the processing root folder.
3. Read the inspection results in the log panel.

Main checks:

- whether date folders use a `YYYYMMDD`-style prefix
- whether dates inside file names match the folder date
- whether a date folder contains suspiciously few files
- whether one time period appears to contain more than one pig identifier

Recommended structure:

- processing folder / date folder / sensor data

Recommended date folder format:

- `20250612`

Important:

- one time period should only contain one pig's data

## 11. Movement Processing

### 11.1 Movement Processing (Legacy)

Use when:

- the sensor contains gyroscope data only

Workflow:

1. Click `Movement Processing (Legacy)`.
2. Select the source folder.
3. Select the picture output folder.
4. Select the result save path.

### 11.2 Movement Processing (New)

Use when:

- the sensor includes acceleration data

Workflow:

1. Click `Movement Processing (New)`.
2. Select the source folder.
3. Select the picture output folder.
4. Select the result save path.

Notes:

- selecting the processing folder is enough for automatic data discovery

## 12. Log Panel

The log panel shows:

- background task start messages
- standard output
- standard error
- success and failure notifications

## 13. FAQ

### 13.1 Why is synthetic-data export missing

Because it has been intentionally removed from the open-source release.

### 13.2 Why is the GUI more responsive now

Because heavy tasks are now executed in a separate background subprocess.

## 14. Reference Datasets

If users want related reference datasets, they can apply through ScienceDB for:

- Dataset 1 DOI: `10.57760/sciencedb.35116`
- Dataset 2 DOI: `10.57760/sciencedb.35878`

Additional notes:

- The Scientific Data paper DOI is pending update.
- Users need to submit an access request through ScienceDB.
- Requests with an appropriate reason are generally expected to be approved.
- Contact email: `zengzhengcheng@cau.edu.cn`
