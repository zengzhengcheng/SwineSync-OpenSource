# SwineSync OpenSource

This directory is prepared as a standalone open-source release that can be published to GitHub independently.

## Documentation

- [Chinese User Guide](./使用说明.zh-CN.md)
- [English User Guide](./USER_GUIDE.en.md)
- [中文 README](./README.zh-CN.md)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch:

```bash
python main.py
```

## Features

- Convert ECG
- Convert ECG (Use Cache)
- Build Cache Only
- Heart-Rate Processing
- Folder Inspection
- Movement Processing (Legacy)
- Movement Processing (New)
- Config file support
- Chinese / English switching

## Key Notes

- The first ECG conversion button is effectively the combination of the next two steps.
- Unless the machine has 64 GB RAM and a 50-series-or-better GPU, splitting the workflow is usually recommended.
- Synthetic-data export is intentionally removed from this open-source release.
- Heavy processing is executed in a separate background process so the GUI remains responsive.

## Ready for GitHub

- All source files are contained inside this directory.
- Model files are stored under [models](./models).
- Runtime configuration is stored in [config.json](./config.json).
- Chinese and English user documents are both included.

## License and Citation

- Non-commercial use is allowed.
- Commercial use requires separate written permission.
- Public use, papers, reports, and derivative releases must attribute and cite this project.
- See [LICENSE](./LICENSE) and [CITATION.cff](./CITATION.cff).
