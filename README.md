# SwineSync OpenSource

Standalone open-source release for ECG conversion, cache generation, heart-rate processing, movement processing, and folder inspection.

## Documents

- [简体中文 README](./README.zh-CN.md)
- [English README](./README.en.md)
- [中文使用说明](./使用说明.zh-CN.md)
- [English User Guide](./USER_GUIDE.en.md)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch:

```bash
python main.py
```

## Included Features

- Convert ECG
- Convert ECG (Use Cache)
- Build Cache Only
- Heart-Rate Processing
- Folder Inspection
- Movement Processing (Legacy)
- Movement Processing (New)
- Config file support
- Chinese / English switching

## Notes

- The first ECG conversion button is effectively the combination of the next two steps.
- Unless the machine has 64 GB RAM and a 50-series-or-better GPU, splitting the workflow is usually safer.
- Synthetic-data export has been removed from this open-source release.
- Heavy processing runs in a separate background process to keep the GUI responsive.

## Reference Datasets

Two related reference datasets are available through ScienceDB:

- Dataset 1 DOI: `10.57760/sciencedb.35116`
- Dataset 2 DOI: `10.57760/sciencedb.35878`

Notes:

- Access requires a request through ScienceDB.
- The Scientific Data paper DOI is pending update.
- Users should submit a reasonable application statement through ScienceDB.
- Contact email: `zengzhengcheng@cau.edu.cn`

## GitHub Use

This folder is organized so it can be used as an independent GitHub repository.

- All source files are self-contained inside this directory.
- Model files are stored under [models](./models).
- Runtime configuration is stored in [config.json](./config.json).
- User documentation is included in both Chinese and English.

## Related Repositories

- Open-source processing repository: [SwineSync-OpenSource](https://github.com/zengzhengcheng/SwineSync-OpenSource)
- Public packaged desktop release: [SwineSync-Studio](https://github.com/zengzhengcheng/SwineSync-Studio)

## License and Citation

- Non-commercial use is allowed.
- Commercial use is not allowed without separate written permission.
- Public use, reports, papers, and derivative releases must attribute this project.
- See [LICENSE](./LICENSE) and [CITATION.cff](./CITATION.cff).
