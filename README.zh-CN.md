# SwineSync OpenSource

这是一个可以单独发布到 GitHub 的开源版目录，包含 ECG 转换、缓存生成、心率处理、运动量处理、文件夹检查和基础配置能力。

## 文档导航

- [中文使用说明](./使用说明.zh-CN.md)
- [English User Guide](./USER_GUIDE.en.md)
- [English README](./README.en.md)

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

启动软件：

```bash
python main.py
```

## 功能范围

- 转换 ECG
- 转换 ECG（使用缓存）
- 仅生成缓存
- 心率处理
- 文件夹检查
- 运动处理（旧版）
- 运动处理（新版）
- 配置文件保存
- 中英文切换

## 关键说明

- 第一个 ECG 转换按钮相当于后两个流程合并。
- 除非机器有 64G 内存和 50 系以上显卡，否则更建议拆开流程跑。
- 开源版已移除“合成数据输出”功能。
- 耗时任务通过独立后台子进程执行，界面不会像以前那样容易卡死。

## 适合独立仓库

这个目录已经按可独立托管的方式组织：

- 所有源码都在当前文件夹内
- 模型文件集中放在 [models](./models)
- 配置文件在 [config.json](./config.json)
- 中文和英文文档都已经包含

## 许可与引用

- 允许非商用使用
- 商用必须另行获得书面授权
- 公开使用、论文、报告、发布衍生版本时必须注明并引用本项目
- 具体条款见 [LICENSE](./LICENSE) 和 [CITATION.cff](./CITATION.cff)
