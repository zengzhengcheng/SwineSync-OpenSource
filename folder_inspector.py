import re
from pathlib import Path


DATE_FOLDER_PATTERN = re.compile(r"^\d{8}.*$")
DATE_IN_FILE_PATTERN = re.compile(r"(\d{8}|\d{4}-\d{2}-\d{2})")
PIG_HINT_PATTERN = re.compile(r"\b([A-Za-z]{1,3}\d{1,4})\b")


def inspect_folder(root_path, language="zh"):
    root_dir = Path(root_path)
    messages = []

    if not root_dir.exists():
        return [_msg(language, "root_missing", root=str(root_dir))]
    if not root_dir.is_dir():
        return [_msg(language, "root_not_dir", root=str(root_dir))]

    date_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    if not date_dirs:
        return [_msg(language, "root_empty", root=str(root_dir))]

    messages.append(_msg(language, "scan_start", root=str(root_dir)))
    for date_dir in date_dirs:
        messages.extend(_inspect_date_folder(date_dir, language))

    messages.append(_msg(language, "scan_done", count=len(date_dirs)))
    return messages


def _inspect_date_folder(date_dir, language):
    messages = []
    folder_name = date_dir.name

    if not DATE_FOLDER_PATTERN.match(folder_name):
        messages.append(_msg(language, "bad_date_folder", folder=folder_name))

    expected_date = _extract_folder_date(folder_name)
    files = sorted(path for path in date_dir.rglob("*") if path.is_file())
    if len(files) < 2:
        messages.append(_msg(language, "few_files", folder=folder_name, count=len(files)))

    pig_hints = []
    for file_path in files:
        file_name = file_path.name
        file_date = _extract_file_date(file_name)
        if expected_date and file_date and expected_date != file_date:
            messages.append(
                _msg(
                    language,
                    "date_mismatch",
                    folder=folder_name,
                    file=file_name,
                    folder_date=expected_date,
                    file_date=file_date,
                )
            )

        pig_hints.extend(PIG_HINT_PATTERN.findall(file_name))

    unique_hints = sorted(set(pig_hints))
    if len(unique_hints) > 1:
        messages.append(
            _msg(
                language,
                "multiple_pigs",
                folder=folder_name,
                pigs=", ".join(unique_hints[:8]),
            )
        )

    if not messages:
        messages.append(_msg(language, "folder_ok", folder=folder_name, files=len(files)))

    return messages


def _extract_folder_date(folder_name):
    match = re.match(r"^(\d{8})", folder_name)
    return match.group(1) if match else None


def _extract_file_date(file_name):
    match = DATE_IN_FILE_PATTERN.search(file_name)
    if not match:
        return None
    return match.group(1).replace("-", "")


def _msg(language, key, **kwargs):
    templates = {
        "zh": {
            "root_missing": "错误：找不到待检查文件夹：{root}",
            "root_not_dir": "错误：路径不是文件夹：{root}",
            "root_empty": "警告：文件夹内没有可检查的日期目录：{root}",
            "scan_start": "开始检查文件夹：{root}",
            "scan_done": "检查完成，共扫描 {count} 个日期目录。",
            "bad_date_folder": "警告：目录名 `{folder}` 不符合推荐格式，建议以 YYYYMMDD 开头，例如 20250612。",
            "few_files": "警告：目录 `{folder}` 内文件较少，当前仅发现 {count} 个文件，请确认数据是否完整。",
            "date_mismatch": "警告：目录 `{folder}` 的日期是 {folder_date}，但文件 `{file}` 中识别到日期 {file_date}。",
            "multiple_pigs": "警告：目录 `{folder}` 中检测到多个疑似猪只标识：{pigs}。建议一个时间段只放一头猪的数据。",
            "folder_ok": "通过：目录 `{folder}` 命名和文件日期检查正常，共发现 {files} 个文件。",
        },
        "en": {
            "root_missing": "Error: folder to inspect was not found: {root}",
            "root_not_dir": "Error: path is not a folder: {root}",
            "root_empty": "Warning: no date folders were found under: {root}",
            "scan_start": "Start inspecting folder: {root}",
            "scan_done": "Inspection finished. Scanned {count} date folders in total.",
            "bad_date_folder": "Warning: folder `{folder}` does not follow the recommended format. Use a YYYYMMDD prefix such as 20250612.",
            "few_files": "Warning: folder `{folder}` contains only {count} files. Please confirm the dataset is complete.",
            "date_mismatch": "Warning: folder `{folder}` implies date {folder_date}, but file `{file}` appears to contain date {file_date}.",
            "multiple_pigs": "Warning: folder `{folder}` appears to contain multiple pig identifiers: {pigs}. Only one pig should exist in a single time period.",
            "folder_ok": "OK: folder `{folder}` passed naming and date checks with {files} files detected.",
        },
    }
    selected = templates["en"] if language == "en" else templates["zh"]
    return selected[key].format(**kwargs)
