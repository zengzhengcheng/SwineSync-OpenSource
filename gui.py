import json
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QProcess
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config_manager import load_config, save_config
from dialogs import FileConfirmationDialog
from translations import t


def collect_ecg_source_files(folder_path):
    file_map = {}
    for path in Path(folder_path).rglob("*"):
        if path.is_file() and (path.name.startswith("Raw") or ("BMD" in path.name)):
            file_map[path.name] = str(path.resolve())
    return list(file_map.values())


class OpenSourceMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.process = None
        self.current_success_message = ""
        self.current_task_file = None
        self.config = load_config()
        self.language = self.config["language"]
        self._build_ui()
        self.apply_config_to_widgets()
        self.update_texts()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(14)

        self.title_label = QLabel()
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        self.subtitle_label = QLabel()
        self.subtitle_label.setStyleSheet("color: #5f6b7a;")

        self.notes_group = QGroupBox()
        notes_layout = QVBoxLayout(self.notes_group)
        self.notes_label = QLabel()
        self.notes_label.setWordWrap(True)
        self.notes_label.setStyleSheet("color: #4a5568;")
        notes_layout.addWidget(self.notes_label)

        root.addWidget(self.title_label)
        root.addWidget(self.subtitle_label)
        root.addWidget(self.notes_group)

        self.config_group = QGroupBox()
        form = QFormLayout(self.config_group)
        self.batch_label = QLabel()
        self.cache_label = QLabel()
        self.interpolation_label = QLabel()
        self.batch_size_edit = QLineEdit()
        self.cache_size_edit = QLineEdit()
        self.interpolation_combo = QComboBox()
        self.correct_check = QCheckBox()
        self.save_config_button = QPushButton()
        self.language_button = QPushButton()
        self.save_config_button.clicked.connect(self.save_current_config)
        self.language_button.clicked.connect(self.toggle_language)
        form.addRow(self.batch_label, self.batch_size_edit)
        form.addRow(self.cache_label, self.cache_size_edit)
        form.addRow(self.interpolation_label, self.interpolation_combo)
        form.addRow("", self.correct_check)
        button_row = QHBoxLayout()
        button_row.addWidget(self.save_config_button)
        button_row.addWidget(self.language_button)
        form.addRow(button_row)
        root.addWidget(self.config_group)

        self.ecg_group = QGroupBox()
        ecg_layout = QHBoxLayout(self.ecg_group)
        self.btn_convert = QPushButton()
        self.btn_convert_cache = QPushButton()
        self.btn_convert_only_cache = QPushButton()
        self.btn_convert.clicked.connect(lambda: self.run_convert(use_cache=False))
        self.btn_convert_cache.clicked.connect(lambda: self.run_convert(use_cache=True))
        self.btn_convert_only_cache.clicked.connect(self.run_cache_only)
        ecg_layout.addWidget(self.btn_convert)
        ecg_layout.addWidget(self.btn_convert_cache)
        ecg_layout.addWidget(self.btn_convert_only_cache)
        root.addWidget(self.ecg_group)

        self.tools_group = QGroupBox()
        tools_layout = QHBoxLayout(self.tools_group)
        self.btn_heart_process = QPushButton()
        self.btn_folder_inspect = QPushButton()
        self.btn_heart_process.clicked.connect(self.run_heart_process)
        self.btn_folder_inspect.clicked.connect(self.run_folder_inspect)
        tools_layout.addWidget(self.btn_heart_process)
        tools_layout.addWidget(self.btn_folder_inspect)
        root.addWidget(self.tools_group)

        self.move_group = QGroupBox()
        move_layout = QHBoxLayout(self.move_group)
        self.btn_move_old = QPushButton()
        self.btn_move_new = QPushButton()
        self.btn_move_old.clicked.connect(self.run_move_old)
        self.btn_move_new.clicked.connect(self.run_move_new)
        move_layout.addWidget(self.btn_move_old)
        move_layout.addWidget(self.btn_move_new)
        root.addWidget(self.move_group)

        self.log_group = QGroupBox()
        log_layout = QVBoxLayout(self.log_group)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.clear_log_button = QPushButton()
        self.clear_log_button.clicked.connect(self.log_edit.clear)
        log_layout.addWidget(self.log_edit)
        log_layout.addWidget(self.clear_log_button)
        root.addWidget(self.log_group, 1)

    def closeEvent(self, event):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(2000)
        super().closeEvent(event)

    def apply_config_to_widgets(self):
        self.batch_size_edit.setText(str(self.config["batch_size"]))
        self.cache_size_edit.setText(str(self.config["cache_size"]))
        self._populate_interpolation_combo()
        algorithm = self.config["interpolation_algorithm"]
        index = self.interpolation_combo.findData(algorithm)
        self.interpolation_combo.setCurrentIndex(max(index, 0))
        self.correct_check.setChecked(self.config["correct"])

    def update_texts(self):
        self.setWindowTitle(t(self.language, "window_title"))
        self.title_label.setText(t(self.language, "title"))
        self.subtitle_label.setText(t(self.language, "subtitle"))
        self.notes_group.setTitle(t(self.language, "notes_title"))
        self.notes_label.setText(t(self.language, "notes_body"))
        self.config_group.setTitle(t(self.language, "config_group"))
        self.batch_label.setText(t(self.language, "batch_size"))
        self.cache_label.setText(t(self.language, "cache_size"))
        self.interpolation_label.setText(t(self.language, "interpolation_algorithm"))
        self.correct_check.setText(t(self.language, "correct"))
        self.save_config_button.setText(t(self.language, "save_config"))
        self.language_button.setText(t(self.language, "toggle_language"))
        self.ecg_group.setTitle(t(self.language, "ecg_group"))
        self.btn_convert.setText(t(self.language, "convert"))
        self.btn_convert_cache.setText(t(self.language, "convert_cache"))
        self.btn_convert_only_cache.setText(t(self.language, "convert_only_cache"))
        self.tools_group.setTitle(t(self.language, "tools_group"))
        self.btn_heart_process.setText(t(self.language, "heart_process"))
        self.btn_folder_inspect.setText(t(self.language, "folder_inspect"))
        self.move_group.setTitle(t(self.language, "move_group"))
        self.btn_move_old.setText(t(self.language, "move_old"))
        self.btn_move_new.setText(t(self.language, "move_new"))
        self.log_group.setTitle(t(self.language, "log_group"))
        self.clear_log_button.setText(t(self.language, "clear_log"))
        self._populate_interpolation_combo()

    def _populate_interpolation_combo(self):
        current_value = self.interpolation_combo.currentData()
        self.interpolation_combo.blockSignals(True)
        self.interpolation_combo.clear()
        self.interpolation_combo.addItem(t(self.language, "algo_original_timestamps"), "original_timestamps")
        self.interpolation_combo.addItem(t(self.language, "algo_uniform_grid"), "uniform_grid")
        if current_value is not None:
            index = self.interpolation_combo.findData(current_value)
            if index >= 0:
                self.interpolation_combo.setCurrentIndex(index)
        self.interpolation_combo.blockSignals(False)

    def append_log(self, text):
        cursor = self.log_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_edit.setTextCursor(cursor)
        self.log_edit.ensureCursorVisible()

    def toggle_language(self):
        self.language = "en" if self.language == "zh" else "zh"
        self.config["language"] = self.language
        self.update_texts()
        save_config(self.config)

    def save_current_config(self):
        try:
            self.config = self._read_config_from_widgets()
        except ValueError as exc:
            QMessageBox.warning(self, t(self.language, "param_error_title"), str(exc))
            return
        self.config["language"] = self.language
        save_config(self.config)
        QMessageBox.information(self, t(self.language, "done_title"), t(self.language, "config_saved"))

    def _read_config_from_widgets(self):
        try:
            batch_size = int(self.batch_size_edit.text().strip())
            cache_size = int(self.cache_size_edit.text().strip())
        except ValueError:
            raise ValueError(t(self.language, "param_int_error"))
        if batch_size <= 0 or cache_size <= 0:
            raise ValueError(t(self.language, "param_positive_error"))
        return {
            "language": self.language,
            "batch_size": batch_size,
            "cache_size": cache_size,
            "interpolation_algorithm": self.interpolation_combo.currentData(),
            "correct": self.correct_check.isChecked(),
        }

    def _select_ecg_files(self):
        source_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_ecg_dir"), "./")
        if not source_dir:
            return None, None
        filepaths = collect_ecg_source_files(source_dir)
        if not filepaths:
            QMessageBox.information(self, t(self.language, "no_files_title"), t(self.language, "no_files_found"))
            return None, None
        dialog = FileConfirmationDialog(filepaths, self.language, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return None, None
        selected = dialog.selected_files()
        if not selected:
            QMessageBox.information(self, t(self.language, "no_files_title"), t(self.language, "no_files_selected"))
            return None, None
        return source_dir, selected

    def _run_task_process(self, task_payload, success_message=""):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, t(self.language, "param_error_title"), t(self.language, "running_task"))
            return

        runtime_dir = Path(__file__).resolve().parent / "runtime"
        runtime_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="task_",
            dir=runtime_dir,
            delete=False,
        ) as fp:
            json.dump(task_payload, fp, ensure_ascii=False, indent=2)
            self.current_task_file = fp.name

        script_path = str(Path(__file__).resolve().parent / "cli_runner.py")
        self.current_success_message = success_message
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments([script_path, self.current_task_file])
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_stdout)
        self.process.readyReadStandardError.connect(self._read_process_stderr)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)
        self._set_buttons_enabled(False)
        self.append_log(f"\n{t(self.language, 'task_running')}\n")
        self.process.start()

    def _read_process_stdout(self):
        if not self.process:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.append_log(text)

    def _read_process_stderr(self):
        if not self.process:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self.append_log(text)

    def _cleanup_task_file(self):
        if self.current_task_file:
            try:
                Path(self.current_task_file).unlink(missing_ok=True)
            except OSError:
                pass
            self.current_task_file = None

    def _on_process_finished(self, exit_code, exit_status):
        self._set_buttons_enabled(True)
        self._cleanup_task_file()
        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            if self.current_success_message:
                self.append_log(f"\n{self.current_success_message}\n")
                QMessageBox.information(self, t(self.language, "done_title"), self.current_success_message)
        else:
            message = f"{t(self.language, 'task_exit_error')}{exit_code}"
            self.append_log(f"\n{message}\n")
            QMessageBox.critical(self, t(self.language, "failed_title"), message)
        self.process = None

    def _on_process_error(self, _error):
        self._set_buttons_enabled(True)
        self._cleanup_task_file()
        self.append_log(f"\n{t(self.language, 'process_start_failed')}\n")
        QMessageBox.critical(self, t(self.language, "failed_title"), t(self.language, "process_start_failed"))
        self.process = None

    def _set_buttons_enabled(self, enabled):
        for button in [
            self.btn_convert,
            self.btn_convert_cache,
            self.btn_convert_only_cache,
            self.btn_heart_process,
            self.btn_folder_inspect,
            self.btn_move_old,
            self.btn_move_new,
            self.save_config_button,
            self.language_button,
        ]:
            button.setEnabled(enabled)

    def run_convert(self, use_cache):
        try:
            config = self._read_config_from_widgets()
        except ValueError as exc:
            QMessageBox.warning(self, t(self.language, "param_error_title"), str(exc))
            return
        _, selected_files = self._select_ecg_files()
        if not selected_files:
            return
        output_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_output_dir"), "./")
        if not output_dir:
            return
        task_name = t(self.language, "task_convert_cache") if use_cache else t(self.language, "task_convert")
        self.append_log(f"\n{task_name}\n")
        self._run_task_process(
            {
                "task_type": "convert_cache" if use_cache else "convert",
                "config": config,
                "selected_files": selected_files,
                "output_dir": output_dir,
                "language": self.language,
            },
            success_message=f"{task_name}{t(self.language, 'task_done_suffix')}",
        )

    def run_cache_only(self):
        try:
            config = self._read_config_from_widgets()
        except ValueError as exc:
            QMessageBox.warning(self, t(self.language, "param_error_title"), str(exc))
            return
        _, selected_files = self._select_ecg_files()
        if not selected_files:
            return
        output_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_cache_dir"), "./")
        if not output_dir:
            return
        task_name = t(self.language, "task_only_cache")
        self.append_log(f"\n{task_name}\n")
        self._run_task_process(
            {
                "task_type": "cache_only",
                "config": config,
                "selected_files": selected_files,
                "output_dir": output_dir,
                "language": self.language,
            },
            success_message=f"{task_name}{t(self.language, 'task_done_suffix')}",
        )

    def run_heart_process(self):
        source_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_heart_dir"), "./")
        if not source_dir:
            return
        picture_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_picture_dir"), "./")
        if not picture_dir:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, t(self.language, "save_heart"), "", t(self.language, "csv_filter"))
        if not save_path:
            return
        task_name = t(self.language, "task_heart_process")
        self.append_log(f"\n{task_name}\n")
        self._run_task_process(
            {
                "task_type": "heart_process",
                "source_dir": source_dir,
                "picture_dir": picture_dir,
                "save_path": save_path,
                "language": self.language,
            },
            success_message=f"{task_name}{t(self.language, 'task_done_suffix')}",
        )

    def run_folder_inspect(self):
        source_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_move_dir"), "./")
        if not source_dir:
            return
        task_name = t(self.language, "task_folder_inspect")
        self.append_log(f"\n{task_name}\n")
        self._run_task_process(
            {
                "task_type": "folder_inspect",
                "source_dir": source_dir,
                "language": self.language,
            },
            success_message=f"{task_name}{t(self.language, 'task_done_suffix')}",
        )

    def run_move_old(self):
        source_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_move_dir"), "./")
        if not source_dir:
            return
        picture_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_picture_dir"), "./")
        if not picture_dir:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, t(self.language, "save_move_old"), "", t(self.language, "csv_filter"))
        if not save_path:
            return
        task_name = t(self.language, "task_move_old")
        self.append_log(f"\n{task_name}\n")
        self._run_task_process(
            {
                "task_type": "move_old",
                "source_dir": source_dir,
                "picture_dir": picture_dir,
                "save_path": save_path,
                "language": self.language,
            },
            success_message=f"{task_name}{t(self.language, 'task_done_suffix')}",
        )

    def run_move_new(self):
        source_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_move_dir"), "./")
        if not source_dir:
            return
        picture_dir = QFileDialog.getExistingDirectory(self, t(self.language, "select_picture_dir"), "./")
        if not picture_dir:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, t(self.language, "save_move_new"), "", t(self.language, "csv_filter"))
        if not save_path:
            return
        task_name = t(self.language, "task_move_new")
        self.append_log(f"\n{task_name}\n")
        self._run_task_process(
            {
                "task_type": "move_new",
                "source_dir": source_dir,
                "picture_dir": picture_dir,
                "save_path": save_path,
                "language": self.language,
            },
            success_message=f"{task_name}{t(self.language, 'task_done_suffix')}",
        )


def launch():
    app = QApplication(sys.argv)
    window = OpenSourceMainWindow()
    window.show()
    sys.exit(app.exec())
