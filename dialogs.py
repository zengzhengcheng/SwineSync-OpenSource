from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from translations import t


class FileConfirmationDialog(QDialog):
    def __init__(self, filepaths, language="zh", parent=None):
        super().__init__(parent)
        self.language = language
        self.file_boxes = {}
        self.setWindowTitle(t(language, "select_files_title"))
        self.setMinimumSize(700, 420)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel(t(language, "select_hint")))

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        self.select_all = QCheckBox(t(language, "select_all"))
        self.select_all.setChecked(True)
        self.select_all.stateChanged.connect(self.toggle_all_selection)
        content_layout.addWidget(self.select_all)

        for path in filepaths:
            checkbox = QCheckBox(path)
            checkbox.setChecked(True)
            self.file_boxes[path] = checkbox
            content_layout.addWidget(checkbox)

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def toggle_all_selection(self, state):
        checked = state == Qt.CheckState.Checked.value
        for checkbox in self.file_boxes.values():
            checkbox.setChecked(checked)

    def selected_files(self):
        return [path for path, checkbox in self.file_boxes.items() if checkbox.isChecked()]
