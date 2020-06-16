__all__ = ["RawDataTable"]

import logging
from typing import Dict, List, Tuple
from uuid import UUID

import numpy as np
from PySide2.QtCore import QCoreApplication, QEventLoop, QObject, Qt, Signal
from PySide2.QtGui import QCursor
from PySide2.QtWidgets import (QAbstractItemView, QComboBox, QGridLayout,
                               QLabel, QMenu, QPushButton, QTableWidget,
                               QTableWidgetItem, QWidget)

from QGrain.models.SampleData import SampleData
from QGrain.models.SampleDataset import SampleDataset


class RawDataTable(QWidget):
    logger = logging.getLogger("root.ui.RawDataTable")
    gui_logger = logging.getLogger("GUI")
    show_distribution_signal = Signal(UUID)
    perform_sample_signal = Signal(UUID)

    def __init__(self, parent=None, sample_number_each_page=20):
        super().__init__(parent)
        self.sample_number_each_page = sample_number_each_page
        self.dataset = None  # type: SampleDataset
        self.init_ui()

    def init_ui(self):
        self.data_table = QTableWidget(100, 100)
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_table.setAlternatingRowColors(True)
        # self.data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_table.hideColumn(0)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.data_table, 0, 0, 1, 3)

        self.previous_button = QPushButton(self.tr("Previous"))
        self.previous_button.setToolTip(self.tr("Click to back to the previous page."))
        self.previous_button.clicked.connect(self.on_previous_button_clicked)
        self.current_page_combo_box = QComboBox()
        self.current_page_combo_box.currentIndexChanged.connect(self.on_page_changed)
        self.next_button = QPushButton(self.tr("Next"))
        self.next_button.setToolTip(self.tr("Click to jump to the next page."))
        self.next_button.clicked.connect(self.on_next_button_clicked)
        self.main_layout.addWidget(self.previous_button, 1, 0)
        self.main_layout.addWidget(self.current_page_combo_box, 1, 1)
        self.main_layout.addWidget(self.next_button, 1, 2)

        # self.menu = QMenu(self.data_table)
        # self.show_distribution_action = self.menu.addAction(self.tr("Show Distribution"))
        # self.show_distribution_action.triggered.connect(self.show_distribution)
        # self.perform_sample_action = self.menu.addAction(self.tr("Perform"))
        # self.perform_sample_action.triggered.connect(self.perform_sample)
        # self.data_table.customContextMenuRequested.connect(self.show_menu)

    def show_menu(self, pos):
        self.menu.popup(QCursor.pos())

    def on_data_loaded(self, dataset: SampleDataset):
        self.dataset = dataset
        self.current_page_combo_box.clear()
        page_count, left = divmod(self.dataset.data_count, self.sample_number_each_page)
        if left != 0:
            page_count += 1
        self.current_page_combo_box.addItems([self.tr("Page") + str(i+1) for i in range(page_count)])
        self.on_page_changed(0)

    def on_page_changed(self, page_index: int):
        def write(row: int, col: int, value: str):
            assert isinstance(value, str)
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter)
            self.data_table.setItem(row, col, item)

        # necessary to clear
        self.data_table.clear()

        nrows = self.dataset.data_count
        ncols = len(self.dataset.classes)
        self.data_table.setRowCount(self.sample_number_each_page+1)
        self.data_table.setColumnCount(ncols+2)
        # hide the first column
        self.data_table.setHorizontalHeaderLabels(["HIDEN"]+[str(i+1) for i in range(ncols+1)])
        # use the hiden column (index = 0) to store the uuid string
        # headers
        write(0, 0, "UUID")
        write(0, 1, self.tr("Sample Name"))
        for col, class_value in enumerate(self.dataset.classes, 2):
            write(0, col, f"{class_value:.2f}")

        start = page_index * self.sample_number_each_page
        end = (page_index+1) * self.sample_number_each_page
        for i, sample in enumerate(self.dataset.samples):
            if i < start:
                continue
            elif i >= end:
                break
            else:
                row = i - start + 1
                write(row, 0, sample.uuid.hex)
                write(row, 1, str(sample.name))
                for col, value in enumerate(sample.distribution, 2):
                    write(row, col, f"{value:.4f}")
        self.data_table.resizeColumnsToContents()

    def get_selections(self):
        uuids = set()
        for item in self.data_table.selectedRanges():
            for i in range(item.topRow(), min(self.sample_number_each_page+1, item.bottomRow()+1)):
                # do not remove the header
                if i > 0:
                    uuids.add(UUID(self.data_table.item(i, 0).text()))
        uuids = list(uuids)
        uuids.sort()
        return uuids

    def on_previous_button_clicked(self):
        current_index = self.current_page_combo_box.currentIndex()
        if current_index > 0:
            self.current_page_combo_box.setCurrentIndex(current_index-1)
        else:
            pass

    def on_next_button_clicked(self):
        current_index = self.current_page_combo_box.currentIndex()
        if current_index < self.current_page_combo_box.count() - 1:
            self.current_page_combo_box.setCurrentIndex(current_index+1)
        else:
            pass

    def show_distribution(self):
        uuids = self.get_selections()
        if len(uuids) == 0:
            return
        self.show_distribution_signal.emit(uuids[0])

    def perform_sample(self):
        uuids = self.get_selections()
        if len(uuids) == 0:
            return
        self.perform_sample_signal.emit(uuids[0])


if __name__ == "__main__":
    from QGrain.generate_test_dataset import get_random_sample
    sample_classes = np.logspace(0, 5, 101) * 0.02
    sample_names = []
    sample_distributions = []
    sample_params = ((1.1, 6, 0.1), (8.0, 9.0, 0.5), (30.0, 4.5, 0.4))
    sample_max_scale_ratios = ((0.2, 0.2, 0.4), (0.4, 0.4, 0.6), (0.4, 0.4, 0.6))
    for i in range(321):
        distribution, _, _ = get_random_sample(sample_classes, sample_params, sample_max_scale_ratios)
        sample_names.append(f"TestSample{i}")
        sample_distributions.append(distribution)

    dataset = SampleDataset()
    dataset.add_batch(sample_classes, sample_names, sample_distributions)

    import sys
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    table = RawDataTable()
    table.show()
    table.on_data_loaded(dataset)
    sys.exit(app.exec_())
