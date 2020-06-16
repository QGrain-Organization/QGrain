__all__ = ["ViewDataManager", "RecordedDataTable"]

import logging
from datetime import date, datetime, time
from typing import Dict, List, Tuple
from uuid import UUID

import numpy as np
from PySide2.QtCore import QCoreApplication, QEventLoop, QObject, Qt, Signal
from PySide2.QtGui import QCursor
from PySide2.QtWidgets import (QAbstractItemView, QGridLayout, QMenu,
                               QTableWidget, QTableWidgetItem, QWidget)

from QGrain.algorithms import DistributionType
from QGrain.models.FittingResult import FittingResult


class ViewDataManager(QObject):
    MAX_PARAM_COUNT = 3
    COMPONENT_SPAN = 1
    TABLE_HEADER_ROWS = 2
    TABLE_ROW_EXPAND_SETP = 50
    COMPONENT_START_COLUMN = 4
    def __init__(self, table:QTableWidget, is_detailed: bool):
        super().__init__()
        self.table = table
        self.is_detailed = is_detailed
        self.existing_key2display_name = {
            "fraction": self.tr("Fraction"),
            "mean": self.tr("Mean")+" (μm)",
            "median": self.tr("Median")+" (μm)",
            "mode": self.tr("Mode")+" (μm)",
            # "variance": self.tr("Variance"),
            "standard_deviation": self.tr("Standard Deviation"),
            "skewness": self.tr("Skewness"),
            "kurtosis": self.tr("Kurtosis")}
        self.summary_keys = ("fraction", "mean", "median")
        self.detailed_keys = tuple([key for key, display_name in self.existing_key2display_name.items()])
        self.records = [] #List[SingleViewData]

    @property
    def data_count(self) -> int:
        return len(self.records)

    @property
    def max_component_number(self) -> int:
        if self.data_count == 0:
            return 0
        else:
            return max([record.component_number for record in self.records])

    @property
    def component_columns(self) -> int:
        if self.is_detailed:
            return len(self.detailed_keys) + self.MAX_PARAM_COUNT
        else:
            return len(self.summary_keys)

    @property
    def actual_component_columns(self):
        return self.component_columns + self.COMPONENT_SPAN

    def get_distribution_name(self, distribution_type: DistributionType):
        if distribution_type == DistributionType.Normal:
            return self.tr("Normal")
        elif distribution_type == DistributionType.Weibull:
            return self.tr("Weibull")
        elif distribution_type == DistributionType.GeneralWeibull:
            return self.tr("Gen. Weibull")
        else:
            raise NotImplementedError(distribution_type)

    # the local func to handle write
    # also put data validation here
    def write(self, row, col, value, format_str="{0:0.2f}"):
        # the local func to check the value validation
        def check_value(value):
            accepted_types = (bool, datetime, date, time)
            if value is None:
                return "None"
            # must judge this first,
            # because `Formula` can pass `numpy.isreal` but raise error in `numpy.isnan`
            elif type(value) in accepted_types:
                return value
            # if value is real number (i.e. int, float, numpy.int32, numpy.float32, etc.)
            elif np.isreal(value):
                # this func will both recognize float("nan") and numpy.nan
                if np.isnan(value):
                    return "NaN"
                # similar to above
                elif np.isinf(value):
                    return "Inf"
                else:
                    return format_str.format(value)
            # if the type of value is unaccepted, use its readable str to write
            else:
                return str(value)

        item = QTableWidgetItem(check_value(value))
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, col, item)


    def update_headers(self):
        max_component_number = self.max_component_number
        # update headers
        self.write(0, 0, self.tr("Sample Name"))
        self.table.setSpan(0, 0, self.TABLE_HEADER_ROWS, 1)
        self.write(0, 1, self.tr("Distribution Type"))
        self.table.setSpan(0, 1, self.TABLE_HEADER_ROWS, 1)
        self.write(0, 2, self.tr("Component Number"))
        self.table.setSpan(0, 2, self.TABLE_HEADER_ROWS, 1)
        self.write(0, 3, self.tr("Mean Squared Error"))
        self.table.setSpan(0, 3, self.TABLE_HEADER_ROWS, 1)
        for comp_index in range(max_component_number):
            first_column_index = comp_index*self.actual_component_columns+self.COMPONENT_START_COLUMN
            self.write(0, first_column_index, self.tr("Component {0}").format(comp_index+1))
            self.table.setSpan(0, first_column_index, 1, self.component_columns)
            if self.is_detailed:
                headers = list((self.existing_key2display_name[key] for key in self.detailed_keys))
                for i in range(self.MAX_PARAM_COUNT):
                    headers.append(self.tr("Parameter {0}").format(i+1))
            else:
                headers = list((self.existing_key2display_name[key] for key in self.summary_keys))
            for i, header in enumerate(headers):
                self.write(1, first_column_index+i, header)

    def add_records(self, records_to_add: List[FittingResult]):
        if records_to_add is None or len(records_to_add) == 0:
            return
        length = len(records_to_add)
        # if it's necessary to expand the rows and columns of table
        least_row_number = self.data_count + length + self.TABLE_HEADER_ROWS
        if  least_row_number>= self.table.rowCount():
            self.table.setRowCount(least_row_number + self.TABLE_ROW_EXPAND_SETP)
        max_component_number_before = self.max_component_number
        max_component_number_new = max([record.component_number for record in records_to_add])
        # reset the column number of table
        column_count = max(max_component_number_before, max_component_number_new) * self.actual_component_columns + self.COMPONENT_START_COLUMN
        self.table.setColumnCount(column_count)

        # update contents
        first_row_index = self.data_count + self.TABLE_HEADER_ROWS
        for record_index, record in enumerate(records_to_add):
            # will bring thread unsafe issue
            # QCoreApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            row = first_row_index + record_index
            self.write(row, 0, record.name)
            self.write(row, 1, self.get_distribution_name(record.distribution_type))
            self.write(row, 2, record.component_number, format_str="{0:d}")
            self.write(row, 3, record.mean_squared_error, format_str="{0:0.2E}")
            if self.is_detailed:
                keys = self.detailed_keys
            else:
                keys = self.summary_keys
            for i, component in enumerate(record.components):
                last_column = 0
                for key_index, key in enumerate(keys):
                    column_index = i*self.actual_component_columns + self.COMPONENT_START_COLUMN + key_index
                    last_column = column_index
                    self.write(row, column_index, component.__getattribute__(key))
                if self.is_detailed:
                    for param_index, param_value in enumerate(component.params):
                        self.write(row, last_column + param_index + 1, param_value)
            # store the records
            self.records.append(record)

        # means there is greater component number and the headers are need to be updated
        if max_component_number_new > max_component_number_before:
            self.update_headers()
            self.table.resizeColumnsToContents()


    def remove_selected_records(self) -> List[Tuple[UUID, str]]:
        # get rows to remove
        rows_to_remove = set()
        # The behaviour of `selectedRanges` differs when clicking in table or at the edge
        # When clicking in table it returns multi ranges (each row)
        # When clicking at the edge it returns a single range
        for item in self.table.selectedRanges():
            for i in range(item.topRow(), min(self.data_count+self.TABLE_HEADER_ROWS, item.bottomRow()+1)):
                # do not remove headers
                if i >= self.TABLE_HEADER_ROWS:
                    rows_to_remove.add(i)
        rows_to_remove = list(rows_to_remove)
        rows_to_remove.sort()

        uuids_and_names_to_remove = []
        offset = 0
        for row in rows_to_remove:
            self.table.removeRow(row-offset)
            removed_view_data = self.records.pop(row-offset-self.TABLE_HEADER_ROWS)
            uuids_and_names_to_remove.append((removed_view_data.uuid, removed_view_data.name))
            offset += 1

        # reset the column number of table
        column_count = self.max_component_number * self.actual_component_columns + self.COMPONENT_START_COLUMN
        self.table.setColumnCount(column_count)

        return uuids_and_names_to_remove

    def get_selected_record(self) -> FittingResult:
        ranges = self.table.selectedRanges()
        if len(ranges) != 0:
            row = ranges[0].topRow()
            return self.records[row-self.TABLE_HEADER_ROWS]


class RecordedDataTable(QWidget):
    logger = logging.getLogger("root.ui.RecordedDataTable")
    gui_logger = logging.getLogger("GUI")
    sigRemoveRecords = Signal(list)
    sigShowDistribution = Signal(FittingResult)
    sigShowLoss = Signal(FittingResult)
    sigGenerateDistributionVideo = Signal(FittingResult)
    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.main_layout.addWidget(self.table)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.manager = ViewDataManager(self.table, is_detailed=True)
        self.menu = QMenu(self.table)
        self.remove_action = self.menu.addAction(self.tr("Remove"))
        self.remove_action.triggered.connect(self.remove_selection)
        self.show_distribution_action = self.menu.addAction(self.tr("Show Distribution"))
        self.show_distribution_action.triggered.connect(self.show_distribution)
        self.generate_distribution_video_action = self.menu.addAction(self.tr("Generate Distribution Video"))
        self.generate_distribution_video_action.triggered.connect(self.generate_distribution_video)
        self.show_loss_action = self.menu.addAction(self.tr("Show Loss"))
        self.show_loss_action.triggered.connect(self.show_loss)
        self.table.customContextMenuRequested.connect(self.show_menu)
        self.table.cellClicked.connect(self.on_data_item_clicked)

    def show_menu(self, pos):
        self.menu.popup(QCursor.pos())

    def on_data_item_clicked(self, row, column):
        self.show_distribution()

    def on_data_recorded(self, results: List[FittingResult]):
        self.manager.add_records(results)
        self.logger.debug("Recorded data has been updated in table widget.")

    def remove_selection(self):
        uuids_and_names = self.manager.remove_selected_records()
        self.logger.debug("The selected records below will be removed: [%s].", uuids_and_names)
        self.sigRemoveRecords.emit(uuids_and_names)

    def show_distribution(self):
        record = self.manager.get_selected_record()
        if record is not None:
            self.sigShowDistribution.emit(record)

    def show_loss(self):
        record = self.manager.get_selected_record()
        if record is not None:
            self.sigShowLoss.emit(record)

    def generate_distribution_video(self):
        record = self.manager.get_selected_record()
        if record is not None:
            self.sigGenerateDistributionVideo.emit(record)
