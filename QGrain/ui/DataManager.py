__all__ = ["BackgroundLoader", "BackgroundWriter", "DataManager"]

import logging
import os
import pickle
from typing import Callable, Dict, Iterable, List, Tuple
from uuid import UUID

import numpy as np
from PySide2.QtCore import QObject, QStandardPaths, Qt, QThread, Signal
from PySide2.QtWidgets import QFileDialog, QMessageBox, QWidget

from QGrain.algorithms import DistributionType
from QGrain.models.DataLayoutSettings import *
from QGrain.models.DataLoader import *
from QGrain.models.DataWriter import *
from QGrain.models.FittingResult import *
from QGrain.models.SampleData import *
from QGrain.models.SampleDataset import *
from QGrain.resolvers.HeadlessResolver import FittingTask
from QGrain.resolvers.MultiprocessingResolver import ProcessState


class BackgroundLoader(QObject):
    sigWorkSucceeded = Signal(SampleDataset)
    sigWorkFailed = Signal(str, Exception)

    def __init__(self):
        super().__init__()
        self.actual_loader = DataLoader()

    def on_work_started(self, filename: str, file_type: FileType,
                        layout: DataLayoutSettings):
        try:
            dataset = self.actual_loader.try_load_data(filename, file_type, layout)
            self.sigWorkSucceeded.emit(dataset)
        except Exception as e:
            self.sigWorkFailed.emit(filename, e)


class BackgroundWriter(QObject):
    sigWorkSucceeded = Signal()
    sigWorkFailed = Signal(str, Exception)

    def __init__(self):
        super().__init__()
        self.actual_writer = DataWriter()

    def on_work_started(self, filename: str, file_type: FileType,
                        results: List[FittingResult], draw_charts: bool):
        try:
            self.actual_writer.try_save_data(filename, file_type, results, draw_charts)
            self.sigWorkSucceeded.emit()
        except Exception as e:
            self.sigWorkFailed.emit(filename, e)


class DataManager(QObject):
    sigLoadingStarted = Signal(str, FileType, DataLayoutSettings)
    sigSavingStarted = Signal(str, FileType, list, bool)
    sigDataLoaded = Signal(SampleDataset)
    sigTargetDataChanged = Signal(SampleData)
    sigDataRecorded = Signal(list) # List[FittingResult]
    sigShowFittingResult = Signal(FittingResult)
    logger = logging.getLogger("root.data.DataManager")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, host_widget: QWidget):
        super().__init__()
        # to attach msg boxed on this widget
        self.host_widget = host_widget
        # data
        self.dataset = None  # type: SampleDataset
        self.current_fitting_result = None  # type: FittingResult
        self.records = []  # type: List[FittingResult]
        # loader
        self.loader = BackgroundLoader()
        self.loading_thread = QThread()
        self.loader.moveToThread(self.loading_thread)
        self.sigLoadingStarted.connect(self.loader.on_work_started)
        self.loader.sigWorkSucceeded.connect(self.on_loading_succeeded)
        self.loader.sigWorkFailed.connect(self.on_loading_failed)
        # writer
        self.writer = BackgroundWriter()
        self.saving_thread = QThread()
        self.writer.moveToThread(self.saving_thread)
        self.sigSavingStarted.connect(self.writer.on_work_started)
        self.writer.sigWorkSucceeded.connect(self.on_saving_succeeded)
        self.writer.sigWorkFailed.connect(self.on_saving_failed)
        # settings
        self.data_layout_setting = DataLayoutSettings()
        self.draw_charts_flag = True
        self.auto_record_flag = True

        self.file_dialog = QFileDialog(self.host_widget)
        self.msg_box = QMessageBox(self.host_widget)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.retry_msg_box = QMessageBox(self.host_widget)
        self.retry_msg_box.addButton(QMessageBox.StandardButton.Retry)
        self.retry_msg_box.addButton(QMessageBox.StandardButton.Ok)
        self.retry_msg_box.setDefaultButton(QMessageBox.StandardButton.Retry)
        self.retry_msg_box.setWindowFlags(Qt.Drawer)
        self.record_msg_box = QMessageBox(self.host_widget)
        self.record_msg_box.addButton(QMessageBox.StandardButton.Discard)
        self.record_msg_box.addButton(QMessageBox.StandardButton.Ok)
        self.record_msg_box.setDefaultButton(QMessageBox.StandardButton.Discard)
        self.record_msg_box.setWindowFlags(Qt.Drawer)

    def show_message(self, title: str, message: str):
        self.msg_box.setWindowTitle(title)
        self.msg_box.setText(message)
        self.msg_box.exec_()

    def show_info(self, message: str):
        self.show_message(self.tr("Info"), message)

    def show_warning(self, message: str):
        self.show_message(self.tr("Warning"), message)

    def show_error(self, message: str):
        self.show_message(self.tr("Error"), message)

    def ask_retry(self, message: str, func: Callable):
        self.retry_msg_box.setWindowTitle(self.tr("Error"))
        self.retry_msg_box.setText(message)
        result = self.retry_msg_box.exec_()
        if result == QMessageBox.Retry:
            func()

    def load_data(self):
        # NOTE:
        # if don't assign the initial directory,
        # there is an about 5s delay while first calling `getOpenFileName` func
        # if there is a unreachable network disk
        init_path = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)[0]
        filename, type_str = self.file_dialog.getOpenFileName(
            self.host_widget, self.tr("Select Data File"),
            init_path, "Excel (*.xlsx);;97-2003 Excel (*.xls);;CSV (*.csv)")
        if filename is None or filename == "":
            self.logger.info("The user did not select a file, ignored.")
            return
        # get the file type info
        if ".xlsx" in type_str:
            file_type = FileType.XLSX
        elif "97-2003" in type_str:
            file_type = FileType.XLS
        elif ".csv" in type_str:
            file_type = FileType.CSV
        else:
            raise NotImplementedError(type_str)

        self.logger.info("Try to load data, selected data file is [%s].", filename)
        self.sigLoadingStarted.emit(filename, file_type, self.data_layout_setting)

    def on_loading_succeeded(self, dataset: SampleDataset):
        if self.dataset is None:
            self.dataset = dataset
        else:
            try:
                self.dataset.combine(dataset)
            except ClassesNotMatchError:
                self.logger.exception("The new batch of data's classes are not equal to the existing's.", stack_info=True)
                self.ask_retry(self.tr("The new batch of data's classes are not equal to the existing's."), self.load_data)
                return
        self.sigDataLoaded.emit(self.dataset)
        self.logger.info("Data has been loaded.")
        self.show_info(self.tr("Data has been loaded."))

    def on_loading_failed(self, filename: str, exception: Exception):
        try:
            raise exception
        # handle the exceptions
        except PermissionError:
            self.logger.exception("Can not access the selected file due to permission issue. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("Can not access the selected file due to permission issue, please check whether it's opened by another program."), self.load_data)
        except FileNotFoundError:
            self.logger.exception("The selected file does not exist. Maybe it's removed by others. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("The selected file does not exist now, please check whether it's removed."), self.load_data)
        except XLRDError:
            self.logger.exception("A exception raised while reading Excel file. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("Can not read data from this Excel file."), self.load_data)
        except CSVEncodingError:
            self.logger.exception("The encoding of this CSV file is not utf-8. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("Please make sure the encoding of CSV file is utf-8."), self.load_data)
        except ValueNotNumberError:
            self.logger.exception("Can not convert the value to real number. Data layout setting: [%s].", self.data_layout_setting, stack_info=True)
            self.ask_retry(self.tr("The value can not be converted to real number.\nPlease make sure the data layout setting matchs your data file."), self.load_data)
        except ClassesNotIncrementalError:
            self.logger.exception("The classes values are not incremental. Filename: [%s]. Data layout setting: [%s].", filename, self.data_layout_setting, stack_info=True)
            self.ask_retry(self.tr("The array of grain size classes is not incremental.\nPlease make sure the data file is correct and the data layout setting matchs the file."), self.load_data)
        except ClassesNotMatchError:
            self.logger.exception("The new batch of data's classes are not equal to the existing's.", stack_info=True)
            self.ask_retry(self.tr("The new batch of data's classes are not equal to the existing's."), self.load_data)
        except NaNError:
            self.logger.exception("There is a NaN value in this data file.", stack_info=True)
            self.ask_retry(self.tr("There is a NaN value in this data file."), self.load_data)
        except ArrayEmptyError:
            self.logger.exception("There is an array that is empty.", stack_info=True)
            self.ask_retry(self.tr("There is an array that is empty.\nIf you are sure that the CSV file and layout setting are correct, please check the dialect of CSV is excel, NOT excel-tab."), self.load_data)
        except SampleNameEmptyError:
            self.logger.exception("At leaset one sample name is empty.", stack_info=True)
            self.ask_retry(self.tr("At leaset one sample name is empty. Please give each sample a unique name to identify."), self.load_data)
        except DistributionSumError:
            self.logger.exception("The sum of distribution array is not equal to 1 or 100.", stack_info=True)
            self.ask_retry(self.tr("The sum of distribution array is not equal to 1 or 100. Please make sure the data file you selected is grain size distribution data."), self.load_data)
        except Exception:
            self.logger.exception("Unknown exception raised.", stack_info=True)
            self.ask_retry(self.tr("Unknown exception raised. See the log for more details."), self.load_data)

    def on_focus_sample_changed(self, sample_id: UUID):
        if self.dataset is None:
            self.logger.info("Grain size data is still None, ignored.")
            return
        self.sigTargetDataChanged.emit(self.dataset.get_sample_by_id(sample_id))
        self.logger.debug("Focus sample data changed, the data has been emitted.")

    def on_fitting_suceeded(self, result: FittingResult):
        self.logger.info("Epoch for sample [%s] has finished, mean squared error is [%E].", result.name, result.mean_squared_error)
        self.current_fitting_result = result
        if self.auto_record_flag:
            if result.has_invalid_value:
                self.record_msg_box.setWindowTitle(self.tr("Warning"))
                self.record_msg_box.setText(self.tr("The fitting result may be invalid."))
                exec_result = self.record_msg_box.exec_()
                if exec_result == QMessageBox.Discard:
                    self.logger.info("Fitting result of sample [%s] was discarded by user.", result.name)
                    return

            self.record_current_data()
            self.sigShowFittingResult.emit(result)

    def on_multiprocessing_task_finished(self, succeeded_results: List[FittingResult]):
        for fitting_result in succeeded_results:
            if fitting_result.has_invalid_value:
                self.logger.warning("There is invalid value in the fitting result of sample [%s].", fitting_result.name)
                self.gui_logger.warning(self.tr("There is invalid value in the fitting result of sample [%s]."), fitting_result.name)
        self.records.extend(succeeded_results)
        self.sigDataRecorded.emit(succeeded_results)

    def on_settings_changed(self, kwargs: dict):
        for key, value in kwargs.items():
            if key == "layout":
                self.data_layout_setting = value
            elif key == "draw_charts":
                self.draw_charts_flag = value
            elif key == "auto_record":
                self.auto_record_flag = value
            else:
                raise NotImplementedError(key)
            self.logger.info("Setting [%s] have been changed to [%s].", key, value)

    def record_current_data(self):
        if self.current_fitting_result is None:
            self.logger.info("There is no fitting result to record, ignored.")
            self.show_warning(self.tr("There is no fitting result to record."))
            return
        self.records.append(self.current_fitting_result)
        self.sigDataRecorded.emit([self.current_fitting_result])

    def remove_data(self, uuids_and_names: Iterable[Tuple[UUID, str]]):
        for uuid, name in uuids_and_names:
            for i, data in enumerate(self.records):
                if uuid == data.uuid:
                    assert name == data.name
                    self.records.pop(i)
                    self.logger.info("Record of sample [%s] has been removed.", name)
                    break

    def save_data(self):
        # NOTE:
        # if don't assign the initial directory,
        # there is an about 5s delay while first calling `getOpenFileName` func
        # if there is a unreachable network disk
        init_path = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)[0]
        filename, type_str = self.file_dialog.getSaveFileName(
            self.host_widget, self.tr("Save Recorded Data"),
            init_path, "Excel (*.xlsx);;97-2003 Excel (*.xls);;CSV (*.csv)")
        if filename is None or filename == "":
            self.logger.info("The path is None or empty, ignored.")
            return
        if os.path.exists(filename):
            self.logger.warning("This file has existed and will be replaced. Filename: %s.", filename)
        self.logger.info("File path to save is [%s].", filename)
        if ".xlsx" in type_str:
            file_type = FileType.XLSX
        elif "97-2003" in type_str:
            file_type = FileType.XLS
        elif ".csv" in type_str:
            file_type = FileType.CSV
        else:
            raise NotImplementedError(type_str)
        self.logger.info("Selected file type is [%s].", file_type)
        self.sigSavingStarted.emit(filename, file_type, self.records, self.draw_charts_flag)

    def on_saving_succeeded(self):
            self.logger.info("File saved.")
            self.show_info(self.tr("The data has been saved to the file."))

    def on_saving_failed(self, filename: str, exception: Exception):
        try:
            raise exception
        except PermissionError:
            self.logger.exception("Can not access the selected file due to permission issue. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("Can not access the selected file due to permission issue, please check whether it's opened by another program."), self.save_data)
        except Exception:
            self.logger.exception("Unknown exception raised.", stack_info=True)
            self.ask_retry(self.tr("Unknown exception raised. See the log for more details."), self.save_data)

    def setup_all(self):
        self.loading_thread.start()
        self.saving_thread.start()

    def cleanup_all(self):
        self.loading_thread.terminate()
        self.saving_thread.terminate()

    def save_session(self):
        # NOTE:
        # if don't assign the initial directory,
        # there is an about 5s delay while first calling `getOpenFileName` func
        # if there is a unreachable network disk
        init_path = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)[0]
        filename, type_str = self.file_dialog.getSaveFileName(
            self.host_widget, self.tr("Save Session File"),
            init_path, "Session File (*.dat)")
        if filename is None or filename == "":
            self.logger.info("The path is None or empty, ignored.")
            return
        if os.path.exists(filename):
            self.logger.warning("This file has existed and will be replaced. Filename: %s.", filename)
        self.logger.info("File path to save is [%s].", filename)

        with open(filename, mode="wb") as f:
            pickle.dump(self.dataset, f)
            pickle.dump(self.records, f)
        self.logger.info("Session file has been saved.")

    def load_session(self):
        # NOTE:
        # if don't assign the initial directory,
        # there is an about 5s delay while first calling `getOpenFileName` func
        # if there is a unreachable network disk
        init_path = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)[0]
        filename, type_str = self.file_dialog.getOpenFileName(
            self.host_widget, self.tr("Select Session File"),
            init_path, "Session File (*.dat)")
        if filename is None or filename == "":
            self.logger.info("The user did not select a file, ignored.")
            return

        if os.path.exists(filename):
            with open(filename, mode="rb") as f:
                self.dataset = pickle.load(f)
                self.records = pickle.load(f)
                self.sigDataLoaded.emit(self.dataset)
                # TODO: records need to be clear
                self.sigDataRecorded.emit(self.records)
