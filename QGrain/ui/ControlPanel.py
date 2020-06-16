__all__ = ["ControlPanel"]

import logging
import os
from uuid import UUID

import numpy as np
from PySide2.QtCore import Qt, QTimer, Signal
from PySide2.QtGui import QFont
from PySide2.QtWidgets import (QCheckBox, QGridLayout, QLabel, QMessageBox,
                               QPushButton, QRadioButton, QWidget)

from QGrain.algorithms import DistributionType
from QGrain.models.FittingResult import FittingResult
from QGrain.models.SampleDataset import SampleDataset


class ControlPanel(QWidget):
    sigDistributionTypeChanged = Signal(int)
    sigComponentNumberChanged = Signal(int)
    sigFocusSampleChanged = Signal(UUID) # uuid of the sample
    sigDataSettingsChanged = Signal(dict)
    sigObserveIterationChanged = Signal(bool)
    sigInheritParamsChanged = Signal(bool)
    sigGUIResolverFittingStarted = Signal()
    sigGUIResolverFittingCanceled = Signal()
    sigMultiProcessingFittingButtonClicked = Signal()
    logger = logging.getLogger("root.ui.ControlPanel")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.__component_number = 2
        self.__data_index = 0
        self.sample_names = None
        self.sample_uuids = None
        self.auto_fit_flag = True
        self.auto_run_timer = QTimer()
        self.auto_run_timer.setSingleShot(True)
        self.auto_run_timer.timeout.connect(self.on_auto_run_timer_timeout)
        self.auto_run_flag = False

        self.init_ui()
        self.connect_all()
        self.change_enable_states(True)

        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        # Distribution type
        self.distribution_type_label = QLabel(self.tr("Distribution Type:"), self)
        self.distribution_type_label.setToolTip(self.tr("Select the base distribution function of each component."))
        self.distribution_normal_radio_button = QRadioButton(self.tr("Normal"), self)
        self.distribution_normal_radio_button.setToolTip(self.tr("See [%s] for more details.") % "https://en.wikipedia.org/wiki/Normal_distribution")
        self.distribution_weibull_radio_button = QRadioButton(self.tr("Weibull"), self)
        self.distribution_weibull_radio_button.setToolTip(self.tr("See [%s] for more details.") % "https://en.wikipedia.org/wiki/Weibull_distribution")
        self.distribution_gen_weibull_radio_button = QRadioButton(self.tr("Gen. Weibull"), self)
        self.distribution_gen_weibull_radio_button.setToolTip(self.tr("General Weibull distribution which has an additional location parameter."))
        self.main_layout.addWidget(self.distribution_type_label, 0, 0, 1, 2)
        self.main_layout.addWidget(self.distribution_normal_radio_button, 0, 1)
        self.main_layout.addWidget(self.distribution_weibull_radio_button, 0, 2)
        self.main_layout.addWidget(self.distribution_gen_weibull_radio_button, 0, 3)
        # Component number
        self.component_number_label = QLabel(self.tr("Component Number:"))
        self.component_number_label.setToolTip(self.tr("Select the component number of the mixed distribution."))
        self.component_number_display = QLabel(self.tr("Unknown"))
        self.component_number_add_button = QPushButton(self.tr("+"))
        self.component_number_add_button.setToolTip(self.tr("Click to add one component.\nIt should be less than 11."))
        self.component_reduce_button = QPushButton(self.tr("-"))
        self.component_reduce_button.setToolTip(self.tr("Click to reduce one component.\nIt should be positive."))
        self.main_layout.addWidget(self.component_number_label, 1, 0)
        self.main_layout.addWidget(self.component_number_display, 1, 1)
        self.main_layout.addWidget(self.component_number_add_button, 1, 2)
        self.main_layout.addWidget(self.component_reduce_button, 1, 3)
        # Some usual settings
        self.observe_iteration_checkbox = QCheckBox(self.tr("Observe Iteration"))
        self.observe_iteration_checkbox.setToolTip(self.tr("Whether to display the iteration procedure."))
        self.inherit_params_checkbox = QCheckBox(self.tr("Inherit Parameters"))
        self.inherit_params_checkbox.setToolTip(self.tr("Whether to inherit the parameters of last fitting.\nIt will improve the accuracy and efficiency when the samples are continuous."))
        self.main_layout.addWidget(self.observe_iteration_checkbox, 2, 0, 1, 2)
        self.main_layout.addWidget(self.inherit_params_checkbox, 2, 2, 1, 2)
        self.auto_fit_checkbox = QCheckBox(self.tr("Auto Fit"))
        self.auto_fit_checkbox.setToolTip(self.tr("Whether to automaticlly fit after the sample data changed."))
        self.auto_record_checkbox = QCheckBox(self.tr("Auto Record"))
        self.auto_record_checkbox.setToolTip(self.tr("Whether to automaticlly record the fitting result after fitting finished."))
        self.main_layout.addWidget(self.auto_fit_checkbox, 3, 0, 1, 2)
        self.main_layout.addWidget(self.auto_record_checkbox, 3, 2, 1, 2)
        # Target data to fit
        self.data_index_label = QLabel(self.tr("Current Sample:"))
        self.data_index_label.setToolTip(self.tr("Current sample to fit."))
        self.data_index_display = QLabel(self.tr("Unknown"))
        self.data_index_previous_button = QPushButton(self.tr("Previous"))
        self.data_index_previous_button.setToolTip(self.tr("Click to back to the previous sample."))
        self.data_index_next_button = QPushButton(self.tr("Next"))
        self.data_index_next_button.setToolTip(self.tr("Click to jump to the next sample."))
        self.main_layout.addWidget(self.data_index_label, 4, 0)
        self.main_layout.addWidget(self.data_index_display, 4, 1)
        self.main_layout.addWidget(self.data_index_previous_button, 4, 2)
        self.main_layout.addWidget(self.data_index_next_button, 4, 3)
        # Control bottons
        self.auto_run = QPushButton(self.tr("Auto Run Orderly"))
        self.auto_run.setToolTip(self.tr("Click to run the program automatically.\nThe samples from current to the end will be processed one by one."))
        self.cancel_run = QPushButton(self.tr("Cancel"))
        self.cancel_run.setToolTip(self.tr("Click to cancel the fitting progress."))
        self.try_fit_button = QPushButton(self.tr("Try Fit"))
        self.try_fit_button.setToolTip(self.tr("Click to fit the current sample."))
        self.record_button = QPushButton(self.tr("Record"))
        self.record_button.setToolTip(self.tr("Click to record the current fitting result.\nNote: It will record the LAST SUCCESS fitting result, NOT CURRENT SAMPLE."))
        self.main_layout.addWidget(self.auto_run, 5, 0)
        self.main_layout.addWidget(self.cancel_run, 5, 1)
        self.main_layout.addWidget(self.try_fit_button, 5, 2)
        self.main_layout.addWidget(self.record_button, 5, 3)
        # Multi cores
        self.multiprocessing_button = QPushButton(self.tr("Multi Cores Fitting"))
        self.multiprocessing_button.setToolTip(self.tr("Click to fit all samples. It will utilize all cores of cpu to accelerate calculation."))
        self.main_layout.addWidget(self.multiprocessing_button, 6, 0, 1, 4)

    def connect_all(self):
        self.distribution_normal_radio_button.clicked.connect(self.on_distribution_type_changed)
        self.distribution_weibull_radio_button.clicked.connect(self.on_distribution_type_changed)
        self.distribution_gen_weibull_radio_button.clicked.connect(self.on_distribution_type_changed)
        self.component_number_add_button.clicked.connect(self.on_component_number_add_clicked)
        self.component_reduce_button.clicked.connect(self.on_component_number_reduce_clicked)

        self.data_index_previous_button.clicked.connect(self.on_data_index_previous_clicked)
        self.data_index_next_button.clicked.connect(self.on_data_index_next_clicked)

        self.observe_iteration_checkbox.stateChanged.connect(self.on_show_iteration_changed)
        self.inherit_params_checkbox.stateChanged.connect(self.on_inherit_params_changed)
        self.auto_fit_checkbox.stateChanged.connect(self.on_auto_fit_changed)
        self.auto_record_checkbox.stateChanged.connect(self.on_auto_record_changed)

        self.auto_run.clicked.connect(self.on_auto_run_clicked)
        self.cancel_run.clicked.connect(self.on_cancel_run_clicked)
        self.try_fit_button.clicked.connect(self.on_try_fit_clicked)
        self.multiprocessing_button.clicked.connect(self.on_multiprocessing_clicked)

    @property
    def component_number(self):
        return self.__component_number

    @component_number.setter
    def component_number(self, value: int):
        # check the validity
        # component number should be non-negative
        if value < 1 or value > 10:
            self.gui_logger.info(self.tr("The component number should be >= 1 and <= 10."))
            return
        # update the label to display the value
        self.component_number_display.setText(str(value))
        self.__component_number = value
        self.sigComponentNumberChanged.emit(value)
        # auto emit target data change signal again when component number changed
        if self.sample_names is not None:
            self.data_index = self.data_index

    @property
    def data_index(self) -> int:
        return self.__data_index

    @data_index.setter
    def data_index(self, value: int):
        if not self.has_data:
            return
        if value < 0 or value >= self.data_length:
            self.gui_logger.info(self.tr("It has reached the first/last sample."))
            return
        # update the label to display the name of this sample
        self.data_index_display.setText(self.sample_names[value])
        self.__data_index = value
        self.logger.debug("Data index has been set to [%d].", value)
        self.sigFocusSampleChanged.emit(self.sample_uuids[value])
        if self.auto_fit_flag:
            self.sigGUIResolverFittingStarted.emit()

    @property
    def current_name(self) -> str:
        return self.sample_names[self.data_index]

    @property
    def has_data(self) -> bool:
        if self.data_length <= 0:
            return False
        else:
            return True

    @property
    def data_length(self) -> int:
        if self.sample_names is None:
            return 0
        else:
            return len(self.sample_names)

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

    def check_data_loaded(self, show_msg=True):
        if not self.has_data:
            if show_msg:
                self.show_warning(self.tr("The grain size data has not been loaded, the operation is invalid."))
            return False
        else:
            return True

    def on_component_number_add_clicked(self):
        self.component_number += 1

    def on_component_number_reduce_clicked(self):
        self.component_number -= 1

    def on_distribution_type_changed(self):
        if self.distribution_normal_radio_button.isChecked():
            distribution_type = DistributionType.Normal
        elif self.distribution_weibull_radio_button.isChecked():
            distribution_type = DistributionType.Weibull
        else:
            assert self.distribution_gen_weibull_radio_button.isChecked()
            distribution_type = DistributionType.GeneralWeibull
        self.logger.info("Distribution type has been changed to [%s].", distribution_type)
        self.sigDistributionTypeChanged.emit(distribution_type.value)

        if self.sample_names is not None:
            self.data_index = self.data_index

    def on_data_index_previous_clicked(self):
        if not self.check_data_loaded():
            return
        self.data_index -= 1

    def on_data_index_next_clicked(self):
        if not self.check_data_loaded():
            return
        self.data_index += 1

    def on_show_iteration_changed(self, state: Qt.CheckState):
        if state == Qt.Checked:
            self.sigObserveIterationChanged.emit(True)
        else:
            self.sigObserveIterationChanged.emit(False)

    def on_inherit_params_changed(self, state: Qt.CheckState):
        if state == Qt.Checked:
            self.sigInheritParamsChanged.emit(True)
        else:
            self.sigInheritParamsChanged.emit(False)

    def on_auto_fit_changed(self, state: Qt.CheckState):
        if state == Qt.Checked:
            self.auto_fit_flag = True
        else:
            self.auto_fit_flag = False

    def on_auto_record_changed(self, state: Qt.CheckState):
        if state == Qt.Checked:
            self.sigDataSettingsChanged.emit({"auto_record": True})
        else:
            self.sigDataSettingsChanged.emit({"auto_record": False})

    def on_data_loaded(self, dataset: SampleDataset):
        sample_names = []
        sample_uuids = []
        for i, sample in enumerate(dataset.samples):
            sample_names.append(str(sample.name))
            sample_uuids.append(sample.uuid)
        self.sample_names = sample_names
        self.sample_uuids = sample_uuids
        self.logger.info("Data was loaded.")
        self.data_index = 0
        self.logger.info("Data index has been set to 0.")

    def on_data_selected(self, index: int):
        self.data_index = index
        self.logger.debug("Sample data at [%d] is selected.", index)

    def change_enable_states(self, enable: bool):
        if self.auto_run_flag and enable:
            return
        self.distribution_weibull_radio_button.setEnabled(enable)
        self.distribution_normal_radio_button.setEnabled(enable)
        self.component_number_add_button.setEnabled(enable)
        self.component_reduce_button.setEnabled(enable)
        self.data_index_previous_button.setEnabled(enable)
        self.data_index_next_button.setEnabled(enable)
        self.observe_iteration_checkbox.setEnabled(enable)
        self.inherit_params_checkbox.setEnabled(enable)
        self.auto_fit_checkbox.setEnabled(enable)
        self.auto_record_checkbox.setEnabled(enable)
        self.auto_run.setEnabled(enable)
        self.try_fit_button.setEnabled(enable)
        self.record_button.setEnabled(enable)
        self.multiprocessing_button.setEnabled(enable)
        self.cancel_run.setEnabled(not enable)

    def on_fitting_suceeded(self, result: FittingResult):
        if self.auto_run_flag and result.has_invalid_value:
            self.logger.warning("The fitting result may be not valid, auto running stoped.")
            self.gui_logger.warning(self.tr("The fitting result may be not valid, auto running stoped."))
            self.auto_run_flag = False
            self.change_enable_states(True)

        if self.data_index == self.data_length-1:
            self.logger.info("The auto running has reached the last sample and stoped.")
            self.gui_logger.info(self.tr("The auto running has reached the last sample and stoped."))
            self.auto_run_flag = False
            self.change_enable_states(True)

        if self.auto_run_flag:
            self.auto_run_timer.start(100)

    def on_auto_run_timer_timeout(self):
        self.data_index += 1

    def on_auto_run_clicked(self):
        if not self.check_data_loaded():
            return
        self.auto_fit_checkbox.setCheckState(Qt.Checked)
        self.auto_record_checkbox.setCheckState(Qt.Checked)
        # from current sample to fit, to avoid that it need to resart from the first sample every time
        self.auto_run_flag = True
        self.data_index = self.data_index
        self.logger.info("Auto running started from sample [%s].", self.current_name)

    def on_cancel_run_clicked(self):
        if self.auto_run_flag:
            self.auto_run_flag = False
            self.change_enable_states(True)
            self.logger.info("Auto run flag has been changed to False.")

        self.sigGUIResolverFittingCanceled.emit()

    def on_try_fit_clicked(self):
        self.sigGUIResolverFittingStarted.emit()

    def on_multiprocessing_clicked(self):
        if not self.check_data_loaded():
            return
        self.sigMultiProcessingFittingButtonClicked.emit()

    def on_fitting_started(self):
        self.change_enable_states(False)

    def on_fitting_finished(self):
        self.change_enable_states(True)

    def on_fitting_failed(self, message: str):
        if self.auto_run_flag:
            self.auto_run_flag = False
            self.change_enable_states(True)
            self.logger.info("Auto running was canceled due to the failure of fitting.")
        self.show_error(self.tr("Fitting failed. {0}").format(message))

    def setup_all(self):
        self.component_number = 3
        self.distribution_gen_weibull_radio_button.setChecked(True)
        self.observe_iteration_checkbox.setCheckState(Qt.Unchecked)
        self.inherit_params_checkbox.setCheckState(Qt.Checked)
        self.auto_fit_checkbox.setCheckState(Qt.Checked)
        self.auto_record_checkbox.setCheckState(Qt.Checked)
