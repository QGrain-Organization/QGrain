__all__ = ["GUILogHandler", "MainWindow"]

import logging
import os
import time
from typing import List

from PySide2.QtCore import (QCoreApplication, QEventLoop, QMutex, QSettings,
                            Qt, QThread, Signal)
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import (QAbstractItemView, QAction, QDockWidget,
                               QMainWindow, QMessageBox, QTableWidget,
                               QTableWidgetItem)

from QGrain.models.SampleDataset import SampleDataset
from QGrain.resolvers.GUIResolver import GUIResolver
from QGrain.resolvers.MultiprocessingResolver import MultiProcessingResolver
from QGrain.ui.AboutWindow import AboutWindow
from QGrain.ui.AllDistributionCanvas import AllDistributionCanvas
from QGrain.ui.ControlPanel import ControlPanel
from QGrain.ui.DataManager import DataManager
from QGrain.ui.DistributionCanvas import DistributionCanvas
from QGrain.ui.LossCanvas import LossCanvas
from QGrain.ui.PCAPanel import PCAPanel
from QGrain.ui.RawDataTable import RawDataTable
from QGrain.ui.RecordedDataTable import RecordedDataTable
from QGrain.ui.SettingWindow import SettingWindow
from QGrain.ui.TaskWindow import TaskWindow

QGRAIN_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

class GUILogHandler(logging.Handler):
    def __init__(self, target_widget):
        logging.Handler.__init__(self)
        self.target = target_widget
        self.__mutex = QMutex()

    def emit(self, record):
        self.__mutex.lock()
        if record.levelno < self.level:
            return
        self.target.show_message(record.msg % record.args)
        self.__mutex.unlock()


class MainWindow(QMainWindow):
    sigDataSelected = Signal(int)
    sigSetup = Signal()
    sigCleanup = Signal()
    logger = logging.getLogger("root.MainWindow")
    gui_logger = logging.getLogger("GUI")
    def __init__(self, is_dark=True):
        super().__init__()
        self.gui_fitting_thread = QThread()
        self.gui_resolver = GUIResolver()
        # disable multi-thread for debug
        self.gui_resolver.moveToThread(self.gui_fitting_thread)
        self.multiprocessing_fitting_thread = QThread()
        self.multiprocessing_resolver = MultiProcessingResolver()
        self.multiprocessing_resolver.moveToThread(self.multiprocessing_fitting_thread)
        self.data_manager = DataManager(self)
        if is_dark:
            self.icon_folder = os.path.join(QGRAIN_ROOT_PATH, "settings", "icons", "dark")
        else:
            self.icon_folder = os.path.join(QGRAIN_ROOT_PATH, "settings", "icons", "light")
        self.init_ui(is_dark)
        self.status_bar = self.statusBar()

        self.connect_all()
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.reset_dock_layout()

    def show_message(self, message):
        self.status_bar.showMessage(message)

    def init_ui(self, is_dark: bool):
        # Menu
        self.file_menu = self.menuBar().addMenu(self.tr("File"))
        self.load_data_action = QAction(QIcon(os.path.join(self.icon_folder, "load.png")), self.tr("Load Data"), self)
        self.file_menu.addAction(self.load_data_action)
        self.save_results_action = QAction(QIcon(os.path.join(self.icon_folder, "save.png")), self.tr("Save Results"), self)
        self.file_menu.addAction(self.save_results_action)
        self.load_session_action = QAction(QIcon(os.path.join(self.icon_folder, "load.png")), self.tr("Load Session"), self)
        self.file_menu.addAction(self.load_session_action)
        self.save_session_action = QAction(QIcon(os.path.join(self.icon_folder, "save.png")), self.tr("Save Session"), self)
        self.file_menu.addAction(self.save_session_action)
        self.docks_menu = self.menuBar().addMenu(self.tr("Docks"))
        self.pca_panel_action = QAction(QIcon(os.path.join(self.icon_folder, "pca.png")), self.tr("PCA Panel"), self)
        self.docks_menu.addAction(self.pca_panel_action)
        self.loss_canvas_action = QAction(QIcon(os.path.join(self.icon_folder, "canvas.png")), self.tr("Loss Canvas"), self)
        self.docks_menu.addAction(self.loss_canvas_action)
        self.all_distribution_canvas_action = QAction(QIcon(os.path.join(self.icon_folder, "canvas.png")), self.tr("All Distribution Canvas"), self)
        self.docks_menu.addAction(self.all_distribution_canvas_action)
        self.distribution_canvas_action = QAction(QIcon(os.path.join(self.icon_folder, "canvas.png")), self.tr("Distribution Canvas"), self)
        self.docks_menu.addAction(self.distribution_canvas_action)
        self.control_panel_action = QAction(QIcon(os.path.join(self.icon_folder, "control.png")), self.tr("Control Panel"), self)
        self.docks_menu.addAction(self.control_panel_action)
        self.raw_data_table_action = QAction(QIcon(os.path.join(self.icon_folder, "raw_data_table.png")), self.tr("Raw Data Table"), self)
        self.docks_menu.addAction(self.raw_data_table_action)
        self.recorded_data_table_action = QAction(QIcon(os.path.join(self.icon_folder, "recorded_data_table.png")), self.tr("Recorded Data Table"), self)
        self.docks_menu.addAction(self.recorded_data_table_action)
        self.reset_docks_actions = QAction(QIcon(os.path.join(self.icon_folder, "reset.png")), self.tr("Reset"), self)
        self.docks_menu.addAction(self.reset_docks_actions)
        self.settings_action = self.menuBar().addAction(self.tr("Settings"))
        self.about_action = self.menuBar().addAction(self.tr("About"))

        self.setDockNestingEnabled(True)
        # PCA Panel
        self.pca_panel_dock = QDockWidget(self.tr("PCA Panel"))
        self.pca_panel = PCAPanel(is_dark=is_dark)
        self.pca_panel_dock.setWidget(self.pca_panel)
        self.pca_panel_dock.setObjectName("PCAPanelDock")
        # Loss Canvas
        self.loss_canvas_dock = QDockWidget(self.tr("Loss Canvas"))
        self.loss_canvas = LossCanvas(is_dark=is_dark)
        self.loss_canvas_dock.setWidget(self.loss_canvas)
        self.loss_canvas_dock.setObjectName("LossCanvasDock")
        # All Distribution Canvas
        self.all_distribution_canvas_dock = QDockWidget(self.tr("All Distribution Canvas"))
        self.all_distribution_canvas = AllDistributionCanvas(is_dark=is_dark)
        self.all_distribution_canvas_dock.setWidget(self.all_distribution_canvas)
        self.all_distribution_canvas_dock.setObjectName("AllDistributionCanvasDock")
        # Distribution Canvas
        self.distribution_canvas_dock = QDockWidget(self.tr("Distribution Canvas"))
        self.distribution_canvas = DistributionCanvas(is_dark=is_dark)
        self.distribution_canvas_dock.setWidget(self.distribution_canvas)
        self.distribution_canvas_dock.setObjectName("DistributionCanvasDock")

        # Control Panel
        self.control_panel_dock = QDockWidget(self.tr("Control Panel"))
        self.control_panel = ControlPanel()
        self.control_panel_dock.setWidget(self.control_panel)
        self.control_panel_dock.setObjectName("ControlPanelDock")

        # Raw Data Table
        self.raw_data_dock = QDockWidget(self.tr("Raw Data Table"))
        self.raw_data_table = RawDataTable()
        self.raw_data_dock.setWidget(self.raw_data_table)
        self.raw_data_dock.setObjectName("RawDataTableDock")

        # Recorded Data Table
        self.recorded_data_dock = QDockWidget(self.tr("Recorded Data Table"))
        self.recorded_data_table = RecordedDataTable()
        self.recorded_data_dock.setWidget(self.recorded_data_table)
        self.recorded_data_dock.setObjectName("RecordedDataTableDock")

        self.settings_window = SettingWindow(self)
        self.about_window = AboutWindow(self)
        self.task_window = TaskWindow(self, self.multiprocessing_resolver)

    def connect_all(self):
        self.control_panel.sigDistributionTypeChanged.connect(self.gui_resolver.on_distribution_type_changed)
        self.control_panel.sigComponentNumberChanged.connect(self.gui_resolver.on_component_number_changed)
        self.control_panel.sigComponentNumberChanged.connect(self.distribution_canvas.on_component_number_changed)
        self.control_panel.sigFocusSampleChanged.connect(self.data_manager.on_focus_sample_changed)
        self.control_panel.sigObserveIterationChanged.connect(self.distribution_canvas.on_observe_iteration_changed)
        self.control_panel.sigObserveIterationChanged.connect(self.loss_canvas.on_observe_iteration_changed)
        self.control_panel.sigInheritParamsChanged.connect(self.gui_resolver.on_inherit_params_changed)
        self.control_panel.sigGUIResolverFittingStarted.connect(self.gui_resolver.try_fit)
        self.control_panel.sigMultiProcessingFittingButtonClicked.connect(self.on_multiprocessing_fitting_button_clicked)

        self.control_panel.sigDataSettingsChanged.connect(self.data_manager.on_settings_changed)
        self.control_panel.sigGUIResolverFittingCanceled.connect(self.on_gui_fitting_task_canceled)
        # Connect directly
        self.control_panel.record_button.clicked.connect(self.data_manager.record_current_data)

        self.distribution_canvas.sigExpectedMeanValueChanged.connect(self.gui_resolver.on_excepted_mean_value_changed)

        self.data_manager.sigDataLoaded.connect(self.raw_data_table.on_data_loaded)
        self.data_manager.sigDataLoaded.connect(self.control_panel.on_data_loaded)
        self.data_manager.sigDataLoaded.connect(self.task_window.on_data_loaded)
        self.data_manager.sigDataLoaded.connect(self.pca_panel.on_data_loaded)
        self.data_manager.sigDataLoaded.connect(self.all_distribution_canvas.on_data_loaded)
        self.data_manager.sigTargetDataChanged.connect(self.gui_resolver.on_target_data_changed)
        self.data_manager.sigTargetDataChanged.connect(self.distribution_canvas.show_target_distribution)
        self.data_manager.sigDataRecorded.connect(self.recorded_data_table.on_data_recorded)
        self.data_manager.sigShowFittingResult.connect(self.distribution_canvas.show_fitting_result)
        self.data_manager.sigShowFittingResult.connect(self.loss_canvas.show_fitting_result)

        self.gui_resolver.sigFittingStarted.connect(self.control_panel.on_fitting_started)
        self.gui_resolver.sigFittingFinished.connect(self.control_panel.on_fitting_finished)
        self.gui_resolver.sigFittingSucceeded.connect(self.control_panel.on_fitting_suceeded)
        self.gui_resolver.sigFittingSucceeded.connect(self.data_manager.on_fitting_suceeded)
        self.gui_resolver.sigFittingFailed.connect(self.control_panel.on_fitting_failed)


        self.multiprocessing_resolver.task_state_updated.connect(self.task_window.on_task_state_updated)

        self.task_window.task_generated_signal.connect(self.multiprocessing_resolver.on_task_generated)
        self.task_window.fitting_started_signal.connect(self.multiprocessing_resolver.execute_tasks)
        self.task_window.fitting_finished_signal.connect(self.data_manager.on_multiprocessing_task_finished)

        self.recorded_data_table.sigRemoveRecords.connect(self.data_manager.remove_data)
        self.recorded_data_table.sigShowDistribution.connect(self.distribution_canvas.show_fitting_result)
        self.recorded_data_table.sigShowLoss.connect(self.loss_canvas.show_fitting_result)
        self.recorded_data_table.sigGenerateDistributionVideo.connect(self.distribution_canvas.generate_video)
        # Dock menu actions
        self.pca_panel_action.triggered.connect(self.show_pca_panel_dock)
        self.loss_canvas_action.triggered.connect(self.show_loss_canvas_dock)
        self.all_distribution_canvas_action.triggered.connect(self.show_all_distribution_canvas_dock)
        self.distribution_canvas_action.triggered.connect(self.show_distribution_canvas_dock)
        self.control_panel_action.triggered.connect(self.show_control_panel_dock)
        self.raw_data_table_action.triggered.connect(self.show_raw_data_dock)
        self.recorded_data_table_action.triggered.connect(self.show_recorded_data_dock)
        self.reset_docks_actions.triggered.connect(self.reset_dock_layout)
        self.settings_action.triggered.connect(self.settings_window.show)
        self.about_action.triggered.connect(self.about_window.show)

        self.load_data_action.triggered.connect(self.data_manager.load_data)
        self.save_results_action.triggered.connect(self.data_manager.save_data)
        self.load_session_action.triggered.connect(self.data_manager.load_session)
        self.save_session_action.triggered.connect(self.data_manager.save_session)
        self.sigDataSelected.connect(self.control_panel.on_data_selected)

        self.sigSetup.connect(self.control_panel.setup_all)
        self.sigSetup.connect(self.data_manager.setup_all)
        self.sigSetup.connect(self.settings_window.setup_all)
        self.sigSetup.connect(self.multiprocessing_resolver.setup_all)
        self.sigCleanup.connect(self.data_manager.cleanup_all)
        self.sigCleanup.connect(self.multiprocessing_resolver.cleanup_all)

        self.settings_window.data_settings.data_settings_changed_signal.connect(self.data_manager.on_settings_changed)
        self.settings_window.algorithm_settings.setting_changed_signal.connect(self.gui_resolver.on_algorithm_settings_changed)

    def reset_dock_layout(self):
        self.pca_panel_dock.show()
        self.loss_canvas_dock.show()
        self.all_distribution_canvas_dock.show()
        self.distribution_canvas_dock.show()
        self.control_panel_dock.show()
        self.raw_data_dock.show()
        self.recorded_data_dock.show()

        self.pca_panel_dock.setFloating(False)
        self.loss_canvas_dock.setFloating(False)
        self.all_distribution_canvas_dock.setFloating(False)
        self.distribution_canvas_dock.setFloating(False)
        self.control_panel_dock.setFloating(False)
        self.raw_data_dock.setFloating(False)
        self.recorded_data_dock.setFloating(False)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.pca_panel_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.raw_data_dock)
        self.splitDockWidget(self.raw_data_dock, self.recorded_data_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self.pca_panel_dock, self.control_panel_dock, Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.pca_panel_dock, self.loss_canvas_dock)
        self.tabifyDockWidget(self.pca_panel_dock, self.all_distribution_canvas_dock)
        self.tabifyDockWidget(self.pca_panel_dock, self.distribution_canvas_dock)

        self.resizeDocks((self.pca_panel_dock, self.control_panel_dock), (self.height()*0.65, self.height()*0.35), Qt.Orientation.Vertical)
        self.resizeDocks((self.pca_panel_dock, self.control_panel_dock, self.raw_data_dock, self.recorded_data_dock), (self.width()*0.5, self.width()*0.5, self.width()*0.25, self.width()*0.25), Qt.Orientation.Horizontal)

    def show_pca_panel_dock(self):
        self.pca_panel_dock.show()

    def show_loss_canvas_dock(self):
        self.loss_canvas_dock.show()

    def show_all_distribution_canvas_dock(self):
        self.all_distribution_canvas_dock.show()

    def show_distribution_canvas_dock(self):
        self.distribution_canvas_dock.show()

    def show_control_panel_dock(self):
        self.control_panel_dock.show()

    def show_raw_data_dock(self):
        self.raw_data_dock.show()

    def show_recorded_data_dock(self):
        self.recorded_data_dock.show()

    def on_gui_fitting_task_canceled(self):
        self.gui_resolver.cancel()

    def on_multiprocessing_fitting_button_clicked(self):
        self.task_window.exec_()

    def setup_all(self):
        # must call this again to make the layout as expected
        self.reset_dock_layout()
        # emit signal to let other object to setup
        self.gui_fitting_thread.start()
        self.multiprocessing_fitting_thread.start()
        # emit after the threads are started
        self.sigSetup.emit()

        settings = QSettings(os.path.join(QGRAIN_ROOT_PATH, "settings", "ui.ini"), QSettings.IniFormat)
        settings.beginGroup("MainWindow")
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        windowState = settings.value("windowState")
        if windowState is not None:
            self.restoreState(windowState)
        settings.endGroup()

    def closeEvent(self, e):
        # TODO: add task running check
        self.sigCleanup.emit()
        self.hide()

        settings = QSettings(os.path.join(QGRAIN_ROOT_PATH, "settings", "ui.ini"), QSettings.IniFormat)
        settings.beginGroup("MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.endGroup()

        time.sleep(2)
        self.gui_fitting_thread.terminate()
        self.multiprocessing_fitting_thread.terminate()
        e.accept()
