import logging

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter
from PySide2.QtCore import Qt
from PySide2.QtGui import QFont
from PySide2.QtWidgets import QGridLayout, QWidget

from models.FittingResult import FittingResult


class LossCanvas(QWidget):
    logger = logging.getLogger("root.ui.FittingCanvas")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget(enableMenu=False)
        self.main_layout.addWidget(self.plot_widget, 0, 0)
        # add image exporters
        self.png_exporter = ImageExporter(self.plot_widget.plotItem)
        self.svg_exporter = SVGExporter(self.plot_widget.plotItem)
        # show all axis
        self.plot_widget.plotItem.showAxis("left")
        self.plot_widget.plotItem.showAxis("right")
        self.plot_widget.plotItem.showAxis("top")
        self.plot_widget.plotItem.showAxis("bottom")
        # plot data item
        self.style = dict(pen=pg.mkPen("#062170", width=3))
        self.plot_data_item = pg.PlotDataItem(name="Loss", **self.style)
        self.plot_widget.plotItem.addItem(self.plot_data_item)
        # set labels
        self.label_styles = {"font-family": "Times New Roman"}
        self.plot_widget.plotItem.setLabel("left", self.tr("Loss"), **self.label_styles)
        self.plot_widget.plotItem.setLabel("bottom", self.tr("Iteration"), **self.label_styles)
        # set title
        self.title_format = """<font face="Times New Roman">%s</font>"""
        self.plot_widget.plotItem.setTitle(self.title_format % self.tr("Loss Canvas"))
        # show grids
        self.plot_widget.plotItem.showGrid(True, True)
        # set the font of ticks
        self.tickFont = QFont("Arial")
        self.tickFont.setPointSize(8)
        self.plot_widget.plotItem.getAxis("left").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("right").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("top").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("bottom").tickFont = self.tickFont
        # set auto SI
        self.plot_widget.plotItem.getAxis("left").enableAutoSIPrefix(enable=False)
        self.plot_widget.plotItem.getAxis("right").enableAutoSIPrefix(enable=False)
        self.plot_widget.plotItem.getAxis("top").enableAutoSIPrefix(enable=False)
        self.plot_widget.plotItem.getAxis("bottom").enableAutoSIPrefix(enable=False)
        # set legend
        self.legend_format = """<font face="Times New Roman">%s</font>"""
        self.legend = pg.LegendItem(offset=(80, 50))
        self.legend.setParentItem(self.plot_widget.plotItem)
        self.legend.addItem(self.plot_data_item, self.legend_format % self.tr("Loss"))
        # set y log
        self.plot_widget.plotItem.setLogMode(y=True)

        # data
        self.x = []
        self.y = []
        self.result_info = None

    def on_fitting_started(self):
        self.x.clear()
        self.y.clear()

    def on_fitting_finished(self):
        name, distribution_type, component_number = self.result_info
        self.png_exporter.export("./temp/loss_canvas/png/{0} - {1} - {2}.png".format(
            name, distribution_type, component_number))
        self.svg_exporter.export("./temp/loss_canvas/svg/{0} - {1} - {2}.svg".format(
            name, distribution_type, component_number))

    def on_single_iteration_finished(self, current_iteration: int, result: FittingResult):
        if current_iteration == 0:
            self.result_info = (result.name, result.distribution_type, result.component_number)
            self.plot_widget.plotItem.setTitle(self.title_format % ("{0} "+self.tr("Iteration")+" ({1})").format(result.name, current_iteration))
        loss = result.mean_squared_error
        self.x.append(current_iteration)
        self.y.append(loss)
        self.plot_data_item.setData(self.x, self.y)
