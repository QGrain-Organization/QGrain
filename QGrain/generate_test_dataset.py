import os
import random

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm


def get_scale_ratio(max_ratio):
    assert max_ratio > 0.0
    return (random.random() - 0.5) / 0.5 * max_ratio  + 1.0

def get_random_sample(classes, params, max_scale_ratios, noise=0.005):
    assert len(params) == len(max_scale_ratios)
    bin_numbers = np.array(range(len(classes))) + 1
    converter = interp1d(classes, bin_numbers)
    components = []
    # get random params
    actual_params = []
    for i in range(len(params)):
        assert len(params[i]) == 3
        assert len(max_scale_ratios[i]) == 3
        component_prams = []
        for j in range(3):
            param = params[i][j] * get_scale_ratio(max_scale_ratios[i][j])
            component_prams.append(param)
        actual_params.append(component_prams)
    # standardize the fractions to make its sum equals 1
    sum_of_fractions = sum(component_prams[2] for component_prams in actual_params)
    for component_prams in actual_params:
        component_prams[2] /= sum_of_fractions
    # generate distributions
    distribution = np.zeros_like(classes)
    for mean, std, fraction in actual_params:
        component_distribution = norm.pdf(bin_numbers, converter(mean), std) * fraction
        components.append(component_distribution)
        np.add(distribution, component_distribution, out=distribution)
    # add noise
    noise_array = np.random.normal(0.0, noise, len(classes)) + 1.0
    np.multiply(distribution, noise_array, out=distribution)
    # standardize
    np.divide(distribution, np.sum(distribution), out=distribution)
    return distribution, components, actual_params


if __name__ == "__main__":
    from PySide2.QtCore import Qt, QPointF, QTimer
    from PySide2.QtCharts import QtCharts
    from PySide2.QtWidgets import QApplication
    from PySide2.QtGui import QPainter, QFont, QIcon
    import sys

    sample_classes = np.logspace(0, 5, 101) * 0.02
    # sample_params = ((1.1, 6, 0.1), (8.0, 9.0, 0.5), (30.0, 4.5, 0.4))
    # sample_max_scale_ratios = ((0.2, 0.2, 0.4), (0.4, 0.4, 0.6), (0.4, 0.4, 0.6))
    sample_params = ((1.1, 6, 0.1), (8.0, 9.0, 0.2), (30.0, 4.5, 0.4), (300, 6.5, 0.3))
    sample_max_scale_ratios = ((0.2, 0.2, 0.4), (0.2, 0.2, 0.4), (0.4, 0.4, 0.8), (0.3, 0.3, 0.6))

    app = QApplication(sys.argv)
    chart = QtCharts.QChart()
    chart.setBackgroundVisible(False)
    # chart.setAnimationOptions(QtCharts.QChart.SeriesAnimations)
    chart.setTitle("Artificial samples")
    chart.setTitleFont(QFont("Times New Roman", 14))
    axis_x = QtCharts.QLogValueAxis()
    axis_x.setTitleText("Grain size (μm)")
    axis_x.setTitleFont(QFont("Times New Roman", 12))
    axis_y = QtCharts.QValueAxis()
    axis_y.setLabelFormat("%0.3f")
    axis_y.setTitleText("Probability density")
    axis_y.setTitleFont(QFont("Times New Roman", 12))
    chart.addAxis(axis_x, Qt.AlignBottom)
    chart.addAxis(axis_y, Qt.AlignLeft)

    sum_series = QtCharts.QLineSeries()
    sum_series.setName("Sum")
    chart.addSeries(sum_series)
    sum_series.attachAxis(axis_x)
    sum_series.attachAxis(axis_y)
    components_series = []
    for i in range(len(sample_params)):
        series = QtCharts.QLineSeries()
        series.setName("C" + str(i+1))
        chart.addSeries(series)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        components_series.append(series)

    def to_points(x, y):
        return [QPointF(x_value, y_value) for x_value, y_value in zip(x, y)]

    def update():
        # prepare data
        distribution, components, _ = get_random_sample(sample_classes, sample_params, sample_max_scale_ratios)

        sum_series.replace(to_points(sample_classes, distribution))
        for series, component in zip(components_series, components):
            series.replace(to_points(sample_classes, component))
        axis_y.setRange(0.0, round(np.max(distribution)*1.2, 2))

    axis_x.setRange(sample_classes[0], sample_classes[-1])
    update()
    view = QtCharts.QChartView()
    view.setChart(chart)
    view.setRenderHint(QPainter.Antialiasing)
    view.setWindowTitle("Artificial Sample Generator")
    view.setGeometry(200, 200, 600, 500)
    view.show()

    timer = QTimer()
    timer.timeout.connect(update)
    # timer.start(2000)
    timer.start(100)

    sys.exit(app.exec_())
