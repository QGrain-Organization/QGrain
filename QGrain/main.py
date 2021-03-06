__all__ = ["QGRAIN_VERSION", "main"]

import logging
import os
import shutil
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from multiprocessing import freeze_support

from PySide2.QtCore import QSettings, QTranslator, Qt
from PySide2.QtGui import QIcon, QPixmap
from PySide2.QtWidgets import QApplication, QSplashScreen

from QGrain.ui.MainWindow import GUILogHandler, MainWindow

QGRAIN_VERSION = "0.2.8"
QGRAIN_ROOT_PATH = os.path.dirname(__file__)


def getdirsize(dir):
   size = 0
   for root, _, files in os.walk(dir):
      size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
   return size

def create_necessary_folders():
    necessary_folders = (os.path.join(QGRAIN_ROOT_PATH, "logs"),)
    for folder in necessary_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

def get_language():
    settings = QSettings(os.path.join(QGRAIN_ROOT_PATH, "settings", "QGrain.ini"), QSettings.Format.IniFormat)
    settings.beginGroup("app_settings")
    lang = settings.value("language", defaultValue="en", type=str)
    settings.endGroup()
    return lang

def get_theme():
    settings = QSettings(os.path.join(QGRAIN_ROOT_PATH, "settings", "QGrain.ini"), QSettings.Format.IniFormat)
    settings.beginGroup("app_settings")
    theme = settings.value("theme", defaultValue="MaterialDark", type=str)
    settings.endGroup()
    return theme

def setup_language(app: QApplication):
    lang = get_language()
    trans = QTranslator(app)
    trans.load(os.path.join(QGRAIN_ROOT_PATH, "i18n", lang))
    app.installTranslator(trans)

def setup_theme(app: QApplication) -> bool:
    theme = get_theme()
    with open(os.path.join(QGRAIN_ROOT_PATH, "settings", "qss", "{0}.qss".format(theme))) as template:
        template_styles = template.read()
    with open(os.path.join(QGRAIN_ROOT_PATH, "settings", "custom.qss")) as custom:
        custom_style = custom.read()
    app.setStyleSheet(template_styles+custom_style)

    if theme == "Aqua":
        return False
    elif theme == "Ubuntu":
        return False
    elif theme == "ElegantDark":
        return True
    elif theme == "MaterialDark":
        return True
    else:
        raise NotImplementedError(theme)

def setup_logging(main_window: MainWindow):
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler = TimedRotatingFileHandler(os.path.join(QGRAIN_ROOT_PATH, "logs", "qgrain.log"), when="D", backupCount=8, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(format_str))
    gui_handler = GUILogHandler(main_window)
    gui_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, format=format_str)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("GUI").addHandler(gui_handler)

def exec_qgrain():
    create_necessary_folders()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    logo = QPixmap(os.path.join(QGRAIN_ROOT_PATH, "settings", "icons", "splash_logo.png"))
    splash = QSplashScreen(logo)
    splash.show()
    app.processEvents()
    setup_language(app)
    is_dark = setup_theme(app)
    main_window = MainWindow(is_dark=is_dark)
    main_window.setWindowTitle("QGrain")
    main_window.setWindowIcon(QIcon(os.path.join(QGRAIN_ROOT_PATH, "settings", "icons", "icon.png")))
    setup_logging(main_window)
    main_window.show()
    main_window.setup_all()
    splash.finish(main_window)
    sys.exit(app.exec_())

def main():
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)
    freeze_support()
    exec_qgrain()

if __name__ == "__main__":
    main()
