from qtpy.QtCore import QObject, QEvent
from qtpy.QtWidgets import QApplication, QMessageBox

class CloseEventFilter(QObject):
    def __init__(self, viewer, make_export_handler):
        super().__init__()
        self.viewer = viewer
        self.make_export_handler = make_export_handler

    def eventFilter(self, obj, event):
        viewer = self.viewer
        if event.type() == QEvent.Close:
            export_handler = self.make_export_handler()
            if not export_handler or not export_handler.is_updated():
                viewer.close()
            else:
                reply = QMessageBox.question(None, 'Save', 'Do you want to save changes before closing?', QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
                if reply == QMessageBox.Cancel:
                    # Cancel close event
                    event.ignore()
                    return True

                if reply == QMessageBox.Save:
                    export_handler.export_to_file()

                viewer.close()

        return super().eventFilter(obj, event)
