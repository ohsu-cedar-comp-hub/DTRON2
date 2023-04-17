from qtpy.QtCore import QObject, QEvent
from util.ui import prompt_save_dialog

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
                prompt_text = 'Do you want to save changes before closing?'

                def on_save():
                    export_handler.export_to_file()

                def on_cancel():
                    event.ignore()

                canceled = not prompt_save_dialog(prompt_text, on_save, on_cancel)

                if canceled:
                    return True

                viewer.close()

        return super().eventFilter(obj, event)
