from qtpy.QtWidgets import QPushButton, QFileDialog
from util.ui import prompt_save_dialog

class FolderSelectButton(QPushButton):
    def __init__(self, text='Select Folder', initial_directory=None, on_folder_selected=None, make_export_handler=None, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.current_directory = initial_directory
        self.on_folder_selected = on_folder_selected
        self.clicked.connect(self.open_folder_dialog)
        self.make_export_handler = make_export_handler

    # returns False if the user selected cancel option, returns True otherwise
    def _handle_unsaved_changes(self):
        # if the make_export_handler option is not set we should not prompt a message box
        if not self.make_export_handler:
            return True

        export_handler = self.make_export_handler()
        if not export_handler or not export_handler.is_updated():
            return True

        def on_save():
            export_handler.export_to_file()

        prompt_text = 'There are unsaved changes. Do you want to save them before changing the selection?'
        return prompt_save_dialog(prompt_text, on_save)

    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select a folder", self.current_directory)
        if folder_path:
            canceled = not self._handle_unsaved_changes()
            if canceled:
                return
            self.current_directory = folder_path
            if self.on_folder_selected:
                self.on_folder_selected(folder_path)
