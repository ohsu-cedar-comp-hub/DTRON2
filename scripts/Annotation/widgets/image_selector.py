from qtpy.QtWidgets import QComboBox
from functools import partial
from input_handler import InputHandler
from util.ui import prompt_save_dialog
import logging

class ImageSelector(QComboBox):
    def __init__(self, folder_path, pattern, parent=None, on_item_selected=None, include_blank_option=True, make_export_handler=None):
        super().__init__(parent)
        self.input_handler = InputHandler(folder_path, pattern)
        self.include_blank_option = include_blank_option
        if include_blank_option:
            self.addItem('')
        self.addItems(self.input_handler.get_image_file_names())
        self._persisted_index = self.currentIndex()
        self.currentIndexChanged.connect(self._handle_selection)

        # set the on_item_selected function or noop if not provided
        self.on_item_selected = on_item_selected or partial(lambda *args: None)

        self.make_export_handler = make_export_handler

    # Displays a message box if there are unsaved changes in annotations
    # Returns true if must proceed, returns false if user selected to cancel the operation.
    def _handle_unsaved_changes(self):
        # if the make_export_handler option is not set we should not prompt a message box
        if not self.make_export_handler:
            return True

        export_handler = self.make_export_handler()
        if not export_handler or not export_handler.is_updated():
            return True

        def on_save():
            export_handler.export_to_file()

        def on_cancel():
            self._rollback_to_persisted_index()

        prompt_text = 'There are unsaved changes. Do you want to save them before changing the selection?'
        return prompt_save_dialog(prompt_text, on_save, on_cancel)

    def _rollback_to_persisted_index(self):
        self.blockSignals(True)
        self.setCurrentIndex(self._persisted_index)
        self.blockSignals(False)

    def _handle_selection(self, index):
        canceled = not self._handle_unsaved_changes()
        if canceled:
            return

        file_index = index
        blank = False
        if self.include_blank_option:
            if index == 0:
                blank = True
            file_index -= 1

        if blank:
            image_file_name = ""
            image_file_path = None
            annot_file_path = None
        else:
            image_file_name = self.input_handler.get_image_file_name_at(file_index)
            image_file_path = self.input_handler.get_image_file_path_at(file_index)
            annot_file_path = self.input_handler.get_annotation_file_path_at(file_index)

        try:
            self.on_item_selected(image_file_name, image_file_path, annot_file_path)
            self._persisted_index = index
        except Exception as e:
            self._rollback_to_persisted_index()
            logging.error(e)
