from qtpy.QtWidgets import QMessageBox

# returns False if cancel option is selected returns True otherwise
def prompt_save_dialog(prompt_text, on_save, on_cancel=None):
    # if export_handler and export_handler.is_updated():
    reply = QMessageBox.question(
        None,
        'Save',
        prompt_text,
        QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
    )

    if reply == QMessageBox.Cancel:
        # Cancel the ComboBox selection change
        if on_cancel:
            on_cancel()
        # cancel option is selected return false to indicate that
        return False

    if reply == QMessageBox.Save:
        # Save changes
        on_save()

    # cancel option is not selected return true to indicate that
    return True
