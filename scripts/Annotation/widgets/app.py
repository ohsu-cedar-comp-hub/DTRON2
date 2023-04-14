import qtpy.QtWidgets as widgets
from qtpy.QtCore import Qt
import numpy as np
from napari import Viewer, gui_qt
from .image_selector import ImageSelector
from .annotation_table import AnnotationTable
import os
"""
Some thoughts:
To make it more obvious which cell is being examined (when you click on the table),
we could follow the click event and change the color of that cell temporarily.
"""

class App(widgets.QWidget):
	def __init__(self, folder_path, image_pattern, update_image, property_choices, make_export_handler, folder_select_button):
		super().__init__()
		self.property_choices = property_choices
		self._table_widget = None
		self.initUI(folder_path, image_pattern, update_image, make_export_handler, folder_select_button)
		self._layer = None
		self._make_export_handler = make_export_handler
		update_image(None, None, None)

	def initUI(self, folder_path, image_pattern, update_image, make_export_handler, folder_select_button):
		self.setGeometry(700, 100, 350, 380)
		self.layout = widgets.QVBoxLayout()
		self.layout.setAlignment(Qt.AlignTop)
		self.refresh_table_button = widgets.QPushButton('Refresh Annotations Table', self)
		self.save_button = widgets.QPushButton('Save Annotation Updates', self)

		self.selected_folder_label = self._make_selected_folder_label(folder_path)
		self.layout.addWidget(self.selected_folder_label)
		self.layout.addWidget(self.save_button)
		self.layout.addWidget(self.refresh_table_button)
		self.layout.addWidget(folder_select_button)

		def handle_image_selection(image_file_name, image_file_path, annot_file_path):
			layer = update_image(image_file_name, image_file_path, annot_file_path)
			self._layer = layer
			self._refresh_table_widget()
			self.setLayout(self.layout)

		image_selector = ImageSelector(folder_path, image_pattern, on_item_selected=handle_image_selection, make_export_handler=make_export_handler)
		self.layout.addWidget(image_selector)

		self.setLayout(self.layout)
		self.refresh_table_button.clicked.connect(self._refresh_table_widget)
		self.save_button.clicked.connect(self._save_changes)

	def _make_selected_folder_label(self, folder_path):
		folder_name = os.path.basename(folder_path)
		selected_folder_label = widgets.QLabel(f'Selected folder: {folder_name}')
		selected_folder_label.setToolTip(folder_path)

		selected_folder_label.setStyleSheet("""
		    QLabel {
		        font-family: "Arial";
		        font-size: 14px;
		        font-weight: bold;
		        color: #ffffff;
		        background-color: #3a9bce;
		        padding: 5px;
		        border-radius: 5px;
		    }
		""")

		return selected_folder_label

	def get_layer(self):
		return self._layer

	def get_property_choices(self):
		return self.property_choices

	def _clear_table_widget(self):
		if self._table_widget is None:
			return
		self.layout.removeWidget(self._table_widget)
		self._table_widget.setParent(None)
		self._table_widget.deleteLater()
		self._table_widget = None

	def _save_changes(self):
		export_handler = self._make_export_handler()
		if export_handler and export_handler.is_updated():
			export_handler.export_to_file()

	def _add_table_widget(self):
		layer = self._layer
		if layer and layer.features.shape[0] > 0:
			self._table_widget = AnnotationTable(self)
			self.layout.addWidget(self._table_widget)

	def _refresh_table_widget(self):
		self._clear_table_widget()
		self._add_table_widget()
