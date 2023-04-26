import qtpy.QtWidgets as widgets
from qtpy.QtCore import Qt
import numpy as np
from napari import Viewer, gui_qt
from .image_selector import ImageSelector
from .annotation_table import AnnotationTable
import os
import time
"""
Added auto table update when removed or added shapes object.
Need to add 
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
		self.stored_selection = []

	def initUI(self, folder_path, image_pattern, update_image, make_export_handler, folder_select_button):
		self.setGeometry(700, 100, 350, 380)
		self.layout = widgets.QVBoxLayout()
		self.layout.setAlignment(Qt.AlignTop)
		self.save_button = widgets.QPushButton('Save Annotation Updates', self)

		self.selected_folder_label = self._make_selected_folder_label(folder_path)
		self.layout.addWidget(self.selected_folder_label)
		self.layout.addWidget(self.save_button)
		self.layout.addWidget(folder_select_button)

		def handle_image_selection(image_file_name, image_file_path, annot_file_path):
			layer = update_image(image_file_name, image_file_path, annot_file_path)
			self._layer = layer
			self._refresh_table_widget()
			self.setLayout(self.layout)
			# Attach the callback function to the shapes layer's events
			#self._layer.events.highlight.connect(self.on_polygon_click) # works, but overactive
			if self._layer:
				self._layer.events.highlight.connect(self.on_polygon_click) # works, but overactive
				self.stored_selection = []
				self._layer.events.data.connect(self.onEditedShapes)
				#self._layer.events.current_properties.connect(self.on_polygon_click) #activates when the current properties change
				#note that current_properties won't update if you select an object with the same properties.
				if self._table_widget:
					self._table_widget.cellChanged.connect(self.onCellChanged)
	

		image_selector = ImageSelector(folder_path, image_pattern, on_item_selected=handle_image_selection, make_export_handler=make_export_handler)
		self.layout.addWidget(image_selector)

		self.setLayout(self.layout)
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
	
	def onCellChanged(self, row, column):
		if self._layer and self._table_widget:
			if len(self._layer.selected_data)>1:
				self._refresh_table_widget()

	def onEditedShapes(self):
		self._refresh_table_widget()

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
			self._table_widget.cellChanged.connect(self.onCellChanged)

	def _refresh_table_widget(self):
		if self._layer and hasattr(self._layer, 'stored_selection'):
			self.stored_selection = self._layer.stored_selection
		self._clear_table_widget()
		self._add_table_widget()
		if self._table_widget:
			self._table_widget.setSelectionBehavior(widgets.QAbstractItemView.SelectRows) 
			self._table_widget.setSelectionMode(widgets.QAbstractItemView.MultiSelection)
			self._table_widget.setEditTriggers(widgets.QAbstractItemView.DoubleClicked)
			self._table_widget.setFocusPolicy(Qt.NoFocus)
			self._layer.selected_data = set(self.stored_selection)
			self._layer.stored_selection = self.stored_selection
			for rr in self._layer.selected_data:
				self._table_widget.selectRow(rr)

	def on_polygon_click(self, event):
		layer = self._layer
		if layer and layer.features.shape[0]>0:
			if self._table_widget:
				# Check if any shapes were clicked on
				indices = layer.selected_data
				if indices:
					#if len(indices)>=len(self.stored_selection):
					newrow = [r for r in list(indices) if r not in layer.stored_selection]
					rmrow = [r for r in layer.stored_selection if r not in list(indices)]
					toggle_rows = newrow+rmrow 
					if toggle_rows:
						for rr in toggle_rows:
							self._table_widget.selectRow(rr)
						
						self._layer.stored_selection = list(indices)
						self.stored_selection = list(indices)
				else:
					#sometimes the table elements get left turned on 
					#they are left in self._layer.stored_selection, and the self.stored_selection may be wrong.
					rmrow = [r for r in layer.stored_selection]
					if rmrow:
						for rr in rmrow:
							self._table_widget.selectRow(rr)
						self._layer.stored_selection = []
						self.stored_selection = [] 
				###Removed this code, but it scans the datatable as finds the active rows, if desired.
				# selected_ranges = self._table_widget.selectedRanges()
				# selected_rows = set()
				# for selected_range in selected_ranges:
				# 	for row in range(selected_range.topRow(), selected_range.bottomRow()+1):
				# 		selected_rows.add(row)
				# #if any in selected rows that are not in self.stored_selection, toggle them off.
				# addto_deck = [x for x in self.stored_selection if x not in selected_rows]
				# rmfrom_deck = [x for x in selected_rows if x not in self.stored_selection]
				# toggle_rows = addto_deck + rmfrom_deck
				# if toggle_rows:
				# 	print('deck:{}'.format(toggle_rows))
				# 	for rr in rmfrom_deck:
				# 			self._table_widget.selectRow(rr)
			

				






