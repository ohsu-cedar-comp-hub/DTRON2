import qtpy.QtWidgets as widgets
import numpy as np

"""
Some thoughts:
To make it more obvious which cell is being examined (when you click on the table), 
we could follow the click event and change the color of that cell temporarily. 
"""

class TableWidget(widgets.QTableWidget):
	def __init__(self, parent):
		super().__init__()
		self._layer = parent._layer
		self.property_choices = parent.property_choices
		self.class_dict = {i:x for (i,x) in zip(self.property_choices['class'],self._layer.face_color_cycle)}
		self.anno_dict = {i:x for (i,x) in zip(self.property_choices['anno_style'],self._layer.edge_color_cycle)}

		if hasattr(self._layer, "properties"):
			self.set_content(self._layer.properties, init=True)

		else:
			self.set_content({})
 
		self.cellChanged.connect(self.onCellChanged) #this is VERY sensitive. Changing in a loop can be bad.
 
	def onCellChanged(self, row, column):
		text = self.item(row, column).text()
		#Make sure content is valid. 
		C = list(self._table.keys())[column]
		try:
			entry = float(text)
		except:
			entry = text
		
		#what if entry is NaN, which happens when we go from filled to empty?

		if not isinstance(entry,str): #if it is a string, continue
			if np.isnan(entry): #if it is nan, change, otherwise continue
				entry = self.property_choices[C][0] #default.

		if entry not in self.property_choices[C]:
			print("Annotation {} with {} '{}' not in options {}".format(row, C, entry, self.property_choices[C]))
			#return back to original value:
			self.setItem(row, column, widgets.QTableWidgetItem(str(self._table[C][row])))
			return
		else:
			self._table[C][row] = entry

			self.set_content(table = self._table)

	def set_content(self, table : dict, init=False):
		"""
		Overwrites the content of the table with the content of a given dictionary.
		
		table should look like {'hi': ['a', 'b'], 'hi2': ['d', 'c']}
		"""
		if table is None:
			table = {}

		# Workaround to fix wrong row display in napari status bar
		# https://github.com/napari/napari/issues/4250
		# https://github.com/napari/napari/issues/2596
		if "label" in table.keys() and "index" not in table.keys():
			table["index"] = table["label"]

		# workaround until these issue are fixed:
		# https://github.com/napari/napari/issues/4342
		# https://github.com/napari/napari/issues/5417
		# if len(np.unique(table['index'])) != len(table['index']):
		def get_status(
				position,
				*,
				view_direction=None,
				dims_displayed=None,
				world: bool = False,
		) -> str:
			value = self._layer.get_value(
				position,
				view_direction=view_direction,
				dims_displayed=dims_displayed,
				world=world,
			)

			from napari.utils.status_messages import generate_layer_status
			msg = generate_layer_status(self._layer.name, position, value)
			return msg
		
		# disable napari status bar because it increases the window size, which makes zero sense
		self._layer.get_status = get_status
		#print('Napari status bar display of label properties disabled because https://github.com/napari/napari/issues/5417 and https://github.com/napari/napari/issues/4342')

		self._table = table

		self._layer.properties = table

		if init:
			self.clear()
			try:
				self.setRowCount(len(next(iter(table.values()))))
				self.setColumnCount(len(table))
			except StopIteration:
				pass

			for i, column in enumerate(table.keys()):
				self.setHorizontalHeaderItem(i, widgets.QTableWidgetItem(column))
				for j, value in enumerate(table.get(column)):
					self.setItem(j, i, widgets.QTableWidgetItem(str(value)))

		else:
			#update colors by updating features as well.
			self._layer.features = table
			self._layer.face_color = np.array([self.class_dict[x] for x in self._layer.features['class'].tolist()])
			self._layer.edge_color = np.array([self.anno_dict[x] for x in self._layer.features['anno_style'].tolist()])
		
 
 
class App(widgets.QWidget):
	def __init__(self, property_choices, layer: "napari.layers.Layer", viewer: "napari.Viewer" = None):
		super().__init__()
		self._layer = layer 
		self._viewer = viewer 
		self.property_choices = property_choices
		self.has_tab_widget=False
		self.initUI()
 
	def initUI(self):
		self.setGeometry(700, 100, 350, 380)
		self.layout = widgets.QVBoxLayout()
		self.button = widgets.QPushButton('I added/deleted annotations', self)
		self.layout.addWidget(self.button)
		if self._layer.features.shape[0] > 0:
			self.tableWidget = TableWidget(self)
			self.layout.addWidget(self.tableWidget)
			self.has_tab_widget = True
		else:
			self.has_tab_widget = False
		self.setLayout(self.layout)
		self.button.clicked.connect(self.changed_num_annotations)
 
	def changed_num_annotations(self):
		if self.has_tab_widget:
			self.layout.removeWidget(self.tableWidget)
			self.has_tab_widget=True
		if self._layer.features.shape[0] > 0:
			self.tableWidget = TableWidget(self)
			self.layout.addWidget(self.tableWidget)
			self.has_tab_widget=True 
		