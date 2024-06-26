import numpy as np
import qtpy.QtWidgets as widgets
import logging

class AnnotationTable(widgets.QTableWidget):
    def __init__(self, parent):
        super().__init__()
        self._layer = parent.get_layer()
        self.property_choices = parent.get_property_choices()
        self.class_dict = {i: x for (i, x) in zip(self.property_choices['class'], self._layer.face_color_cycle)}
        self.anno_dict = {i: x for (i, x) in zip(self.property_choices['anno_style'], self._layer.edge_color_cycle)}

        if hasattr(self._layer, "properties"):
            self.set_content(self._layer.properties, init=True)
        else:
            self.set_content({})

        self.cellChanged.connect(self.onCellChanged)
        self.cellClicked.connect(self.onCellClicked)
        #self.currentItemChanged.connect(self.set_selected_cell_background)


    def onCellChanged(self, row, column):
        text = self.item(row, column).text()
        C = list(self._table.keys())[column]
        #ALSO, if MULTIPLE OBJECTS ARE SELECTED WHILE BEING CHANGED, CHANGE ALL SIMULTANEOUSLY
        #should be in self._layer.selected_data
        try:
            entry = float(text)
        except:
            entry = text

        if not isinstance(entry, str):
            if np.isnan(entry):
                entry = self.property_choices[C][0]

        #what if we are editing, but the row was not in selected data?
        #add the row to selected data.
        if row not in self._layer.selected_data:
            self._layer.selected_data = set(list(self._layer.selected_data)+[row])
            self._layer.stored_selection = list(set(list(self._layer.selected_data)+[row]))
            self.selectRow(row)
        
        for row in self._layer.selected_data:
            if C != "metadata":
                if entry not in self.property_choices[C]:
                    logging.warn(f"Annotation {row} with {C} '{entry}' not in options {self.property_choices[C]}")
                    self.setItem(row, column, widgets.QTableWidgetItem(str(self._table[C][row])))
                    return
                else:
                    self._table[C][row] = entry
            else: #if it is metadata, it can be anything. Set the content as is.
                self._table[C][row] = entry

        self.set_content(table = self._table)
    
    def onCellClicked(self, row, column):
        
        if row not in self._layer.selected_data:
            #Sometimes, you may have began a polygon that you never completed, or escaped out of. In which case, this row
            #was made in the datatable but it no longer exists in the shape layer. We try to store it, if shape object exists.
            try:
                self._layer.selected_data = set(list(self._layer.selected_data)+[row])
                self._layer.stored_selection = list(set(list(self._layer.selected_data)+[row]))
            except: 
                self.removeRow(row)
                #for some reason, removing the row then adds it to selected_data. remove it here. 
                #we deselected it. We should toggle the selection on the layer.
                good_data = list(set([x for x in self._layer.selected_data if x!=row]))
                self._layer.selected_data = set(good_data)
        else:
            #we deselected it. We should toggle the selection on the layer.
            good_data = list(set([x for x in self._layer.selected_data if x!=row]))
            self._layer.selected_data = set(good_data)
            self._layer.stored_selection = good_data
            #toggle the layer. 

    def set_content(self, table: dict, init=False):
        if table is None:
            table = {}

        if "label" in table.keys() and "index" not in table.keys():
            table["index"] = table["label"]

        def get_status(position, *, view_direction=None, dims_displayed=None, world: bool = False) -> str:
            value = self._layer.get_value(position, view_direction=view_direction, dims_displayed=dims_displayed, world=world)
            from napari.utils.status_messages import generate_layer_status
            msg = generate_layer_status(self._layer.name, position, value)
            return msg

        self._layer.get_status = get_status

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
            self._layer.features = table
            self._layer.face_color = np.array([self.class_dict[x] for x in self._layer.features['class'].tolist()])
            self._layer.edge_color = np.array([self.anno_dict[x] for x in self._layer.features['anno_style'].tolist()])
            #note here that it doesn't actualy set the items. Doing a setItem call causes OnCellChanged to be triggered,
            #which 