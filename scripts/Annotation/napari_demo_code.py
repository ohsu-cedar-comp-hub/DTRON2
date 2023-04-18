"""
Author: Christopher Z. Eddy
Date: 02/07/23
"""

import napari
import os
import glob
import numpy as np
from widgets.app import App
from input_handler import InputHandler
from annotation_export_handler import AnnotationExportHandler
from util.file import lazy_read, load_json_data, zarr_read
from util.annotation import make_annotation_data
from event_filter.close_event_filter import CloseEventFilter
from qtpy.QtWidgets import QFileDialog
from widgets.folder_select_button import FolderSelectButton

#import psutil

"""
BUGS:

UNRESOLVED

RESOLVED
	####
	  If you delete ALL the shapes and then put a new one, I think the properties and features is lost.
	  Actually, the properties and features is fine. The table gets messed up, I think it isn't well adjusted if there is no entry.
	  Ex: Delete all cells, update table (fine). Add a new shape, update table and you get nans.
	  Oh.. My guess is something in set content goes awry.
	####
	  If starting from scratch on a new image, to make the table work, we need to initiate a shape layer. However, we can't pass
	  an empty array of annotations to the shape layer, moreover from the error above I know that is a problem. So it seems we either cannot
	  run the app if the shapes layer has not been intiated, or we need to wait to set its properties.


"""


##################################################################

viewer = napari.Viewer()

base_dir = os.getcwd()
relative_folder_path = "../../data/annotation/napari_ex"

curr_folder_path = os.path.join(base_dir, relative_folder_path)
curr_folder_path = os.path.abspath(curr_folder_path)

curr_image_file_name, curr_image_file_path, curr_annot_file_path = None, None, None

def get_layer(layer_name):
	if layer_name in viewer.layers:
		return viewer.layers[layer_name]

	return None

def make_export_handler():
	if not curr_annot_file_path:
		return None

	shapes_layer = get_layer('Shapes')
	labels_layer = get_layer('Labels')

	export_handler = AnnotationExportHandler(curr_annot_file_path, curr_image_file_name, shapes_layer=shapes_layer, labels_layer=labels_layer)
	return export_handler

close_event_filter = CloseEventFilter(viewer, make_export_handler)

# Install the custom event filter to the viewer's QMainWindow
viewer.window._qt_window.installEventFilter(close_event_filter)

#send to viewer
# specify contrast_limits and multiscale=False with big data
property_choices = {'class': [1,2], 'anno_style': ['manual','auto'], 'metadata': None}

def update_image(image_file_name, image_file_path, annot_file_path):
	global curr_image_file_name, curr_image_file_path, curr_annot_file_path

	curr_image_file_name, curr_image_file_path, curr_annot_file_path = image_file_name, image_file_path, annot_file_path
	layers = list(viewer.layers)
	for layer in layers:
		viewer.layers.remove(layer)
	
	if image_file_path is None:
		return None

	if "ome.tif" in os.path.basename(image_file_path):
		img_layer = viewer.add_image(zarr_read(image_file_path), name=image_file_name, contrast_limits=[0,255], rgb=True, multiscale=True)
	else:
		img_layer = viewer.add_image(lazy_read([image_file_path]),name=image_file_name, contrast_limits=[0,65000], rgb=False, multiscale=False)

	if os.path.exists(annot_file_path):
		#pull annotation data from json
		anno_data = load_json_data(annot_file_path)
	else:
		#will want to construct anno_data with the proper fields "image_name", "image_shape", "meta"
		anno_data = {'image_name':image_file_name,
					 'annotation': [],
					 'features':{"anno_style":[], "class":[], "metadata":[]},
					 }

	###############################

	annotations = anno_data["annotation"] #should be in the form of a list of numpy arrays (Z x Row x Col)
	annotations = [np.array(x) for x in annotations]
	shape_features = anno_data["features"]#should be a dictionary with keys 'class', 'anno_style'
	#determine colors; should be equal to number of classes.

	face_color_cycle=['royalblue','green'] #WILL REPRESENT CLASS, assuming only two classes. In the future we can import a cycling color library, colorcet.
	edge_color_cycle=['red','blue']

	# specify the display parameters for the text
	text_parameters = {
		'string': 'label: {class}\n{anno_style}',
		'size': 8,
		'color': 'green',
		'anchor': 'upper_left',
		'translation': [-3, 0],
		}
	data = annotations or None 
	"""
	Sending None here causes an 'invalid value encountered in cast' error, associated with 'corners[:, displayed_axes] = data_bbox_clipped'
	However, this does not break anything, just a minor annoyance.
	Code Example:
	viewer = napari.Viewer()
	viewer.add_shapes(data=None, shape_type='polygon', name="Annotations")
	"""
	shapes_layer = viewer.add_shapes(data=data, shape_type='polygon', name="Shapes",
		features = shape_features,
		properties = shape_features,
		edge_color='anno_style',
		edge_color_cycle = edge_color_cycle,
		edge_width=2,
		face_color='class',
		face_color_cycle=face_color_cycle,
		property_choices = property_choices,
		opacity=0.4,
		text = text_parameters)

	# modify the default feature values
	shapes_layer.feature_defaults['anno_style']='manual'
	shapes_layer.feature_defaults['class']=1
	shapes_layer.feature_defaults['metadata']=''
	
	return shapes_layer

pattern = [".tif", ".png",  ".ome.tif"]

def on_folder_selected(selected_folder_path):
	global curr_folder_path
	curr_folder_path = selected_folder_path
	folder_select_button.setParent(None)
	refresh_dock_app()


folder_select_button = FolderSelectButton(text="Update Image Folder", initial_directory=curr_folder_path, on_folder_selected=on_folder_selected, make_export_handler=make_export_handler)

dock_app = None

def make_dock_app():
	dock_app = App(curr_folder_path, pattern, update_image, property_choices, make_export_handler, folder_select_button=folder_select_button)
	return dock_app

def refresh_dock_app():
	global dock_app
	if dock_app:
		viewer.window.remove_dock_widget(dock_app)
	dock_app = make_dock_app()
	viewer.window.add_dock_widget(dock_app)

refresh_dock_app()

napari.run()
