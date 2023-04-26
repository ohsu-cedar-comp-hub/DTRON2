"""
Author: Christopher Z. Eddy
Date: 02/07/23
"""

import napari
import os
import numpy as np
import colorcet as cc
from widgets.app import App
from annotation_export_handler import AnnotationExportHandler
from util.file import lazy_read, load_json_data, zarr_read
from event_filter.close_event_filter import CloseEventFilter
from widgets.folder_select_button import FolderSelectButton
import vispy.color

"""
Currently, loading an empty viewer image results in a runtimewarning, likely due to a version bug.
For now, we filter and ignore those warnings so they do not interere with our try, except statements.
"""
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

"""
Future updates:
It would be nice to select an annotation from the image and see it highlighted on the datatable.

Still need to figure out the default settings issue. It isn't a huge problem but just a pain. 
It seems resetting feature_defaults doesn't do anything.
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
property_choices = {'class': list(range(20)), 'anno_style': ['manual','auto'], 'metadata': None}
cmap = cc.glasbey_category10

def update_image(image_file_name, image_file_path, annot_file_path):
	global curr_image_file_name, curr_image_file_path, curr_annot_file_path

	curr_image_file_name, curr_image_file_path, curr_annot_file_path = image_file_name, image_file_path, annot_file_path

	layers = list(viewer.layers)
	for layer in layers:
		viewer.layers.remove(layer)
	
	if image_file_path is None:
		return None

	if "ome.tif" in os.path.basename(image_file_path):
		try:
			IM, _ = zarr_read(image_file_path, split_channels=False)
			img_layer = viewer.add_image(IM, name=image_file_name, contrast_limits=[0, 255], rgb=True, multiscale=True)
			"""
			Exiting due to RuntimeWarning: invalid value encountered in cast
  							corners[:, displayed_axes] = data_bbox_clipped
			when loading with no annotations
			"""
		except:
			#load cyclic IF data and display the channels independently.
			#the alternative is just to call 
			#img_layer = viewer.add_image(zarr_read(image_file_path), name=image_file_name, contrast_limits=[0,255], rgb=False, multiscale=True)
			#which will pull up a slider below the image.
			IM, ch_names = zarr_read(image_file_path, split_channels=True)
			#https://forum.image.sc/t/viewing-channel-name-in-multi-channel-image/35830/4
			contrast_max = (255 if IM[0][0].dtype==np.uint8 else 65535)
			for ch_i, channel in reversed(list(enumerate(ch_names))):
				this_cmap = vispy.color.Colormap([[0.0,0.0,0.0], cmap[ch_i]])
				img_layer = viewer.add_image(IM[ch_i], name=channel, contrast_limits=[0,contrast_max], rgb=False, multiscale=True, visible=(True if ch_i==0 else False), blending='additive', colormap = this_cmap)
	else:
		try:
			img_layer = viewer.add_image(lazy_read([image_file_path]),name=image_file_name, contrast_limits=[0,2000], rgb=True, multiscale=False)
		except:
			img_layer = viewer.add_image(lazy_read([image_file_path]),name=image_file_name, contrast_limits=[0,2000], rgb=False, multiscale=False)

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
	shape_features = anno_data["features"]#should be a dictionary with keys 'class', 'anno_style', 'metadata'
	#determine colors; should be equal to number of classes.
	curr_props = {'class':0, 'anno_style':'manual', 'metadata':''}

	# face_color_cycle=['royalblue','green'] #WILL REPRESENT CLASS, assuming only two classes. In the future we can import a cycling color library, colorcet.
	# edge_color_cycle=['red','blue']

	face_color_cycle = [cmap[x] for x in range(len(property_choices['class']))]
	edge_color_cycle = [cmap[x] for x in range(len(property_choices['anno_style']))]

	edge_color_inds = [property_choices['anno_style'].index(x) if x in property_choices['anno_style'] else 0 for x in shape_features['anno_style']]
	face_color_inds = [property_choices['class'].index(x) if x in property_choices['class'] else 0 for x in shape_features['class']]
	edge_colors = [edge_color_cycle[i] for i in edge_color_inds]
	face_colors = [face_color_cycle[i] for i in face_color_inds]
	if len(edge_colors)==0:
		edge_colors='anno_style'
	if len(face_colors)==0:
		face_colors='class'

	# specify the display parameters for the text
	text_parameters = {
		'string': 'class: {class}\n{anno_style}',
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
		edge_color = edge_colors,
		edge_color_cycle = edge_color_cycle,
		edge_width=2,
		face_color = face_colors,
		face_color_cycle = face_color_cycle,
		property_choices = property_choices,
		opacity=0.4,
		text = text_parameters)

	if not data:
		#need some default behavior if no previous shape exists to draw from.
		shapes_layer.feature_defaults['class']=0
		shapes_layer.feature_defaults['anno_style']='manual'
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

#look for interactive shell, if not do not run napari.run()
import sys
if not bool(getattr(sys, 'ps1', sys.flags.interactive)):
	napari.run() #commenting this out enables more interactivity at the command line
