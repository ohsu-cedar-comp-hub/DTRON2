"""
Author: Christopher Z. Eddy
Date: 02/07/23
"""

import napari 
from skimage.io import imread
import skimage.measure
import os
import glob 
import json
import numpy as np
from dask import delayed
import dask.array as da
import app_widget as tw 
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

def import_filenames(filepath, pattern, recursive = False):
		"""
		PURPOSE: Imports ALL files from a chosen folder
		INPUTS
		-------------------------------------------------------------
		pattern = list of strings with particular patterns, including filetype! 
				ex: ["_patched",".csv"] will pull any csv files under filepath with the string "_patched" in its file name.

		filepath = string, path for where to search for image files 
				ex: "/users/<username>/folder"

		recursive = boolean, True if you wish for the search for files to be recursive under filepath.
		"""
		#generate pattern finding 
		fpattern = ["**{}".format(x) for i,x in enumerate(pattern)]
		fpattern = "".join(fpattern)
		#first, get all the filenames in the chosen directory.
		#root directory needs a trailing slash
		if filepath[-1]!="/":
				filepath = filepath+"/"
		fnames = [filename for filename in glob.iglob(filepath + "**/" + fpattern, recursive=recursive)] #best for recursive search one liner.
		fnames = [x for x in fnames if x[0]!="."] #delete hidden files
		fnames.sort() #sort based on name
		return fnames

def load_json_data(pth):
	with open(pth) as f:
		data=json.load(f)
	return data

class NpEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(NpEncoder, self).default(obj)


def load_pair(filenames, i):
	im_name = filenames[i]
	anno_name = os.path.basename(im_name).split(".")[0]+".json" #there is a big assumption that the filename up to the extension does not contain a "." character
	anno_path = os.path.dirname(os.path.dirname(im_name))+"/annotations"
	anno_name = os.path.join(anno_path, anno_name)
	return im_name,anno_name

def PolyArea(x,y):
	"""
	Determine the area given vertices in x, y.
	x, y can be numpy array or list of points. Automatically closes the polygons
	Uses shoestring formula.
	"""
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def binary_mask_to_polygon_skimage(binary_mask,thresh=250):
	"""
	thresh is the areas threshold requirement
	"""
	#we want to pad binary_mask one on each side. Then subtract the same pad from each.
	if binary_mask.dtype=='bool':
		binary_mask = np.pad(binary_mask,((1,1),(1,1)),constant_values=(False,False))
	else:
		binary_mask = np.pad(binary_mask,((1,1),(1,1)),constant_values=(0,0))

	polygons_x = []
	polygons_y = []
	contours = skimage.measure.find_contours(binary_mask, 0.5, fully_connected='high') #see documentation for 0.5
	a=[]
	for contour in contours:
		contour = np.flip(contour, axis=1)
		if len(contour) < 3:
			continue
		segmentation_x = contour[:,0].tolist()
		segmentation_y = contour[:,1].tolist()
		segmentation_x = [0 if i-1 < 0 else i-1 for i in segmentation_x] # resolving indexing issues
		segmentation_y = [0 if i-1 < 0 else i-1 for i in segmentation_y]
		# after padding and subtracting 1 we may get -0.5 points in our segmentation
		#if the threshold area is too low, do not include it
		if PolyArea(segmentation_x,segmentation_y)>=thresh:
			polygons_x.append(segmentation_x)
			polygons_y.append(segmentation_y)
			a.append(PolyArea(segmentation_x,segmentation_y))

	#the vertices are read flipped in skimage.measure.find_contours, we will flip them here....
	vertices = [[[yi,xi] for (xi,yi) in zip(X,Y)] for (X,Y) in zip(polygons_x,polygons_y)]

	return [vertices,a]

def let_user_pick(options):
	for idx, element in enumerate(options):
		print("{}) {}".format(idx+1, element))
	print("0) Quit".format(len(options)+1))
	
	i = input("Enter number: ")
	
	if 0 < int(i) <= len(options):
		return int(i)-1
	
	elif int(i)==0:
		return -1
	else:
		return None
	
def lazy_read(filenames):
	#note that files can be multichanneled, hence 'filenames'
	#import pdb;pdb.set_trace()
	sample = imread(filenames[0]) #read the first file to get the shape and dtype, which 
	#we ASSUME THAT ALL OTHER FILES SHARE THE SAME SHAPE/TYPE
	lazy_imread = delayed(imread)
	lazy_arrays = [lazy_imread(fn) for fn in filenames]
	dask_arrays = [
		da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype) 
		for delayed_reader in lazy_arrays
	]
	#Stack into one large dask.array.
	stack = da.stack(dask_arrays,axis=0)
	#stack.shape = (nfiles, nz, ny, nx)

	## For Multichannel data, we'll do something like this:
	# file_pattern = "/path/to/experiment/*ch{}*.tif"
	# channels = [imread(file_pattern.format(i)) for i in range(nchannels)]
	# stack = da.stack(channels)
	# stack.shape  # (2, 600, 64, 256, 280)

	return stack

def get_all_channels(filename, filenames):
	#assuming we have identified just the unique filenames. 
	all_channels = [x for x in filenames if (os.path.basename(filename)[:-4] in os.path.basename(x)[:-4])]
	return all_channels


##################################################################

viewer = napari.Viewer()

#find the files within the working directory.
filenames = import_filenames("../../data/napari_ex", pattern=[".png"], recursive=False)
#now, we will assume the annotation files have the same filename structure, but will be saved as a json for convenience. 

if len(filenames)>0:
	cont = True
else:
	print("No filenames found under directory.")
	cont=False

##################################################################
while cont:
	print("Please choose an option by typing into the command line:")
	ind = let_user_pick(filenames)
	while ind is None:
		print("Entry not in list. Please choose again:")
		ind = let_user_pick(filenames)
	if ind >= 0:
		im_name,anno_name = load_pair(filenames, ind)
		print("Loading {}...".format(os.path.basename(im_name)))
		#send to viewer
		# specify contrast_limits and multiscale=False with big data
		viewer.add_image(lazy_read([im_name]),name=os.path.basename(im_name),contrast_limits=[0,2000], multiscale=False)
		if os.path.exists(anno_name):
			#pull annotation data from json
			anno_data = load_json_data(anno_name)
		else:
			#will want to construct anno_data with the proper fields "image_name", "image_shape", "meta"
			anno_data = {'image_name':os.path.basename(im_name),
						 'annotation': [], 
						 'features':{"anno_style":[], "class":[]},
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
			'size': 6,
			'color': 'green',
			'anchor': 'upper_left',
			'translation': [-3, 0],
			}

		property_choices = {'class': [1,2], 'anno_style': ['manual','auto']} #doesn't do anything.

		#Add shapes!
		if len(annotations)>0:
			viewer.add_shapes(annotations, shape_type='polygon', name="Shapes", 
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
		else:
			viewer.add_shapes(data=None, shape_type='polygon', name="Shapes", 
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
		viewer.layers['Shapes'].feature_defaults['anno_style']='manual'
		viewer.layers['Shapes'].feature_defaults['class']=1

		dock_app = tw.App(property_choices,viewer.layers["Shapes"], viewer)
		viewer.window.add_dock_widget(dock_app)

		#####################################
		#Prompt the user to press enter once they are done adding annotations
		input("Press 'Enter' if you are done editing annotations...\n")
		#####################################

		"""
		NEED TO FIGURE OUT HERE HOW TO NOT ADD THE DOCK EACH TIME....
		"""
		##print the memory usage.
		#print('RAM memory % used:', psutil.virtual_memory()[2])
		viewer.window.remove_dock_widget(dock_app) #this might get "expensive" in memory IF there is a leak.

		"""
		PULL ANNOTATION DATA
		"""

		#Some people prefer to use polygons to annotate their images. Here, we pull that data.
		try:
			new_polygons = True
			shape_data = viewer.layers['Shapes'].data
			shape_data = [[list(y) for y in x] for x in shape_data] #convert to a list of lists. shape_data will be len == # of annotations, and each sublist (shape_data[0]) will have len == # vertices
			anno_data['annotation'] = shape_data


			shape_features = viewer.layers['Shapes'].features
			if shape_features.shape[0]>0: #if an annotation was made. 
				#convert
				shape_features = shape_features.to_dict()
				#MAKE SURE LOOKS RIGHT. {key:[D[key][i] for i in list(D[key].keys())] for key in list(D.keys()) for x in D[key]}
				anno_data["features"] = {key:[shape_features[key][i] for i in list(shape_features[key].keys())] for key in list(shape_features.keys())}
			viewer.layers.remove('Shapes')
		except:
			new_polygons=False
			print("No layer named 'Shapes' found!")

		#Some people prefer to paintbrush their labels. I will here take that data and store it as polygons.
		try:
			new_brush=True
			brush_data = viewer.layers['Labels'].data # label data same size as brush.
			#Let's assume there are more than 1 class ()
			####
			#https://napari.org/stable/gallery/add_shapes_with_features.html
			####
			brush_data = brush_data[0,:,:] #> 0. #convert to binary, note that brush data loads as a (N x H x W) image. Since we aren't dealing with 3D data...
			allverts = []
			allclasses = []
			#work one object at a time, as that is how I wrote the function.
			for anno_i in range(1,int(np.max(brush_data))+1):
				[vertices,_] = binary_mask_to_polygon_skimage(brush_data==anno_i,thresh=10)
				allverts += vertices
				allclasses += [anno_i for _ in vertices]

			#all annotations were manual. 
			if not new_polygons: #if we didn't already add polygons from shape data
				anno_data['annotation'] = allverts
				anno_data['features']['class'] = allclasses
				anno_data['features']['anno_style'] = ['manual' for _ in allclasses]
			else:
				anno_data['annotation'] = anno_data['annotation']+allverts
				anno_data['features']['class'] = anno_data['features']['class'] + allclasses
				anno_data['features']['anno_style'] = anno_data['features']['anno_style'] + ['manual' for _ in allclasses]
			viewer.layers.remove('Labels')
		except:
			new_brush=False
			print("No layer named 'Labels' found!")
		
		#remove the image.
		viewer.layers.remove(os.path.basename(im_name))

		if new_brush or new_polygons:
			#save the result.
			with open(anno_name,'w') as output_json_file:
				json.dump(anno_data, output_json_file, cls=NpEncoder)

	else:
		cont=False
		

"""
MWE:
## EDIT LINE 177 ON THIS CODE FOR WHERE DATA DIRECTORY IS, and TYPE OF DATA (.TIF vs .PNG). 
## Data directory must follow as 
# DataDir
#	-> /images
#	-> /annotations

#Command line
python -i napari_demo_code.py

"""