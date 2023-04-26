import skimage.measure
from skimage.measure import approximate_polygon
import numpy as np

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
		if _calc_poly_area(segmentation_x,segmentation_y)>=thresh:
			polygons_x.append(segmentation_x)
			polygons_y.append(segmentation_y)
			a.append(_calc_poly_area(segmentation_x,segmentation_y))

	#the vertices are read flipped in skimage.measure.find_contours, we will flip them here....
	vertices = [[[yi,xi] for (xi,yi) in zip(X,Y)] for (X,Y) in zip(polygons_x,polygons_y)]

	#reduce the number of vertices
	for i,anno in enumerate(vertices):
		verts = np.stack(anno,axis=0)
		#reduce the amount of vertices using approximate.
		verts = approximate_polygon(verts, tolerance = 2)
		#convert back to a list and replace
		vertices[i] = verts.tolist()

	return [vertices,a]

def _calc_poly_area(x,y):
	"""
	Determine the area given vertices in x, y.
	x, y can be numpy array or list of points. Automatically closes the polygons
	Uses shoestring formula.
	"""
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
