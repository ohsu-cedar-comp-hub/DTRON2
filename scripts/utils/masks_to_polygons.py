"""
AUTHOR: Christopher Z. Eddy (eddyc@ohsu.edu)
Date: 04/19/23
PURPOSE: Convert completed masks into useable JSON files for Napari.
Reads from annotation_dir and image_dir assuming same naming conventions.
Saves output polygons in JSON files.

MWE
python -i masks_to_polygons.py --image_dir="<path_to_tifs>"
--annotation_dir = "<path_to_annotated_tifs>"
--save_dir = "<path_to_where_you_want_to_save>"
"""

import tifffile
import numpy as np
from skimage import measure, draw
from skimage.measure import approximate_polygon
import glob
import json
import os
import sys


############################################################################
############################################################################

############DEFINE NECESSARY FUNCTIONS AND CLASSES##########################

def PolyArea(x,y):
    """
    Determine the area given vertices in x, y.
    x, y can be numpy array or list of points. Automatically closes the polygons
    Uses shoestring formula.
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def mask_to_polygon_skimage(binary_mask,thresh=250):
    """
    Convert numpy array containing annotations to polygons.
    
    INPUTS
    ------------------------------
    binary_mask = np.array of shape = (H x W), dtype can be boolean or integers, where each annotation is a different integer
    thresh = float, minimum size threshold for object and polygon to be kept as an 'actual' object, otherwise removed.

    OUTPUTS
    ------------------------------
    list of lists, contains polygons and areas as
    [ [[x-coord_i, x-coord_i+1, ...] for each segmented mask] , [[y-coord_i, y-coord_i+1, ...] for each segmented mask], [area for each segmented mask] ]
    """
    #we want to pad binary_mask one on each side. Then subtract the same pad from each.
    import pdb;pdb.set_trace()
    if binary_mask.dtype=='bool':
        binary_mask = np.pad(binary_mask,((1,1),(1,1)),constant_values=(False,False))
    else:
        binary_mask = np.pad(binary_mask,((1,1),(1,1)),constant_values=(0,0))

    polygons_x = []
    polygons_y = []
    contours = measure.find_contours(binary_mask, 0.5, fully_connected='high') #see documentation for 0.5
    a=[]
    for contour in contours:
        contour = np.flip(contour, axis=1)
        if len(contour) < 3:
            continue
        segmentation_x = contour[:,0].tolist()
        segmentation_y = contour[:,1].tolist()
        #now account for padding
        segmentation_x = [0 if i-1 < 0 else i-1 for i in segmentation_x]
        segmentation_y = [0 if i-1 < 0 else i-1 for i in segmentation_y]
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        #if the threshold area is too low, do not include it
        if PolyArea(segmentation_x,segmentation_y)>=thresh:
            polygons_x.append(segmentation_x)
            polygons_y.append(segmentation_y)
            a.append(PolyArea(segmentation_x,segmentation_y))

    return [polygons_x,polygons_y,a]

def mask_to_polygon_regionprops(label_img):
    """
    convert mask to polygon using skimage.measure.regionprops. 
    WARNING: SLOWER than mask_to_polygon_skimage, which uses skimage.measure.find_contours
    """
    contours = [r.coords for i,r in enumerate(measure.regionprops(label_img))]
    polygons_x = [r[:,0].tolist() for r in contours]
    polygons_y = [r[:,1].tolist() for r in contours]
    return [polygons_x, polygons_y]

def create_annotation_info(image_name, annotations):
    """
    Save annotation data in format required by user. Should be edited if different JSON format or keys are required.
    """

    assert isinstance(annotations,list), "annotations is not a list-type."
    assert isinstance(image_name, str), "image_name is not a string-type"

    annotation_info = {
        "image_name": image_name,
        "features": {'anno_style': ['auto' for _ in range(len(annotations))],
                     'class': [0 for _ in range(len(annotations))],
                     'metadata': ['' for _ in range(len(annotations))]
                    },
        "annotation": annotations
    }

    return annotation_info

def create_annotations(annotation_path, size_thresh=250, reduce_vertices=True):
    """
    Convert tif file annotated image into polygons

    INPUTS
    ---------------------------------
    size_thresh = float, Area size threshold, applied in mask_to_polygon_skimage
    reduce_vertices = boolean, True if apply skimage.measure.approximate_polygon

    OUTPUTS
    ---------------------------------
    all_annotations = list of lists, contains polygons as [ [[x-coord_i, y-coord_i],[x-coord_i+1,y-coord_i+1],...] for each segmented mask ]
    """
    annotated_image = tifffile.imread(annotation_path)
    assert len(annotated_image.shape)==2, "The loaded annotated image has shape {}, but we require shape (H x W)"

    [segmentations_x, segmentations_y, _] = mask_to_polygon_skimage(annotated_image, size_thresh)

    all_annotations = [[[x,y] for (x,y) in zip(ix,iy)] for (ix,iy) in zip(segmentations_x, segmentations_y)]
    if reduce_vertices:
        for i,anno in enumerate(all_annotations):
            verts = np.stack(anno,axis=0)
            #reduce the amount of vertices using approximate.
            verts = approximate_polygon(verts, tolerance = 1)
            #convert back to a list and replace
            all_annotations[i] = verts.tolist()
    return all_annotations

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    sys.stdout.flush()

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    import colorsys
    import random
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def show_first_instance(annotation_report,image_path):
    """
    Displays polygons back onto image.
    """
    from matplotlib import pylab as plt
    import matplotlib.patches as patches
    print("\nShowing first instance")
    img = tifffile.imread(image_path)
    img = img.astype("float32")
    img = img/np.max(img)
    assert len(img.shape)==2, 'image has shape {}, we expect shape = (H x W)'

    #grab segmentations from the annotation report, generate the mask.
    annotation_list = annotation_report['annotation']
    
    colors = random_colors(len(annotation_list))
    print("\nadding annotations...")
    annotation_map = np.zeros(shape=(img.shape[0],img.shape[1],1))
    
    for i,verts in enumerate(annotation_list):
        print("annotation {} of {}".format(i+1,len(annotation_list)))
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.stack(verts,axis=0)
        rr,cc = draw.polygon(verts[:,0], verts[:,1], annotation_map.shape[0:-1])
        #calculate rr,cc intersection over union with
        annotation_map[rr,cc, 0] = 1
        

    img = np.stack((img,)*3,axis=-1)
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    for i,verts in enumerate(annotation_list):
        verts = np.stack(verts,axis=0)
        p = patches.Polygon(verts, facecolor="none", edgecolor=colors[i], alpha=0.6)
        ax.add_patch(p)
        p = patches.Polygon(verts, facecolor=colors[i], edgecolor="none", alpha=0.2)
        ax.add_patch(p)
        
    print("The following display shows the first annotated image. \
    If this does not appear correct, you may need to check what went wrong.")
    plt.show()
    
def find_files_by_pattern(path, pattern, recursive=True):
    """
    PURPOSE: Imports ALL files from a chosen folder based on a given pattern
    INPUTS
    -------------------------------------------------------------
    pattern : list of strings with particular patterns, including filetype!
            ex: ["_patched",".csv"] will pull any csv files under filepath with the string "_patched" in its file name.

    filepath : string, path for where to search for files
            ex: "/users/<username>/folder"

    recursive : boolean, True if you wish for the search for files to be recursive under filepath.
    """
    # generate pattern finding
    fpatterns = ["**{}".format(x) for i,x in enumerate(pattern)]
    if path[-1]!="/":
        path = path + "/"
    all_file_names = set()
    for fpattern in fpatterns:
        file_paths = glob.iglob(path + fpattern, recursive=recursive)
        for file_path in file_paths:
            # skip hidden files
            if file_path[0] == ".":
                continue
            file_name = os.path.basename(file_path)
            all_file_names.add(file_name)

    all_file_names = [file_name for file_name in all_file_names]
    all_file_names.sort()  # sort based on name

    return all_file_names

#############################################################################

###########################################################################
############################################################################

if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Create training images (z-stack .ome.tif files)')
    parser.add_argument('--image_dir', required=True,
                        metavar="/path/to/folder/with/training_files.ome.tif",
                        help='Directory of the dataset')
    parser.add_argument('--annotation_dir', required=False,
                        metavar="/path/to/binary/segmented/folder",
                        help='Path to save files (default=/projected)')
    parser.add_argument('--save_dir', required=False,
                        metavar="/path/to/save/annotations/files.json",
                        help='Path to save files (default=/annotations/)')
    parser.add_argument('--delete_overlap', required=False, default=False,
                        metavar="boolean",
                        help="Boolean to specify to delete annotations with >95% overlap with another annotation")

    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise ImportError("'--image_dir' argument does not point to an exisiting directory.")

    if not args.annotation_dir:
        args.annotation_dir = os.path.join(os.path.dirname(args.image_dir),"projected")

    if not os.path.isdir(args.annotation_dir):
        raise ImportError("'--annotation_dir' argument does not point to an exisiting directory.")

    if not args.save_dir:
        args.save_dir = os.path.join(os.path.dirname(args.image_dir),"annotations")

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    image_dir = args.image_dir
    annotation_dir = args.annotation_dir
    save_dir = args.save_dir

    #############PARSE ANNOTATION PATHWAYS AND FILES#########################
    #find all the .tif files in the annotation directory (annotation_dir)
    annotated_fnames = find_files_by_pattern(annotation_dir, ['.tif'])
    #find all the .tif files in the image directory (image_dir)
    im_fnames = find_files_by_pattern(image_dir, [".tif"])
    #these should be the same length, and have the same filenames at this point
    assert len(annotated_fnames) == len(im_fnames), "found {} annotated .tif files, but found unequal number ({}) of image files".format(len(annotated_fnames), len(im_fnames))
    assert np.all([x==y for (x,y) in zip(annotated_fnames, im_fnames)]), "filenames of annotated files did not match filenames of images."
    
    ##################################################################

    for im_num, tif_file in enumerate(im_fnames):
        progbar(im_num,len(im_fnames),20)
        #create annotations
        annotation_info = create_annotations(annotation_path=os.path.join(annotation_dir,tif_file),
            size_thresh=300)
        assert isinstance(annotation_info,list), 'these_annotations is not a list-type'

        annotation_report = create_annotation_info(tif_file, annotation_info)

        #show_first_instance(annotation_report, os.path.join(image_dir,tif_file)) #uncomment to see the placement of the annotations

        #save individual ground truth files for training?
        with open(os.path.join(save_dir, tif_file.split(".")[0]+'.json'),'w') as output_json_file:
            json.dump(annotation_report, output_json_file)


    progbar(len(im_fnames),len(im_fnames),20)
    print("\n")

    print("Complete :)")
