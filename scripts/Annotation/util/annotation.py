from .convert import binary_mask_to_polygon_skimage
import numpy as np

def get_shapes_layer_annotations(shape_data):
    if not shape_data:
        return []
    #convert to a list of lists. shape_data will be len == # of annotations, and each sublist (shape_data[0]) will have len == # vertices
    #annotations = [[list(y) for y in x] for x in shape_data]
    annotations = [[[round(y[0],1),round(y[1],1)] for y in x] for x in shape_data]
    return annotations

def get_shapes_layer_features(shape_features):
    if shape_features is None:
        return make_empty_features()

    #convert
    features_dict = shape_features.to_dict()

    #MAKE SURE LOOKS RIGHT. {key:[D[key][i] for i in list(D[key].keys())] for key in list(D.keys()) for x in D[key]}
    features = {key:[features_dict[key][i] for i in list(features_dict[key].keys())] for key in list(features_dict.keys())}

    return features

def filter_shapes_output(shape_annotations, shape_features):
    #the last entries in features can be bad if a polygon was made but never completed, check before saving. 
    num_objects = len(shape_annotations)
    shape_features = {key:[x for i,x in enumerate(shape_features[key]) if i<num_objects] for key in list(shape_features.keys())}
    return shape_features

def get_labels_layer_annotations(brush_data):
    
    if brush_data is not None:
        #Let's assume there are more than 1 class ()
        ####
        #https://napari.org/stable/gallery/add_shapes_with_features.html
        ####
        brush_data = brush_data[0,:,:] #> 0. #convert to binary, note that brush data loads as a (N x H x W) image. Since we aren't dealing with 3D data...


        allclasses = list(np.unique(brush_data)[1:])

        features = {}
        features['class'] = allclasses
        features['anno_style'] = ['manual' for _ in allclasses]
        features['metadata'] = ['' for _ in allclasses]

        annotations = []

        for anno_i in allclasses:
            [vertices,_] = binary_mask_to_polygon_skimage(brush_data==anno_i,thresh=10)
            annotations += vertices

        return annotations, features
    else:
        features = {'class': [], 'anno_style': [], 'metadata': []}
        return [], features

def make_empty_features():
    return {"anno_style":[], "class":[], "metadata":[]}

def make_annotation_data(image_name, shapes_layer=None, labels_layer=None):
    shape_data = shapes_layer.data if shapes_layer else None
    shape_features = shapes_layer.features if shapes_layer else None
    brush_data = labels_layer.data if labels_layer else None

    shape_annotations = get_shapes_layer_annotations(shape_data)
    shape_features = get_shapes_layer_features(shape_features)
    shape_features = filter_shapes_output(shape_annotations, shape_features)
    
    label_annotations, label_features = get_labels_layer_annotations(brush_data)

    all_annotations = shape_annotations + label_annotations

    all_classes = shape_features['class'] + label_features['class']
    all_styles = shape_features['anno_style'] + label_features['anno_style']
    all_metadata = shape_features['metadata'] + label_features['metadata']

    all_features = { 'class': all_classes, 'anno_style': all_styles, 'metadata': all_metadata}
    annot_data = { 'image_name': image_name, 'features': all_features, 'annotation': all_annotations }
    return annot_data
