"""
Author: Christopher Z. Eddy
Date: 04/18/23
Purpose: 
You can annotate naively on Napari and save shapes layer as a .csv file. This csv has 5 columns (if 2D image)
Our JSON inputs have several keys:
'image_name': string
'annotation': list of lists. 
    ex for two annotations each with three vertices:
    [ [[x1, y1], [x2,y2], [x3,y3]] , [[x1,y1], [x2,y2], [x3,y3]] ]
'features': dictionary with keys
    'anno_style': list of either ['manual'] or ['auto']. 
        Ex for two annotations: ['manual', 'auto']
    'class': list of integers
        Ex for two annotations: [1, 4]
    'metadata': list of strings
        Ex for two annotations: ['test 1', 'another test']

Since the annotations are provided by SAM, we'll designate the annotation styles as 'auto', and default the classes to 0.
"""
import json 
import pandas as pd

"""
MWE
convert_data("<path_to_csv>", "test")
"""

def convert_data(csv_filepath, image_name, output_path = None):
    """
    INPUTS
    --------------------------
    csv_filepath = str, path to csv file
    image_name = str, name of image (without extension) for the corresponding annotations
    output_path = str, path for where to save the output .json file.

    OUTPUTS
    --------------------------
    dumps JSON file to current working directory or output_path, if provided.
    """
    #read csv file into dataframe. Note the data is already sorted first by index then by vertex-index. So read 
    csv_data = pd.read_csv(csv_filepath, usecols = [0,3,4]) #if 3D, you will need to adjust this.
    #convert axes to list of vertices. 
    annotations = []
    for anno_i in csv_data['index'].unique():
        dta = csv_data.loc[csv_data['index']==anno_i]
        poly_i = [[x,y] for (x,y) in zip(list(dta['axis-0']), list(dta['axis-1']))]
        annotations.append(poly_i)

    #prepare JSON output.
    output = dict(image_name = image_name, 
                  annotation = annotations,
                  features = {'anno_style': ['auto' for _ in range(len(annotations))],
                              'class': [0 for _ in range(len(annotations))],
                              'metadata': ['' for _ in range(len(annotations))]
                              }
                  )
    #write output to file
    import json
    if output_path is not None:
        output_name = os.path.join(output_path, image_name+'.json')
    else:
        output_name = image_name+'.json'

    with open(output_name, 'w') as fp:
        json.dump(output, fp)