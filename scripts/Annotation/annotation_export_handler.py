from np_encoder import NpEncoder
from util.file import load_json_data
from util.annotation import make_annotation_data
import json
import os

class AnnotationExportHandler:
    def __init__(self, annot_file_path, image_name, shapes_layer=None, labels_layer=None):
        self._annot_file_path = annot_file_path
        self._image_name = image_name
        self._shapes_layer = shapes_layer
        self._labels_layer = labels_layer
        self._current_content = self._calculate_current_content(image_name, shapes_layer, labels_layer)

    def _calculate_current_content(self, image_name, shapes_layer, labels_layer):
        return make_annotation_data(image_name, shapes_layer, labels_layer)

    def refresh_current_content(self):
        self.current_content = self._calculate_current_content(self._image_name, self._shapes_layer, self._labels_layer)

    def _get_current_content(self):
        if self._current_content is None:
            self.refresh_current_content()

        return self._current_content

    def is_updated(self):
        current_content = self._get_current_content()
        if not os.path.exists(self._annot_file_path):
            file_content = None
        else:
            file_content = load_json_data(self._annot_file_path)

        return current_content != file_content

    def export_to_file(self):
        current_content = self._get_current_content()
        with open(self._annot_file_path,'w') as output_json_file:
            json.dump(current_content, output_json_file, cls=NpEncoder)
