import os
import glob
import pdb

IMAGES_SUBPATH = 'images'
ANNOTATIONS_SUBPATH = 'annotations'

class InputHandler:
    def __init__(self, path, pattern, recursive=False):
        self.path = self._normalize_path(path)
        self.images_path = self._get_images_path(path)
        self.annotations_path = self._get_annotations_path(path)
        self.pattern = pattern
        self.recursive = recursive
        self.refresh_file_names()


    def _normalize_path(self, path):
        if path[-1] != "/":
            path = path + "/"

        return path

    def _get_images_path(self, path):
        return self._normalize_path(os.path.join(path, IMAGES_SUBPATH))

    def _get_annotations_path(self, path):
        return self._normalize_path(os.path.join(path, ANNOTATIONS_SUBPATH))

    def _calculate_annotation_file_name(self, image_file_name):
        annot_file_name = image_file_name.split(".")[0]+".json"
        return annot_file_name

    def _calculate_image_file_names(self, path, pattern, recursive):
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
        fpattern = ["**{}".format(x) for i,x in enumerate(pattern)]
        fpattern = "".join(fpattern)
        file_paths = glob.iglob(self.images_path + fpattern, recursive=recursive)
        file_paths = [file_path for file_path in file_paths if file_path[0] != "."] # delete hidden files
        # file_names = [filepath for filepath in file_paths]
        file_names = [os.path.basename(filepath) for filepath in file_paths]
        file_names.sort()  # sort based on name

        return file_names

    def refresh_file_names(self):
        self.image_file_names = self._calculate_image_file_names(self.path, self.pattern, self.recursive)

    def get_image_file_names(self):
        return self.image_file_names

    def get_image_file_name_at(self, index):
        return self.image_file_names[index]

    def get_image_file_path_at(self, index):
        file_name = self.get_image_file_name_at(index)
        return self.calculate_image_file_path(file_name)

    def get_annotation_file_name_at(self, index):
        img_file_name = self.get_image_file_name_at(index)
        annot_file_name = self._calculate_annotation_file_name(img_file_name)
        return annot_file_name

    def get_annotation_file_path_at(self, index):
        file_name = self.get_annotation_file_name_at(index)
        return self.calculate_annotation_file_path(file_name)

    def calculate_annotation_file_path(self, file_name):
        return os.path.join(self.annotations_path, file_name)

    def calculate_image_file_path(self, file_name):
        return os.path.join(self.images_path, file_name)
