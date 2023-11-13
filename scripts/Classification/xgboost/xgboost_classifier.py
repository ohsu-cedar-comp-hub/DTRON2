"""
Author: Christopher Z. Eddy
Date: 03/15/23
Purpose:
Training an XGBoost model on GPU (1M+ rows, 160+ columns [features]).
All credit to authors, and most code from https://github.com/dmlc/xgboost/tree/master/demo/gpu_acceleration

Update: 03/15/23 Currently missing any parameter search. Cuml doesn't have a gridsearchcv function like sklearn.
However, dask does. We'll want to load our clf model (model = xgb.XGBClassifier()) and pass it
https://blog.dask.org/2019/03/27/dask-cuml
"""

"""
NEED TO CREATE THE ENVIRONMENT FOR THIS! Needs cuml cudf, xgboost, numpy, pickle
"""
import numpy as np 
#from sklearn.model_selection import train_test_split #works for Pandas, numpy, but not cuDF
from cuml.model_selection import train_test_split #works for cuDF only.

import xgboost as xgb
import cudf 
from datetime import datetime
import pickle
# RAPIDS roc_auc_score is 16x faster than sklearn
from cuml.metrics import roc_auc_score



class XGBoostModel(object):
	
	def __init__(self, train_val_split = 0.2, num_classes = 2, class_weights = None):
		self.train_val_split = train_val_split
		# Specify sufficient boosting iterations to reach a minimum
		self.num_round = 3000
		# Leave most parameters as default
		self.param = {'objective': 'multi:softmax', # Specify multiclass classification
					  'num_class': num_classes, # Number of possible output classes
					  'tree_method': 'gpu_hist', # Use GPU accelerated algorithm
					  'class_weight': class_weights,
					}
		
	
	def load_train_val_data(self, path, gt_col = "class", feature_cols = None, bad_cols = None):
		""" feature cols = list of strings,
			bad_cols = list of integers"""

		np.random.seed(0) #initialize random seed.
		#use cudf to load data.
		if os.path.splitext(path)[-1]==".parquet":
			data = cudf.from_parquet(path) 
		elif os.path.splitext(path)[-1]==".csv":
			data = cudf.from_csv(path) 
		else:
			assert "", "Expecting data in the form of a .csv or .parquet format"

		assert gt_col in data.columns.to_list(), "gt_col argument='{}' was not found but must be in data columns."
		#split dataset
		#sklearns train_test_split accepts both numpy arrays and dataframes. Will return the same data type (i.e. dataframe -> dataframe). 
		#What if it is a cudf array?
		#Fails, if stratify is given with cudf array.

		#WORKS FOR PANDAS BUT NOT CUDF
		#X_train, X_val = train_test_split(data, test_size = self.train_val_split, stratify = data[gt_col])

		#WORKS FOR CUDF AND PANDAS
		y = data[gt_col]
		if feature_cols is None:
			#else, use ALL columns except for target class.
			feature_cols = data.columns.to_list()
			import pdb;pdb.set_trace()
			if bad_cols is not None:
				feature_cols = [x for i,x in enumerate(feature_cols) if (x!=gt_col) and i not in bad_cols]
			else:
				feature_cols = [x for x in feature_cols if (x!=gt_col)]

		self.X_train, self.y_train, self.X_val, self.y_val = train_test_split(data[feature_cols], y, test_size = self.train_val_split, stratify = y)


	def load_test_data(self, path, gt_col = "class", feature_cols = None):
		#WITH LABEL
		#use cudf to load data.
		if os.path.splitext(path)[-1]==".parquet":
			data = cudf.from_parquet(path) 
		elif os.path.splitext(path)[-1]==".csv":
			data = cudf.from_csv(path)
		else:
			assert "", "Expecting data in the form of a .csv or .parquet format"

		assert gt_col in data.columns.to_list(), "gt_col argument='{}' was not found but must be in data columns."
		#WORKS FOR CUDF AND PANDAS
		self.y_test = data[gt_col]
		if feature_cols is None:
			#else, use ALL columns except for target class.
			feature_cols = data.columns.to_list()
			feature_cols = [x for x in feature_cols if x!=gt_col]

		self.X_test = data[feature_cols]


	def load_inference_data(self, path, feature_cols=None):
		#without label
		#use cudf to load data.
		if os.path.splitext(path)[-1]==".parquet":
			data = cudf.from_parquet(path) 
		elif os.path.splitext(path)[-1]==".csv":
			data = cudf.from_csv(path) 
		else:
			assert "", "Expecting data in the form of a .csv or .parquet format"

		#WORKS FOR CUDF AND PANDAS
		if feature_cols is None:
			#else, use ALL columns except for target class.
			feature_cols = data.columns.to_list()

		self.X_test = data[feature_cols]

	###################################################################################

	def _save_model_to_pickle(self, model, file_name = None, file_dir = None):
		#assert hasattr(self, 'model'), "self.model does not yet exist. You should have successfully ran 'run_train' prior to this."

		if file_dir is None:
			file_dir = os.getcwd()
		if file_name is None:
			now = datetime.now()
			file_name = "xgb_" + str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + str(now.second) + ".pkl"

		# save
		pickle.dump(model, open(file_name, "wb"))
		#it would also be good to likely save the parameters of the model; all from self._config

	def _load_model_from_pickle(self, file_name):
		# load
		model = pickle.load(open(file_name, "rb"))
		return model

	def _save_model_to_json(self, model, file_name = None, file_dir = None):
		assert hasattr(self, 'model'), "self.model does not yet exist. You should have successfully ran 'run_train' prior to this."

		if file_dir is None:
			file_dir = os.getcwd()
		if file_name is None:
			now = datetime.now()
			file_name = "xgb_" + str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + str(now.second)

		# save
		model.save_model(os.path.join(file_dir, file_name + ".json"))
		# save internal paramters configuration:
		model.save_config(os.path.join(file_dir, file_name + "config.json"))
	
	def _load_model_from_json(self, file_name):
		#I guess we need to initialize a model?
		model = xgb.XGBClassifier() #this may need to change.
		#load the model
		model.load_model(file_name)
		#load config as well. 
		from pathlib import Path
		model.load_config(Path(file_name).stem + "config.json")
		return model

	###################################################################################

	def train_xgb_model(self):
		assert hasattr(self, 'X_train'), "You must first load the data, using '_load_train_val_data'"
		# Convert input data from numpy to XGBoost format
		dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
		#dval = xgb.DMatrix(self.X_val, label=self.y_val)
		# Train model
		#self.model = xgb.train(self.param, dtrain, self.num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
		model = xgb.train(self.param, dtrain, self.num_round)

		# Make prediction
		predicts = model.predict(xgb.DMatrix(self.X_val))
		roc = roc_auc_score(self.y_val.astype('int32'), predicts)
		return model, roc
	
	def xgb_predict(self, model):
		assert hasattr(self, 'X_test'), "You must first load the data, using '_load_test_data'"
		# Make prediction
		predicts = model.predict(xgb.DMatrix(self.X_test))
		return predicts

	def get_SHAP_values(self, model):
		#https://medium.com/rapids-ai/gpu-accelerated-shap-values-with-xgboost-1-3-and-rapids-587fad6822
		# Compute shap values using GPU with xgboost
		# Convert input data from numpy to XGBoost format
		dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
		# model.set_param({"predictor":"cpu_predictor"})
		model.set_param({"predictor": "gpu_predictor"})
		"""
		According to xgboost website, xgboost makes use of the GPUTreeShap as a backend for computing shap values when the GPU predictor is selected. 
		This reduces computation from days to minutes.
		"""
		# Compute shap values using GPU
		shap_values = model.predict(dtrain, pred_contribs=True)
		# Compute shap interaction values using GPU
		shap_interaction_values = model.predict(dtrain, pred_interactions=True) 

		return shap_values, shap_interaction_values
	
	def plot_SHAP_values(self, feature_names, shap_values, shap_interaction_values):
		plot_feature_importance(feature_names, shape_values)
		plt.imshow()
		plot_top_k_interactions(feature_names, shap_interactions, 10)
		plt.imshow()

def plot_feature_importance(feature_names, shap_values):
	# Get the mean absolute contribution for each feature
	aggregate = np.mean(np.abs(shap_values[:, 0:-1]), axis=0)
	# sort by magnitude
	z = [(x, y) for y, x in sorted(zip(aggregate, feature_names), reverse=True)]
	z = list(zip(*z))
	plt.bar(z[0], z[1])
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.show()

def plot_top_k_interactions(feature_names, shap_interactions, k):
	# Get the mean absolute contribution for each feature interaction
	aggregate_interactions = np.mean(np.abs(shap_interactions[:, :-1, :-1]), axis=0)
	interactions = []
	for i in range(aggregate_interactions.shape[0]):
		for j in range(aggregate_interactions.shape[1]):
			if j < i:
				interactions.append(
				(feature_names[i] + "-" + feature_names[j], aggregate_interactions[i][j] * 2))
	# sort by magnitude
	interactions.sort(key=lambda x: x[1], reverse=True)
	interaction_features, interaction_values = map(tuple, zip(*interactions))
	plt.bar(interaction_features[:k], interaction_values[:k])
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.show()

#Let's load this into a Jupyter Notebook.

if __name__ == "__main__":
	class_targets = np.array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
								13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
								26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
								39., 41., 42., 43., 44., 45., 47.], dtype=int)
	class_counts = np.array([50209, 48977, 47267, 46304, 33797, 31330, 23252, 22652, 22063,
								21767, 20026, 19837, 18145, 16927, 16856, 16598, 16357, 10387,
								14178, 13718, 13580, 13362, 12397, 11896, 11859, 11254, 11175,
								10439, 10266,  9461,  8894,  8520,  8104,  1887,  7365,  7109,
								6856,  6151,  5112,  2066,  3186,  2865,  2852,  1627,   377,
								154], dtype=int)
	############ CLASS WEIGHTS #################
	# #we want to overemphasize the smaller class groups:
	class_weights = np.sum(class_counts) / class_counts
	################################################
	XG = XGBoostModel(num_classes = 8,
						class_weights = class_weights)
	XG.load_train_val_data(path = '/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/data/classification/masterdf_allcells.csv', 
							gt_col = "cluster", 
							feature_cols = None, 
							bad_cols = list(range(12)) +  list(range(172,183)))