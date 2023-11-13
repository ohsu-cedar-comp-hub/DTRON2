"""
Executor python script
"""
import marker_classification as Main
import argparse
import torch
import psutil

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parse command line arguments
parser = argparse.ArgumentParser(
	description='Marker Classification for cyclic IF .csv data',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--mode", required=True,
					metavar="<command>",
					help="'training' or 'inference'")
parser.add_argument('--dataset', required=True,
					metavar="/path/to/dataset",
					help='Root directory of the dataset. \nTraining MUST have form:\n\
dataset_directory\n\
----/train\n\
--------/*.csv\n\
----/val\n\
--------/*.csv\n\
Inference MUST have form:\n\
dataset_directory\n\
----/*.csv\n\
\n'
)

parser.add_argument('--logs', required=False,
					metavar="/path/to/logs",
					default=None,
					help = 'Root directory to where to store logs from training!')

parser.add_argument('--weights', required=False,
					metavar="/path/to/checkpoint.tar",
					help="Path to weights .tar file if you wish to load previously trained weights.")

args = parser.parse_args()

if args.weights:
	print("Weights: ", args.weights)
if args.dataset:
	print("Dataset: ", args.dataset)

# Validate arguments
assert args.mode in ["training", "inference"], "mode argument must be one of 'training' or 'inference'"

if args.mode=="inference":
	assert args.weights, "weights checkpoint must be given to load weights for inference."
	pass

if args.mode=="training":
	CP = Main.Marker_Net(mode="training", dataset_path=args.dataset, logs=args.logs)
	#print("Before dataset ::: Used Mem Percent = {}".format(psutil.virtual_memory().percent))
	CP.load_dataset()
	#print("After dataset ::: Used Mem Percent = {}".format(psutil.virtual_memory().percent))
	CP.create_model()
	#print("After model ::: Used Mem Percent = {}".format(psutil.virtual_memory().percent))
	CP.run_train(device=device)