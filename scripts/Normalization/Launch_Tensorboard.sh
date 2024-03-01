#!/bin/bash
#SBATCH --partition exacloud
#SBATCH --output=tb_%j.out         ### File in which to store job output
#SBATCH --error=tb_%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task 2
#SBATCH --time 1:00:00
#SBATCH --job-name tb

# environment with Tensorboard
source activate pytorch_footprinting

node=$(hostname -s)
port=$(/usr/local/bin/get-open-port.py)

echo "Node: ${node}"
echo "Port: ${port}"
echo
echo "Example connection string:"
echo "  $ ssh ${USER}@exahead1.ohsu.edu -L ${port}:${node}:${port}"
echo "  $ ssh ${USER}@exahead2.ohsu.edu -L ${port}:${node}:${port}"
echo
echo "Once the ssh connection is established, copy the URL printed below!!! (not the url from Tensorboard)"
echo "http://127.0.0.1:${port}"
echo
echo "Navigate to that URL in your local browser."

#tensorboard --logdir ${logdir} --port ${port} --host ${node}
tensorboard --logdir /home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Normalization/models/logs --port ${port} --host ${node}