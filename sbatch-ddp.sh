#!/bin/bash
#SBATCH --job-name=unity_workshop  
#SBATCH --nodes=2               # 2 nodes
#SBATCH --ntasks=6              # Total number of tasks (2 tasks per node)
#SBATCH --ntasks-per-node=3     # Number of tasks per node
#SBATCH --partition=gpu-preempt         # partition
#SBATCH --gres=gpu:3            # 2 GPUs per node
#SBATCH --cpus-per-task=8      # CPUs
#SBATCH --mem=64g      # CPUs
#SBATCH --time=04:00:00         
#SBATCH --constraint=2080ti     # Constraint for nodes with 2080 Ti GPUs
#SBATCH --output=output_%j.txt 


# Load necessary modules (if any)  
# Get the list of nodes 
# Environment variables for PyTorch DDP
ml load miniconda/22.11.1-1
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/sinr_icml
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345  # Use any free port number
export WORLD_SIZE=6       # Total number of processes

# Get the list of nodes
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODE_LIST=($NODES)
# Run the PyTorch script on each node and each GPU
GPU_LIST=(3 3)
START_RANK=0
for i in ${!NODE_LIST[@]}
do
  NODE=${NODE_LIST[$i]}
  NUM_GPUS=${GPU_LIST[$i]}
  echo "NODE: $NODE"
  echo "NUM_GPUS: $NUM_GPUS"
  echo "START_RANK $START_RANK"
  srun --nodes=1 --ntasks=1 --gpus=$NUM_GPUS --nodelist=$NODE \
    bash -c "export START_RANK=$START_RANK; export WORLD_SIZE=$WORLD_SIZE; export MASTER_ADDR=$MASTER_ADDR; export MASTER_PORT=$MASTER_PORT; python3 train_and_evaluate_models.py" &
  START_RANK=$(($START_RANK + $NUM_GPUS))
done

wait