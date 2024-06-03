import os
import numpy as np
import torch

import train
import eval

train_params = {}

train_params['experiment_name'] = 'demo' # This will be the name of the directory where results for this run are saved.

'''
species_set
- Which set of species to train on.
- Valid values: 'all', 'snt_birds'
'''
train_params['species_set'] = 'all'

'''
hard_cap_num_per_class
- Maximum number of examples per class to use for training.
- Valid values: positive integers or -1 (indicating no cap).
'''
train_params['hard_cap_num_per_class'] = 1000

'''
num_aux_species
- Number of random additional species to add.
- Valid values: Nonnegative integers. Should be zero if params['species_set'] == 'all'.
'''
train_params['num_aux_species'] = 0

'''
input_enc
- Type of inputs to use for training.
- Valid values: 'sin_cos', 'env', 'sin_cos_env'
'''
train_params['input_enc'] = 'sin_cos'

'''
loss
- Which loss to use for training.
- Valid values: 'an_full', 'an_slds', 'an_ssdl', 'an_full_me', 'an_slds_me', 'an_ssdl_me'
'''
train_params['loss'] = 'an_full'
 

# Setup DDP
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_ddp(rank, world_size):
    master_addr = os.getenv('MASTER_ADDR', 'gypsum-gpu153')
    master_port = os.getenv('MASTER_PORT', '12345')
    world_size = int(os.getenv('WORLD_SIZE', 3))
    start_rank = int(os.getenv('START_RANK', 0))
    init_method = f'tcp://{master_addr}:{master_port}' 
    print(f"world_size: {world_size}, start_rank + rank: {start_rank + rank}, start_rank: {start_rank}, rank: {rank}, init_method: {init_method}")
    # initialize the process group
    dist.init_process_group("nccl", rank=start_rank + rank, world_size=world_size, init_method=init_method) 
    print(init_method)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup_ddp(rank, world_size) 
    # train:    
    train.launch_training_run(train_params)


def run_demo(demo_fn, world_size):
    world_size = int(os.getenv('WORLD_SIZE', 1))
    nprocs = int(os.getenv('NPROCS', 1))

    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=nprocs,
             join=True)

if __name__ == "__main__":
    world_size = int(os.getenv('WORLD_SIZE', 1))
    nprocs = int(os.getenv('NPROCS', 1))
    rank = int(os.getenv('RANK', 0))  # Get the rank set by torchrun
    run_demo(demo_basic, world_size)
    # evaluate:
    for eval_type in ['snt', 'iucn', 'geo_prior', 'geo_feature']:
        eval_params = {}
        eval_params['exp_base'] = './experiments'
        eval_params['experiment_name'] = train_params['experiment_name']
        eval_params['eval_type'] = eval_type
        if eval_type == 'iucn':
            eval_params['device'] = torch.device('cpu') # for memory reasons
        cur_results = eval.launch_eval_run(eval_params)
        np.save(os.path.join(eval_params['exp_base'], train_params['experiment_name'], f'results_{eval_type}.npy'), cur_results)

'''
Note that train_params and eval_params do not contain all of the parameters of interest. Instead,
there are default parameter sets for training and evaluation (which can be found in setup.py).
In this script we create dictionaries of key-value pairs that are used to override the defaults
as needed.
'''
