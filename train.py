import os
import numpy as np
import torch
import setup
import losses
import models
import datasets
import utils
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class Trainer():

    def __init__(self, model, train_loader, params):

        self.params = params
        self.rank = dist.get_rank() % torch.cuda.device_count()
        print('rank', self.rank)
        # define loaders:
        self.train_loader = train_loader

        # define model: 
        torch.cuda.set_device(self.rank)
        self.model = model.cuda(self.rank)
        
        print(f"Rank {self.rank} reached the barrier.")
        dist.barrier() 
        print(f"Rank {self.rank} passed the barrier.")
        self.model = DDP(self.model, device_ids=[self.rank]) # local rank
        
        

        # define important objects:
        self.compute_loss = losses.get_loss_function(params)
        self.encode_location = self.train_loader.dataset.enc.encode

        # define optimization objects:
        self.optimizer = torch.optim.Adam(self.model.parameters(), params['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params['lr_decay'])

    def train_one_epoch(self):

        self.model.train()
        # initialise run stats
        running_loss = 0.0
        samples_processed = 0
        steps_trained = 0
        for _, batch in enumerate(self.train_loader):
            # reset gradients:
            self.optimizer.zero_grad()
            # compute loss:
            batch = [item.to(torch.device('cuda', self.rank)) for item in batch]
            batch_loss = self.compute_loss(batch, self.model, self.params, self.encode_location)
            # backwards pass:
            batch_loss.backward()
            # update parameters:
            self.optimizer.step()
            # track and report:
            running_loss += float(batch_loss.item())
            steps_trained += 1
            samples_processed += batch[0].shape[0]
            if steps_trained % self.params['log_frequency'] == 0:
                print(f'[{samples_processed}/{len(self.train_loader.dataset)}] loss: {np.around(running_loss / self.params["log_frequency"], 4)}')
                running_loss = 0.0
        # update learning rate according to schedule:
        self.lr_scheduler.step()

    def save_model(self):
        save_path = os.path.join(self.params['save_path'], 'model.pt')
        op_state = {'state_dict': self.model.state_dict(), 'params' : self.params}
        torch.save(op_state, save_path)

def launch_training_run(ovr):
    # setup:
    params = setup.get_default_params_train(ovr)
    params['save_path'] = os.path.join(params['save_base'], params['experiment_name'])
    if params['timestamp']:
        params['save_path'] = params['save_path'] + '_' + utils.get_time_stamp()
    os.makedirs(params['save_path'], exist_ok=True)



    # data:
    train_dataset = datasets.get_train_data(params)
    params['input_dim'] = train_dataset.input_dim
    params['num_classes'] = train_dataset.num_classes
    params['class_to_taxa'] = train_dataset.class_to_taxa
    

    print(f"Current batch size: { params['batch_size'] }")

    sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        sampler=sampler, 
        num_workers=4)
    # model:
    model = models.get_model(params)

    # train:
    trainer = Trainer(model, train_loader, params)
    
    for epoch in range(0, params['num_epochs']):
        sampler.set_epoch(epoch)
        print(f'epoch {epoch+1}')
        trainer.train_one_epoch()
    trainer.save_model()


