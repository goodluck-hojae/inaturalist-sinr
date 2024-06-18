import os
import numpy as np
import torch
import setup
import losses
import models
import datasets
import utils
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer():

    def __init__(self, model, train_loader, params):

        self.params = params
        self.rank = self.params['rank']
        # define loaders:
        self.train_loader = train_loader

        # define model:
        self.model = model.cuda(0)

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
            print(f'batch shape {batch[0].shape}, {len(batch)}')
            # compute loss:
            # batch = batch.to(self.rank)
            
            print('3', torch.cuda.memory_allocated() / (1024 ** 2))
            print('3', torch.cuda.memory_reserved() / (1024 ** 2))
            
            batch_loss = self.compute_loss(batch, self.model, self.params, self.encode_location)
            print('4', torch.cuda.memory_allocated() / (1024 ** 2))
            print('4', torch.cuda.memory_reserved() / (1024 ** 2))
            
            # backwards pass:
            
            batch_loss.backward()
            print('5', torch.cuda.memory_allocated() / (1024 ** 2))
            print('5', torch.cuda.memory_reserved() / (1024 ** 2))
            
            # update parameters:
            self.optimizer.step()
            print('6', torch.cuda.memory_allocated() / (1024 ** 2))
            print('6', torch.cuda.memory_reserved() / (1024 ** 2))
            
            # track and report:
            running_loss += float(batch_loss.item() / (1024 ** 2))
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
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4)

    # model:
    model = models.get_model(params)

    # train:
    
    print('1', torch.cuda.memory_allocated() / (1024 ** 2))
    trainer = Trainer(model, train_loader, params)
    print('2', torch.cuda.memory_allocated() / (1024 ** 2))
    for epoch in range(0, params['num_epochs']):
        print(f'epoch {epoch+1}')
        trainer.train_one_epoch()
    trainer.save_model()


