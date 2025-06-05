from abc import abstractmethod
import torch
from torch.optim.lr_scheduler import StepLR

import utils
from engine import train_one_epoch, evaluate


class BaseModel(torch.nn.Module):
    def __init__(self, save_path='model.pth'):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path
        
    @abstractmethod
    def get_model(self):
        pass

    
    def get_optimizer(self, optimizer_name = 'sgd'):

        params = [p for p in self.model.parameters() if p.requires_grad]

        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(params, lr=0.005,
                                        momentum=0.9, weight_decay=0.0005)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
        return optimizer

    def save_model(self, path=''):
        if path:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), self.save_path)

    def train(self,  dataloader, num_epochs=5):
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, self.optimizer, dataloader, self.device, epoch, print_freq=10)
            # update the learning rate
            self.save_model(f'{self.save_path}_epoch_{epoch}.pth')
            self.lr_scheduler.step()

        self.save_model()
