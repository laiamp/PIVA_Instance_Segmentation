from abc import abstractmethod
import torch
from torch.optim.lr_scheduler import StepLR

import utils
from engine import train_one_epoch, evaluate


class BaseModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
        
    @abstractmethod
    def get_model(self):
        pass

    
    def get_optimizer(self):

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        return optimizer

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, num_epochs=5):
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, self.optimizer, self.data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            self.lr_scheduler.step()

        self.save_model()

    def evaluate(self, data_loader_test=None):
        evaluate(self.model, data_loader_test, device=self.device)
