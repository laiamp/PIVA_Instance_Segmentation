from abc import abstractmethod
import torch
from torch.optim.lr_scheduler import StepLR

from dataset import PennFudanDataset
from data import get_transform
import utils
from engine import train_one_epoch, evaluate

class BaseModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
        self.data_loader, self.data_loader_test = self.get_data_loaders()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    @abstractmethod
    def get_model(self):
        pass

    
    def get_optimizer(self):

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        return optimizer


    def get_data_loaders(self):
        # use our dataset and defined transformations
        dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
        dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    
        # split the dataset in train and test set
        torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=2,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=2,
            collate_fn=utils.collate_fn)
        
        return data_loader, data_loader_test


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def train_model(self, num_epochs=5):
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, self.optimizer, self.data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            self.lr_scheduler.step()

        self.save_model()

    def evaluate_model(self):
        evaluate(self.model, self.data_loader_test, device=self.device)
