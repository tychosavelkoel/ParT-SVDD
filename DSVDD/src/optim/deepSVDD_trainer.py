from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import tqdm

class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self,  net: BaseNet, train_loader = None, data_config = None, steps_per_epoch = 100):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net,  data_config = data_config)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            with tqdm.tqdm(train_loader) as tq:
                for X, y, _ in tq:
                    inputs = [X[k].to(self.device) for k in data_config.input_names]

                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = net(*inputs).flatten(start_dim = 1)

                    dist = torch.sum((outputs - self.c) ** 2, dim=1)


                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    else:
                        loss = torch.mean(dist)

                    loss.backward()
                    optimizer.step()


                    # Update hypersphere radius R on mini-batch distances
                    if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                        self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                    loss_epoch += loss.item()
                    n_batches += 1

                    if steps_per_epoch is not None and n_batches >= steps_per_epoch:
                        break

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, net: BaseNet, test_loader = None, data_config = None, steps_per_epoch = 10):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        score = []
        net.eval()
        n_batches =0
        with torch.no_grad():
            with tqdm.tqdm(test_loader) as tq:
                for X, y, _ in tq:
                    inputs = [X[k].to(self.device) for k in data_config.input_names]

                    outputs = net(*inputs).flatten(start_dim = 1)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)

                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                    else:
                        scores = dist

                    # Save triples of (idx, label, score) in a list --> save score in a list
                    score.append(scores.cpu().data.numpy().tolist())

                    n_batches+=1
                    if steps_per_epoch is not None and n_batches >= steps_per_epoch:
                        break

        score = np.concatenate(score)
        
        
        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = list(zip(score))

        # Compute AUC
        #_, labels, scores = zip(*idx_label_score)
        #labels = np.array(labels)
        #scores = np.array(scores)

        #self.test_auc = roc_auc_score(labels, scores)
        #logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1,  data_config = None,):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(16*16, device=self.device) #torch.zeros(net.rep_dim, device=self.device)
        
        num_batches = 0
        with torch.no_grad():
            with tqdm.tqdm(train_loader) as tq:
                for X, y, _ in tq:
                    inputs = [X[k].to(self.device) for k in data_config.input_names]
                    outputs = net(*inputs).flatten(start_dim = 1)
                    n_samples += outputs.shape[0]
                    c += torch.sum(outputs, dim=0)
                    num_batches += 1

                    if num_batches >= 30:
                        break
        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
