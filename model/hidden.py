import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import Deephash
from options import HiDDenConfiguration

class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device):

        super(Hidden, self).__init__()
        self.deephash = Deephash(configuration).to(device)
        self.optimizer = torch.optim.Adam(self.deephash.parameters(),lr=0.0001)
        self.config = configuration
        self.device = device
        self.mse_loss = nn.MSELoss().to(device)


    def train_on_batch(self, images):

        batch_size = images.shape[0]

        with torch.enable_grad():


            hash_code = self.deephash(images)

            # loss_sim = 0
            # loss_dif = 0

            for num_in_fea in range(1, 68):
                self.optimizer.zero_grad()

                loss_sim = self.mse_loss(hash_code[0], hash_code[num_in_fea])
                loss_dif = self.mse_loss(hash_code[0], hash_code[num_in_fea + 67])
                loss_sim = F.sigmoid(loss_sim)
                loss_dif = F.sigmoid(loss_dif)

                loss = loss_sim - loss_dif

                loss.backward(retain_graph=True)

                self.optimizer.step()


        losses = {
            'similar_loss    ': loss_sim.item(),
            'different_loss  ': loss_dif.item(),
            'loss            ': loss.item()
        }
        return losses, hash_code

    def validate_on_batch(self, images):

        batch_size = images.shape[0]

        loss = []

        with torch.no_grad():

            _,_,hash_code = self.deephash(images)

            for num_in_fea in range(1, batch_size):
                loss.append(self.mse_loss(hash_code[0], hash_code[num_in_fea]))

        return loss,hash_code

    def test_single(self, images):
        with torch.no_grad():
            before_a, after_a,hash_code = self.deephash(images)

        return before_a, after_a, hash_code


    def to_stirng(self):
        return '{}\n'.format(str(self.deephash))
