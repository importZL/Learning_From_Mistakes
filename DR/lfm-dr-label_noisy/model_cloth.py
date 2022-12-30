import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResNet50(nn.Module):
    def __init__(self, num_classes, len_train, batch_size):
        super().__init__()
        import torchvision
        import os
        
        os.environ['TORCH_HOME'] = 'cache' # hacky workaround to set model dir
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity() # remote last fc
        self.fc = nn.Linear(2048, num_classes)

        self.init_weights()
        self._initialize_weight_b(len_train, batch_size)

    def init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.zero_()

    def _initialize_weight_b(self, len_train, batch_size):
        self.weight_b = Variable(1.1 * torch.ones(batch_size, len_train).cuda(), requires_grad=True)
        self._b_parameters = [
            self.weight_b,
        ]

    def b_parameters(self):
        return self._b_parameters

    def forward(self, x, return_h=False): # (bs, C, H, W)
        pooled_output = self.resnet50(x)
        logit = self.fc(pooled_output)
        if return_h:
            return logit, pooled_output
        else:
            return logit