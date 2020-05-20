# 3 layer cnn, 2 fc layers
# 32, 8x8, stride 4
# 64, 4x4 stride 2
# 64, 3x3, stride 1
# FC 512
# rms prop

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQCNN(nn.Module):
    def __init__(self, input_dims, n_actions, lr=0.0001, name='NN', chkpt_dir='./nn.nn'):
        super().__init__()
        self.name = name
        self.chkpt_dir = chkpt_dir
        # input_dims [n_channels, height, width]
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, (4,4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, (3,3), stride=1)
        self.fc_size = self.calc_fc_input_dim(input_dims)
        self.fc1 = nn.Linear(self.calc_fc_input_dim(input_dims), 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.loss = nn.MSELoss()

        self.optimizer = optim.RMSprop(self.parameters(), lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)       

    def calc_fc_input_dim(self, input_dims):
        channels, height, width = input_dims
        
        cnn_out_dim = self.conv3(
            self.conv2(
                self.conv1(
                    T.randn(
                        1, channels, height, width
                        )
                    )
                )
        ).shape

        num_out_filters = cnn_out_dim[1]
        out_width = cnn_out_dim[2]
        out_height = cnn_out_dim[3]

        return num_out_filters * out_height * out_width
    
    def forward(self, data):
        conv1 = T.relu(self.conv1(data))
        conv2 = T.relu(self.conv2(conv1))
        conv3 = T.relu(self.conv3(conv2))

        # flatten cnn output to single batch dim plus single dimension
        flat = T.flatten(conv3, start_dim=1)
        fc1 = T.relu(self.fc1(flat))
        outputs = self.fc2(fc1)
        return outputs

    def fit(self, data, targets):
        '''fit model on data and targets'''
        self.optimizer.zero_grad()

        data = data.to(self.device, dtype=T.float)
        targets = targets.to(self.device, dtype=T.float)
        preds = self.forward(data)
        loss = self.loss(preds, targets)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def save_checkpoint(self):
        print('...saving ' + self.name)
        T.save(self.state_dict(), self.chkpt_dir)
    
    def load_checkpoint(self):
        print('...loading checkpoint for ' + self.name)
        self.load_state_dict(T.load(self.chkpt_dir))

# d = DeepQCNN((1,84,84))
# data = T.randn((10,1,84,84)).to(d.device)
# targets = T.tensor([[0.]]*10).to(d.device)
# d.fit(data, targets)