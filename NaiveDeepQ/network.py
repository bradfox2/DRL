import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)
    
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, data):

        layer1 = F.relu(self.fc1(data))
        layer2 = self.fc2(layer1)
        return layer2

    def learn(self, preds, targets):
        
        self.optimizer.zero_grad()
        #preds = T.tensor(preds).to(self.device)
        #targets = T.tensor(targets).to(self.device)
        
        loss = self.loss(preds, targets)

        loss.backward()
        self.optimizer.step()

    def to_tensor(self, data):
        return T.tensor(data, device = self.device)

    def make_input_one_hot(self, data):
        pass
