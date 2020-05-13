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

    def learn(self, data, labels):
        
        self.optimizer.zero_grad()
        data = T.tensor(data).to(self.device)
        targets = T.tensor(labels).to(self.device)
        
        preds = self.forward(data)

        loss = self.loss(preds, targets)

        loss.backward()
        self.optimizer.step()

    def to_tensor(self, data):
        return T.tensor(data, device = self.device)

    def make_input_one_hot(self, data):
        pass

        


#a = LinearClassifier(.001, 4, (16,))
#b=(T.rand(16, device = 'cuda:0'))
#a.forward(b)
#a.learn(b, T.tensor([0.,1.,1.,0.], device = 'cuda:0'))
