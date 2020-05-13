import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.CrossEntropyLoss()
        self.loss2 = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, data):
        layer1 = F.relu(self.fc1(data))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = self.fc3(layer2) #CEL will handle activation

        return layer3

    def learn(self, data, labels):
        self.optimizer.zero_grad()
        data = T.tensor(data).to(self.device)
        targets = T.tensor(labels).to(self.device)
        
        preds = self.forward(data)

        loss = self.loss(preds, targets)

        loss.backward()
        self.optimizer.step()
