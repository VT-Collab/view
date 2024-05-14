import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
# if torch.cuda.is_available():
#     device = "cuda"
# else:
device = "cpu"

class MotionData(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        inputs = torch.FloatTensor(item[0]).to(device)
        gt = torch.FloatTensor(item[1]).to(device)
        return (inputs, gt)

'''     Policy for residual learning    '''
class ResidualNet(nn.Module):
    def __init__(self, input_dim=3):
        super(ResidualNet, self).__init__()

        self.loss_func = nn.MSELoss()
        self.input_dim = input_dim

        # encoder layers
        self.l1 = nn.Linear(self.input_dim, 16)
        self.l2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, 3)

    def get_residual(self, inputs):
        x = torch.relu(self.l1(inputs))
        x = torch.relu(self.l2(x))
        outputs = self.out(x)
        return outputs

    def forward(self, x):
        inputs = torch.FloatTensor(x[0])
        targets = torch.FloatTensor(x[1])

        outputs = self.get_residual(inputs)
        loss = self.loss(outputs, targets)
        return loss

    def loss(self, outputs, targets):
        return self.loss_func(outputs, targets)
    
def train_net(dataset, epochs, lr, batch_size, lr_step_size, lr_gamma, savename):
    train_data = MotionData(dataset)
    train_set = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    model = ResidualNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    save_path = "./models"
    os.makedirs(save_path, exist_ok=True)
    savename = os.path.join(save_path, savename)
    
    for epoch in range(epochs):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            print("epoch: {} loss: {}".format(epoch, loss.item()))
            torch.save(model.state_dict(), savename)
