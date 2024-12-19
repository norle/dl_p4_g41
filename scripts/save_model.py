
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 65)
        self.fc2 = nn.Linear(65, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize and train your model
model = SimpleNN()
# ...training code...

# Save the model architecture and state dictionary
model_path = 'models/simple_nn_model_4_ch_3_65.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': SimpleNN
}, model_path)