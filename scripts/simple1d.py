from pathlib import Path

from numpy import log10
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

from datasets import SpectrogramDataset

from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thop import profile

# GROUP NUMBER
GROUP_NUMBER = 41

# CONSTANTS TO MODIFY AS YOU WISH

LEARNING_RATE = 10**-4
WEIGHT_DECAY = 1e-4
EPOCHS = int(2e6) # the model converges in test perfermance after ~250-300 epochs
BATCH_SIZE = 32
NUM_WORKERS = 4
OPTIMIZER = torch.optim.Adam
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#DEVICE = "cpu"

# You can set the model path name in case you want to keep training it.
# During the training/testing loop, the model state is saved
# (only the best model so far is saved)
LOAD_MODEL_FNAME = None
#LOAD_MODEL_FNAME = f"model_{MODEL.__name__}_colorful-wave-84"

# CONSTANTS TO LEAVE
ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, int(hidden_size*2))
        #self.bn1 = torch.nn.BatchNorm1d(hidden_size*2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(int(hidden_size*2), hidden_size)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.dropout = torch.nn.Dropout(p=0.25)
        #self.fc3 = torch.nn.Linear(hidden_size, hidden_size)

        self.fc4 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.dropout(x)
        out = self.fc1(out)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.bn2(out)
        out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
  
        out = self.fc4(out)

        return out

def transform_input(data):
    input_data = data[:, 5, :, :]  # Use only the 6th channel
    input_data = input_data - 0.5
    projected = input_data.sum(dim=2)  # Average over Y axis
    std_dev = projected.std(dim=1, keepdim=True)  # Calculate std deviation for each input
    normalized = projected / std_dev  # Divide by the std deviation

    return normalized.numpy()

def train_nn(model, X_train, y_train, X_test, y_test, optimizer, epochs):
    model.train()
    inputs = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)  # Reshape targets
    test_inputs = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    test_targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(DEVICE)  # Reshape test targets

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10000 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_inputs)
                test_loss = F.mse_loss(test_outputs, test_targets)
            model.train()
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

def test_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_val = y_val.cpu().numpy().reshape(-1, 1)  # Reshape y_val
        y_val_pred = model(X_val).cpu().numpy()

    val_error = np.mean((y_val - y_val_pred) ** 2)
    print(f"Validation Mean Squared Error: {val_error}")

    return y_val_pred, val_error

if __name__ == "__main__":
    print(f"Using {DEVICE} device")
    state = 'train'
    # DATA SET SETUP
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    TRAIN_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "train"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )
    TEST_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "test"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )
    VAL_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "validation"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )
    dataset_train = SpectrogramDataset(data_dir= data_dir / "train",
                                  stmf_data_path = DATA_ROOT / STMF_FILENAME,
                                  transform=TRAIN_TRANSFORM)

    dataset_test = SpectrogramDataset(data_dir= data_dir / "test",
                                 stmf_data_path = DATA_ROOT / STMF_FILENAME,
                                 transform=TEST_TRANSFORM)
    dataset_val = SpectrogramDataset(data_dir= data_dir / "validation",
                                 stmf_data_path = DATA_ROOT / STMF_FILENAME,
                                 transform=VAL_TRANSFORM)
    
    print(dataset_train[0])
    spectrogram_train_og = [dataset_train[i]["spectrogram"] for i in range(len(dataset_train))]
    spectrogram_train_og = torch.stack(spectrogram_train_og)
    target_train = [dataset_train[i]["target"] for i in range(len(dataset_train))]
    target_train = torch.stack(target_train)

    spectrogram_test_og = [dataset_test[i]["spectrogram"] for i in range(len(dataset_test))]
    spectrogram_test_og = torch.stack(spectrogram_test_og)
    target_test = [dataset_test[i]["target"] for i in range(len(dataset_test))]
    target_test = torch.stack(target_test)

    spectrogram_val_og = [dataset_val[i]["spectrogram"] for i in range(len(dataset_val))]
    spectrogram_val_og = torch.stack(spectrogram_val_og)
    target_val = [dataset_val[i]["target"] for i in range(len(dataset_val))]
    target_val = torch.stack(target_val)

    spectrogram_train = transform_input(spectrogram_train_og)
    spectrogram_test = transform_input(spectrogram_test_og)
    spectrogram_val = transform_input(spectrogram_val_og)
    if state == 'train':
        # Save transformed train inputs to CSV
        train_inputs_df = pd.DataFrame(spectrogram_train)
        #train_inputs_df.to_csv('figures/train_lr_nn.csv', index=False)

        input_size = spectrogram_train.shape[1]
        print(input_size)
        hidden_size = 50
        output_size = 1

        model = SimpleNN(input_size, hidden_size, output_size).to(DEVICE)
        criterion = None  # No need for criterion as we use F.mse_loss directly
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        X_train = spectrogram_train
        print(X_train.shape)
        y_train = target_train.cpu().numpy().reshape(-1, 1)  # Reshape y_train
        X_test = spectrogram_test
        y_test = target_test.cpu().numpy().reshape(-1, 1)  # Reshape y_test

        # Evaluate number of parameters and FLOPs
        dummy_input = torch.randn(1, input_size).to(DEVICE)
        flops, params = profile(model, inputs=(dummy_input,))
        print(f"Number of parameters: {params}")
        print(f"Number of FLOPs: {flops}")

        train_nn(model, X_train, y_train, X_test, y_test, optimizer, EPOCHS)

        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            X_test = torch.tensor(spectrogram_test, dtype=torch.float32).to(DEVICE)
            y_test = target_test.cpu().numpy().reshape(-1, 1)  # Reshape y_test
            y_test_pred = model(X_test).cpu().numpy()

        test_error = np.mean((y_test - y_test_pred) ** 2)
        print(f"Test Mean Squared Error: {test_error}")

        # Save the model
        torch.save(model.state_dict(), MODEL_DIR / "simple_nn_model.pth")
        # Plot and save the error distribution
        errors = y_test - y_test_pred
        plt.figure()
        plt.hist(errors, bins=50)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.savefig('figures/error_dist_nn.png')
        plt.close()
    else:
        # Test a model
        # Load the model
        model_path = 'models/simple_nn_model_4_ch_3_65.pth'
        model = SimpleNN(74, 50, 1).to(DEVICE)
        state_dict = torch.load(model_path)
        model.load_state_dict({k: v for k, v in state_dict.items() if k not in ['total_ops', 'total_params']})

        # Ensure the model is in evaluation mode
        model.eval()
        # Save the entire model architecture and weights
        torch.save(model, MODEL_DIR / "simple_nn_model_full.pth")
        print('Model saved')
        # Profile the model
        dummy_input = torch.randn(1, 74).to(DEVICE)
        flops, params = profile(model, inputs=(dummy_input,))
        print(f"Number of parameters: {params}")
        print(f"Number of FLOPs: {flops}")

        # Test the model on validation data
        y_val_pred, val_error = test_model(model, spectrogram_val, target_val)

        # Plot and save the validation error distribution
        val_errors = target_val.cpu().numpy().reshape(-1, 1) - y_val_pred
        plt.figure()
        plt.hist(val_errors, bins=50)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Validation Error Distribution')
        plt.savefig('figures/val_error_dist_nn.png')
        plt.close()



