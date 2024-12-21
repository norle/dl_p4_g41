from pathlib import Path

from numpy import log10
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import wandb

from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from models import weights_init_uniform_rule, EfficientNetModel
import optuna
import plotly.express as px
import numpy as np

# GROUP NUMBER
GROUP_NUMBER = 41
EPOCHS = 1
EPOCHS_TRIAL = 1
TRIALS = 1
# CONSTANTS TO MODIFY AS YOU WISH
MODEL = EfficientNetModel
NUM_WORKERS = 4
OPTIMIZER = torch.optim.Adam
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# DEVICE = "cpu"

# You can set the model path name in case you want to keep training it.
# During the training/testing loop, the model state is saved
# (only the best model so far is saved)
LOAD_MODEL_FNAME = None
LOAD_MODEL_PATH = f"model_{MODEL.__name__}_noble-meadow-16"

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = "/dtu/blackhole/06/156422/models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)


def train_one_epoch(loss_fn, model, train_data_loader, optimizer):
    running_loss = 0.
    last_loss = 0.
    total_loss = 0.

    for i, data in enumerate(train_data_loader):
        spectrogram, target = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(spectrogram)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(), target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        total_loss += loss.item()
        if i % train_data_loader.batch_size == train_data_loader.batch_size - 1:
            last_loss = running_loss / train_data_loader.batch_size # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return total_loss / (i+1)


def objective(trial):
    # Suggest hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.01, 0.3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Define dataset transforms
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

    # Load training and testing datasets
    dataset_train = MODEL.dataset(data_dir=data_dir / "train",
                                  stmf_data_path=DATA_ROOT / STMF_FILENAME,
                                  transform=TRAIN_TRANSFORM)

    dataset_test = MODEL.dataset(data_dir=data_dir / "test",
                                 stmf_data_path=DATA_ROOT / STMF_FILENAME,
                                 transform=TEST_TRANSFORM)
    
    train_data_loader = DataLoader(dataset_train, 
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=NUM_WORKERS)
    test_data_loader = DataLoader(dataset_test,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)

    # Model, Loss, and Optimizer
    model = MODEL(dropout_rate=dropout_rate).to(DEVICE)
    model.apply(weights_init_uniform_rule)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = OPTIMIZER(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training Loop (simplified)
    best_rmse = float('inf')  # Initialize to a large value to track the best RMSE
    
    for epoch in range(EPOCHS_TRIAL):
        print(f'EPOCH {epoch + 1}:')
        
        # Set model to train mode
        model.train(True)

        # Perform training for one epoch
        avg_loss = train_one_epoch(MODEL.loss_fn, model, train_data_loader, optimizer)

        # Set the model to evaluation mode
        model.eval()
        running_test_loss = 0.
        total_samples = 0

        # Disable gradient computation and evaluate the test data
        with torch.no_grad():
            for i, vdata in enumerate(test_data_loader):
                # Get data and targets
                spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)

                # Get model outputs
                test_outputs = model(spectrogram)

                # Calculate the loss
                test_loss = MODEL.loss_fn(test_outputs.squeeze(), target)

                # Update running test loss and total samples
                running_test_loss += test_loss.item() * spectrogram.size(0)
                total_samples += spectrogram.size(0)

        # Calculate average test loss
        avg_test_loss = running_test_loss / total_samples

        # Calculate the RMSE for the testing predictions
        test_rmse = avg_test_loss**0.5

        # Track the best RMSE value across epochs
        if test_rmse < best_rmse:
            best_rmse = test_rmse

        print(f'Epoch {epoch + 1}: Test RMSE = {test_rmse}, Best RMSE So Far = {best_rmse}')

    # Log metrics for the entire trial once at the end
    wandb.log({
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "best_test_rmse": best_rmse
    })

    # Return the best RMSE value as the objective to be minimized by Optuna
    return best_rmse  # Optuna will try to minimize this value



# Plotting the optimization history with the best parameters stated below the plot
def plot_parallel_coordinates(study, model_name):
    # Obtain the dataframe of trials from the study
    trials_df = study.trials_dataframe()
    print(trials_df)
    
    # Select the relevant columns
    trials_df = trials_df[['params_learning_rate', 'params_weight_decay', 'params_dropout_rate', 'params_batch_size', 'value']]
    
    # Rename columns for readability in the plot
    trials_df.columns = ['Learning Rate', 'Weight Decay', 'Dropout Rate', 'Batch Size', 'Validation Loss']

    # Apply log transformation for specific columns if needed
    trials_df['Learning Rate (log10)'] = np.log10(trials_df['Learning Rate'])
    trials_df['Weight Decay (log10)'] = np.log10(trials_df['Weight Decay'])
    
    # Apply log transformation for Validation Loss
    trials_df['Validation Loss (log10)'] = np.log10(trials_df['Validation Loss'])

    # Identify the best trial based on validation loss
    best_trial_index = trials_df['Validation Loss'].idxmin()
    best_trial = trials_df.loc[best_trial_index]

    # Prepare the text for the best hyperparameters
    best_params_text = (
        f"Best Parameters:\n"
        f"Learning Rate: 1e{best_trial['Learning Rate (log10)']:.2f},\n"
        f"Weight Decay: 1e{best_trial['Weight Decay (log10)']:.2f},\n"
        f"Dropout Rate: {best_trial['Dropout Rate']:.2f},\n"
        f"Batch Size: {best_trial['Batch Size']:.0f},\n"
        f"Validation RMSE: {best_trial['Validation Loss']:.2f}"
    )

    # Define the parallel coordinates plot with Validation Loss (logarithmic)
    fig = px.parallel_coordinates(
        trials_df,
        dimensions=['Learning Rate (log10)', 'Weight Decay (log10)', 'Dropout Rate', 'Batch Size', 'Validation Loss (log10)'],
        color='Validation Loss (log10)',
        color_continuous_scale=px.colors.sequential.Viridis_r,  # Reversed Viridis scale
        title='Parallel Coordinate Plot for Hyperparameter Tuning',
        labels={
            'Learning Rate (log10)': 'Learning Rate (log10)',
            'Weight Decay (log10)': 'Weight Decay (log10)',
            'Dropout Rate': 'Dropout Rate',
            'Batch Size': 'Batch Size',
            'Validation Loss (log10)': 'Validation Loss (log10)'
        }
    )

    # Add annotation with the best parameters text
    fig.add_annotation(
        text=best_params_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.15,  # Adjust x and y to position below the plot
        showarrow=False,
        font=dict(size=10),
        align="left"
    )

    fig.write_image("plots/hyperparameter_tuning_plot_" + model_name + ".png")


if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # DATA SET SETUP
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    # Initialize wandb once before the study.optimize() call
    wandb.init(
        project=f"02456_group_{GROUP_NUMBER}",
        name="optuna_hyperparameter_search",
        config={
            "architecture": MODEL.__name__,
            "epochs": EPOCHS,
            "optimizer": OPTIMIZER.__name__,
            "device": DEVICE,
            "nfft": NFFT,
        }
    )

    # Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=TRIALS)

    best_params = study.best_params

    print("Best hyperparameters: ", best_params)

    wandb.finish()

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
    dataset_train = MODEL.dataset(data_dir= data_dir / "train",
                           stmf_data_path = DATA_ROOT / STMF_FILENAME,
                           transform=TRAIN_TRANSFORM)

    dataset_test = MODEL.dataset(data_dir= data_dir / "test",
                           stmf_data_path = DATA_ROOT / STMF_FILENAME,
                           transform=TEST_TRANSFORM)
    
    train_data_loader = DataLoader(dataset_train, 
                                   batch_size=best_params['batch_size'],
                                   shuffle=True,
                                   num_workers=NUM_WORKERS)
    test_data_loader = DataLoader(dataset_test,
                                  batch_size=best_params['batch_size'],
                                  shuffle=False,
                                  num_workers=1)
    
    # If you want to keep training a previous model
    if LOAD_MODEL_FNAME is not None:
        model = MODEL().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / LOAD_MODEL_FNAME))
        model.eval()
    else:
        model = MODEL(best_params["dropout_rate"]).to(DEVICE)
        model.apply(weights_init_uniform_rule)

    optimizer = OPTIMIZER(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    
    
    # Set up wandb for reporting
    wandb.init(
        project=f"02456_group_{GROUP_NUMBER}",
        config={
            "learning_rate": best_params['learning_rate'],
            "architecture": MODEL.__name__,
            "dataset": MODEL.dataset.__name__,
            "epochs": EPOCHS,
            "batch_size": best_params['batch_size'],
            'weight_decay': best_params['weight_decay'],
            'dropout': best_params['dropout_rate'],
            "transform": "|".join([str(tr).split(".")[1].split(" ")[0] for tr in dataset_train.transform.transforms]),
            "optimizer": OPTIMIZER.__name__,
            "loss_fn": model.loss_fn.__name__,
            "nfft": NFFT
        }
    )

    # Define model output to save weights during training
    #MODEL_DIR.mkdir(exist_ok=True)
    model_name = f"model_{MODEL.__name__}_{wandb.run.name}"
    model_path = MODEL_DIR + "/" + model_name

    plot_parallel_coordinates(study=study, model_name=model_name)

    ## TRAINING LOOP
    epoch_number = 0
    best_vloss = 1_000_000.

    # import pdb; pdb.set_trace()

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on
        model.train(True)

        # Do a pass over the training data and get the average training MSE loss
        avg_loss = train_one_epoch(MODEL.loss_fn, model, train_data_loader, optimizer)
        
        # Calculate the root mean squared error: This gives
        # us the opportunity to evaluate the loss as an error
        # in natural units of the ball velocity (m/s)
        rmse = avg_loss**(1/2)

        # Take the log as well for easier tracking of the
        # development of the loss.
        log_rmse = log10(rmse)

        # Reset test loss
        running_test_loss = 0.

        # Set the model to evaluation mode
        model.eval()

        # Disable gradient computation and evaluate the test data
        with torch.no_grad():
            for i, vdata in enumerate(test_data_loader):
                # Get data and targets
                spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)
                
                # Get model outputs
                test_outputs = model(spectrogram)

                # Calculate the loss
                test_loss = MODEL.loss_fn(test_outputs.squeeze(), target)

                # Add loss to runnings loss
                running_test_loss += test_loss

        # Calculate average test loss
        avg_test_loss = running_test_loss / (i + 1)

        # Calculate the RSE for the training predictions
        test_rmse = avg_test_loss**(1/2)

        # Take the log as well for visualisation
        log_test_rmse = torch.log10(test_rmse)

        print('LOSS train {} ; LOSS test {}'.format(avg_loss, avg_test_loss))
        
        # log metrics to wandb
        wandb.log({
            "loss": avg_loss,
            "rmse": rmse,
            "log_rmse": log_rmse,
            "test_loss": avg_test_loss,
            "test_rmse": test_rmse,
            "log_test_rmse": log_test_rmse,
        })

        # Track best performance, and save the model's state
        if avg_test_loss < best_vloss:
            best_vloss = avg_test_loss
            torch.save(model, model_path)

        epoch_number += 1

    wandb.finish()