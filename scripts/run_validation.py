import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from models import SpectrVelCNNRegr
import numpy as np

# CONSTANTS
GROUP_NUMBER = 41
MODEL = SpectrVelCNNRegr
BATCH_SIZE = 1
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("/dtu-compute/02456-p4-e24/data")
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)
# Baseline
model1 = "model_SpectrVelCNNRegr_splendid-sea-255"
# Stride, removed FF layer
model2 = "model_VelocityEstimation1_scarlet-sea-290"
# Stride, removed FF layer, batchnorm
model3 = "model_VelocityEstimation2_sandy-pyramid-291"
# Stride, removed FF layer, dropout
model4 = "model_VelocityEstimation3_fluent-morning-292"

# Stride, removed FF layer, batchnorm, dropout
model5 = "model_VelocityEstimation4_warm-glade-293"
model12 = "model_VelocityEstimation11_amber-valley-318"
model13 = "model_VelocityEstimation12_denim-darkness-319"
model14 = "model_VelocityEstimation13_curious-firefly-320"

# Stride, batchnorm, dropout
model6 = "model_VelocityEstimation5_giddy-forest-294"
model7 = "model_VelocityEstimation6_lemon-field-295"
model8 = "model_VelocityEstimation7_devoted-energy-296"
model9 = "model_VelocityEstimation8_celestial-music-299"
model10 = "model_VelocityEstimation9_expert-brook-316"

# Stride, batchnorm
model11 = "model_VelocityEstimation10_comfy-bird-317"

# Other guys models
model15 = "leakyrelu"
model16 = "model_ResNet_noble-serenity-401"

MODEL_FILENAMES = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16]

MODEL_DIR = Path("scripts/models")

def evaluate_on_validation_set(model_filename):
    print(f"Starting validation evaluation for model: {model_filename}")

    # Setup dataset name and paths
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    # Define transforms for the validation set
    VALIDATION_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "validation"),
         NormalizeSpectrogram(),
         ToTensor(),
         InterpolateSpectrogram()]
    )

    # Instantiate validation dataset
    print("Creating validation dataset...")
    try:
        dataset_val = MODEL.dataset(data_dir=data_dir / "validation",
                                    stmf_data_path=DATA_ROOT / STMF_FILENAME,
                                    transform=VALIDATION_TRANSFORM)
    except Exception as e:
        print(f"Failed to create validation dataset: {e}")
        return

    # Instantiate DataLoader for validation set
    print("Creating validation DataLoader...")
    try:
        val_data_loader = DataLoader(dataset_val,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=NUM_WORKERS)
    except Exception as e:
        print(f"Failed to create validation DataLoader: {e}")
        return

    # Load model
    model_path = Path(MODEL_DIR) / model_filename
    print(f"Loading model from: {model_path}")
    try:
        model = torch.load(model_path, map_location=DEVICE)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Run validation and calculate metrics
    print("Running validation...")
    running_loss = 0.0
    total_samples = 0
    total_residuals = []

    # Set model to evaluation mode and turn off gradients
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_data_loader):
            # Extract spectrogram and target from batch
            spectrogram, target = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)

            # Forward pass
            outputs = model(spectrogram)

            # COMPUTE RESIDUALS HERE
            residuals = outputs - target

            total_residuals.append(residuals)

            # Calculate loss (assuming model has a loss function defined)
            loss_fn = MODEL.loss_fn  # Replace with appropriate loss function if different
            loss = loss_fn(outputs.squeeze(), target)

            # Update metrics
            running_loss += loss.item() * spectrogram.size(0)
            total_samples += spectrogram.size(0)

            # Calculate root mean squared error (RMSE)
            rmse = torch.sqrt(loss)

    # Calculate average loss over the entire validation set
    avg_loss = running_loss / total_samples
    avg_rmse = avg_loss ** 0.5  # Calculate RMSE from MSE loss

    return avg_rmse, total_residuals

import os
import matplotlib.pyplot as plt

# Ensure 'plots' directory exists
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Loop through each model and evaluate it
    VALIDATION_RMSE_LIST = []

    for model_filename in MODEL_FILENAMES:
        avg_rmse, total_residuals = evaluate_on_validation_set(model_filename)
        VALIDATION_RMSE_LIST.append(avg_rmse)

        # Create histogram for the residuals
        if total_residuals:
            # Flatten residuals
            flattened_residuals = torch.cat([residual.view(-1) for residual in total_residuals], dim=0).cpu().numpy()

            # Plot histogram
            plt.figure()
            plt.hist(flattened_residuals, bins=30, alpha=0.75, color='blue', edgecolor='black')
            plt.title(f"Residuals Histogram: {model_filename}", fontsize=10)
            plt.xlabel("Residual Value [m/s]", fontsize=20)
            plt.ylabel("Frequency", fontsize=20)
            plt.grid(True)

            # Save plot
            plot_filename = PLOTS_DIR / f"{model_filename}_residuals_histogram.png"
            plt.savefig(plot_filename)
            plt.close()  # Close the figure to free up memory
            print(f"Histogram saved for {model_filename} at {plot_filename}")

    # Print validation RMSE for each model
    for i in range(len(VALIDATION_RMSE_LIST)):
        print("Model: ", MODEL_FILENAMES[i])
        print("Validation RMSE: ", VALIDATION_RMSE_LIST[i])
