import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from models import SpectrVelCNNRegr
import numpy as np
import scipy.stats as stats


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

# Models
baseline_model = "model_SpectrVelCNNRegr_splendid-sea-255"
my_model = "model_VelocityEstimation4_warm-glade-293"
resnet_model = "model_ResNet_noble-serenity-401"

MODEL_FILENAMES = [baseline_model, my_model, resnet_model]
legends = ["Baseline", "Adjusted Baseline", "ResNet"]

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

import scipy.stats as stats

# Define custom legends and linestyles
CUSTOM_LEGENDS = ["Baseline", "Adjusted Baseline", "ResNet"]
CUSTOM_LINESTYLES = ["solid", "dashed", "dotted"]  # Different linestyles
LINE_WIDTH = 2.5  # Set line width

if __name__ == "__main__":
    # Loop through each model and evaluate it
    VALIDATION_RMSE_LIST = []
    RESIDUALS_BY_MODEL = {}

    for model_filename in MODEL_FILENAMES:
        avg_rmse, total_residuals = evaluate_on_validation_set(model_filename)
        VALIDATION_RMSE_LIST.append(avg_rmse)

        # Store residuals for histogram
        if total_residuals:
            flattened_residuals = torch.cat([residual.view(-1) for residual in total_residuals], dim=0).cpu().numpy()
            RESIDUALS_BY_MODEL[model_filename] = flattened_residuals

    # Create one figure with histograms for all models
    plt.figure(figsize=(10, 6))
    for model_filename, residuals, legend in zip(MODEL_FILENAMES, RESIDUALS_BY_MODEL.values(), CUSTOM_LEGENDS):
        # Calculate mean and standard deviation
        mean = np.mean(residuals)
        std_dev = np.std(residuals)

        # Update legend to include mean and standard deviation
        legend_with_stats = f"{legend} (\u03bc={mean:.2f}, \u03c3={std_dev:.2f})"

        plt.hist(residuals, bins=15, alpha=0.8, edgecolor='black', label=legend_with_stats)

    # Add title, labels, legend, and adjust font sizes
    plt.title("Residuals Comparison Across Models", fontsize=30)
    plt.xlabel("Residual Value [m/s]", fontsize=25)
    plt.ylabel("Frequency", fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Save the combined plot
    combined_plot_filename = PLOTS_DIR / "combined_residuals_histogram.png"
    plt.savefig(combined_plot_filename)
    plt.close()  # Close the figure to free up memory
    print(f"Combined histogram saved at {combined_plot_filename}")

    # Create a plot for normal distributions based on mean and variance
    plt.figure(figsize=(10, 6))

    x = np.linspace(-5, 5, 1000)  # Define range for plotting distributions

    for model_filename, residuals, legend, linestyle in zip(MODEL_FILENAMES, RESIDUALS_BY_MODEL.values(), CUSTOM_LEGENDS, CUSTOM_LINESTYLES):
        mean = np.mean(residuals)
        variance = np.var(residuals)
        std_dev = np.sqrt(variance)

        # Create normal distribution
        normal_dist = stats.norm.pdf(x, mean, std_dev)

        # Plot the normal distribution with custom linestyle and thickness
        plt.plot(x, normal_dist, label=f"{legend} (\u03bc={mean:.2f}, \u03c3\u00b2={variance:.2f})", linestyle=linestyle, linewidth=LINE_WIDTH)

    # Add title, labels, legend, and adjust font sizes
    plt.title("Normal Distributions of Residuals", fontsize=30)
    plt.xlabel("Residual Value [m/s]", fontsize=25)
    plt.ylabel("Probability Density", fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(fontsize=11)
    plt.grid(True)

    # Save the normal distribution plot
    normal_plot_filename = PLOTS_DIR / "normal_distributions.png"
    plt.savefig(normal_plot_filename)
    plt.close()  # Close the figure to free up memory
    print(f"Normal distribution plot saved at {normal_plot_filename}")

    # Print validation RMSE for each model
    for model, rmse in zip(CUSTOM_LEGENDS, VALIDATION_RMSE_LIST):
        print(f"Model: {model}")
        print(f"Validation RMSE: {rmse}")
