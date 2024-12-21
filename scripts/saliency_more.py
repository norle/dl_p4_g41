import torch
import torch.nn.functional as F
import numpy as np
import zipfile
from datasets import SpectrogramDataset
from torch.utils.data import DataLoader
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from pathlib import Path
from torchvision.transforms import transforms
from data_management import make_dataset_name
from loss import mse_loss
import matplotlib.pyplot as plt
from models import leaky, VelocityEstimation4
import torch.nn as nn

def smooth_grad(model, input_tensor, labels, num_samples=50, noise_level=0.1):
    """
    Compute SmoothGrad saliency map for a regression CNN model.

    Args:
        model: The regression CNN model (e.g., your Thomas model).
        input_tensor: Input tensor (requires_grad=True).
        num_samples: Number of noisy samples to generate.
        noise_level: Standard deviation of Gaussian noise to add to the input.

    Returns:
        SmoothGrad saliency map (same shape as input_tensor).
    """

    print(device)

    model.eval()  # Set model to evaluation mode
    input_tensor = input_tensor.unsqueeze(0) if len(input_tensor.shape) == 3 else input_tensor
    input_tensor.requires_grad = True  # Enable gradient computation for the input

    # Initialize saliency map
    smooth_grad_map = torch.zeros_like(input_tensor)
    stdev = noise_level * (input_tensor.max() - input_tensor.min()).item()

    # Generate noisy samples and compute gradient
    for _ in range(num_samples):
        # Add noise to the input
        noise = torch.randn_like(input_tensor) * stdev
        noisy_image= input_tensor + noise

        noisy_image = noisy_image.to('cuda')
        noisy_image.requires_grad_()
        # Forward pass
        output = model(noisy_image)
        output = output.view(-1)

        print(output.shape)
        print(labels.shape)
        loss = mse_loss(output.squeeze(), labels)
        # Compute gradients of the target output with respect to the input
        model.zero_grad()  # Clear previous gradients
        loss.backward()

        #gradient = torch.ones_like(output)  # Gradient must match output shape
        #output.backward(gradient=gradient)  # Backprop

        # Accumulate gradients
        smooth_grad_map += input_tensor.grad.data.abs()

        # Reset gradients for next iteration
        input_tensor.grad.zero_()

    # Average the gradients

    smooth_grad_map /= num_samples

    smooth_grad_map = smooth_grad_map.cpu().numpy()

    smooth_grad_map = np.max(smooth_grad_map, axis=1)

    return smooth_grad_map



def plot_saliency_maps(models, sigmas, input_tensor, labels, TS_CROPTWIDTH, VR_CROPTWIDTH):
    # Set up a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs = axs.ravel()

    # Original Image Plot (First Channel)
    float_value = labels.item()

    original_image_first_channel = input_tensor[0, 0, :, :].detach().cpu().numpy()
    axs[0].imshow(original_image_first_channel, aspect="auto", 
                  extent=[TS_CROPTWIDTH[0]/1000, TS_CROPTWIDTH[1]/1000,
                          VR_CROPTWIDTH[0], VR_CROPTWIDTH[1]],
                  origin="lower", interpolation='nearest', cmap="jet")
    axs[0].plot([TS_CROPTWIDTH[0]/1000, TS_CROPTWIDTH[1]/1000], [float_value, float_value], 'w--')
    axs[0].set_title("Original Image (First Channel)")
    axs[0].axis('off')
    axs[0].legend([ r"True $v_{r}$", r"Pred. $\bar{v}_{r}$"])
    
    print(labels)
    models_name = ['Baseline', 'ResNet', 'CNN']
    # Loop through each model to create saliency maps
    for i, (model, sigma) in enumerate(zip(models, sigmas)):
        saliency_map = smooth_grad(model, input_tensor, labels, num_samples=50, noise_level=sigma)
        estimated = model(input_tensor)
        estimated = estimated.item()
        # Ensure the saliency map is in the correct shape
        saliency_map_reshaped = saliency_map.squeeze()

        # Plot the saliency map in the corresponding subplot
        axs[i + 1].imshow(saliency_map_reshaped, origin='lower', aspect="auto",
                          extent=[TS_CROPTWIDTH[0]/1000, TS_CROPTWIDTH[1]/1000,
                                  VR_CROPTWIDTH[0], VR_CROPTWIDTH[1]],
                          interpolation='nearest', cmap='hot')
        axs[i + 1].plot([TS_CROPTWIDTH[0]/1000, TS_CROPTWIDTH[1]/1000], [float_value, float_value], 'w--')
        axs[i + 1].plot([TS_CROPTWIDTH[0]/1000, TS_CROPTWIDTH[1]/1000], [estimated, estimated], 'w:')
        axs[i + 1].legend([ r"True $v_{r}$", r"Pred. $\bar{v}_{r}$"])
        axs[i + 1].set_title(f"Saliency Map for {models_name[i]} model (std={sigma})")
        axs[i + 1].axis('off')

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f"plots/saliency_map_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Example usage
    DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
    STMF_FILENAME = "stmf_data_3.csv"
    NFFT = 512
    TS_CROPTWIDTH = (-150, 200)
    VR_CROPTWIDTH = (-60, 15)

    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    TEST_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "validation"),
         NormalizeSpectrogram(),
         ToTensor(),
         InterpolateSpectrogram()]
    )

    models_path = "/dtu/blackhole/06/156422/models/"
    model_names = ['model_SpectrVelCNNRegr_noble-butterfly-372', 'model_ResNet_noble-serenity-401', 'model_VelocityEstimation4_warm-glade-293']

    # Load models
    # state_dict = torch.load(models_path + model_names[2], map_location=torch.device('cuda'))

    # # Optionally inspect the keys to confirm the layers
    # print(state_dict)

    models = [torch.load(models_path + model_name, weights_only=False) for model_name in model_names]

    # Load dataset and data loader
    dataset_test = SpectrogramDataset(data_dir=data_dir / "validation",
                                      stmf_data_path=DATA_ROOT / STMF_FILENAME,
                                      transform=TEST_TRANSFORM)
    test_data_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    # Device setup
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Move models to the appropriate device
    for model in models:
        model.to(device)
        model.eval()

    # Get one batch of data
    images = next(iter(test_data_loader))
    input_tensor, labels = images["spectrogram"].to(device), images["target"].to(device)

    # Plotting saliency maps for the three models
    sigmas = [0.3, 0.05, 0.05]  # You can control these noise levels individually for each model
    plot_saliency_maps(models, sigmas, input_tensor, labels, TS_CROPTWIDTH, VR_CROPTWIDTH)
