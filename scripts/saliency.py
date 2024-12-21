import torch
import torch.nn.functional as F
import numpy as np
# from models import ResNet
import zipfile
from datasets import SpectrogramDataset
from torch.utils.data import DataLoader
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from pathlib import Path
from torchvision.transforms import transforms
from data_management import make_dataset_name
from loss import mse_loss
import matplotlib.pyplot as plt

def smooth_grad(model, input_tensor, labels, num_samples=50, noise_level=0.1,device='cuda'):
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
    device = 'cuda'
    # print(device)

    model.eval()  # Set model to evaluation mode
    input_tensor = input_tensor.unsqueeze(0) if len(input_tensor.shape) == 3 else input_tensor
    input_tensor.requires_grad = True  # Enable gradient computation for the input

    # Initialize saliency map
    smooth_grad_map = torch.zeros_like(input_tensor)
    stdev = noise_level * (input_tensor.max() - input_tensor.min()).item()

    # Generate noisy samples and compute gradients
    for _ in range(num_samples):
        # Add noise to the input
        noise = torch.randn_like(input_tensor) * stdev
        noisy_image= input_tensor + noise

        noisy_image = noisy_image.to('cuda')
        noisy_image.requires_grad_()
        # Forward pass
        output = model(noisy_image)
        output = output.view(-1)

        # print("target", labels)
        # print("output", output)
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


if __name__ == "__main__":
    # Example usage
    DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
    STMF_FILENAME = "stmf_data_3.csv"
    print("saliency")
    NFFT = 512
    TS_CROPTWIDTH = (-150, 200)
    VR_CROPTWIDTH = (-60, 15)

    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    TEST_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "train"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )

    input_tensor = torch.randn(1, 6, 74, 918)  # Example input tensor (batch_size, channels, height, width)
    models_path = "/dtu/blackhole/06/156422/models/"
    model_name = 'model_ResNet_noble-serenity-401'
    #model = SpectrVelCNNRegr()  # Your regression CNN model
    model = torch.load(models_path+model_name, weights_only=False)
    dataset_test = SpectrogramDataset(data_dir= data_dir / "train",
                           stmf_data_path = DATA_ROOT / STMF_FILENAME,
                           transform=TEST_TRANSFORM)
    

    test_data_loader = DataLoader(dataset_test,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1)    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()
    images = next(iter(test_data_loader))
    input_tensor, labels = images["spectrogram"].to(device), images["target"].to(device)
    
    original_image_first_channel = input_tensor[0, 0, :, :].detach().cpu().numpy()
    print(input_tensor.shape)
    fig, axs = plt.subplots(1, 6, figsize=(36, 6))

    # Plot the original image (first channel) on the left
    vmin = -110
    vmax = -40
    
    axs[0].imshow(original_image_first_channel, aspect="auto", 
        extent=[TS_CROPTWIDTH[0]/1000,TS_CROPTWIDTH[1]/1000,
                VR_CROPTWIDTH[0],VR_CROPTWIDTH[1]],
        origin="lower",
        interpolation='nearest',
        cmap="jet")
    axs[0].set_title("Original Image (First Channel)")
    axs[0].axis('off')

    for i in range(1, 6):
        std = 0.01*i
        saliency_map = smooth_grad(model, input_tensor, labels, num_samples=50, noise_level=std)

        # Extract the first channel of the original image

        # Ensure the saliency map is in the correct shape
        saliency_map_reshaped = saliency_map.squeeze()

        # Create the figure with two subplots


        # Plot the saliency map on the right
        axs[i].imshow(saliency_map_reshaped, origin='lower', aspect="auto", 
            extent=[TS_CROPTWIDTH[0]/1000,TS_CROPTWIDTH[1]/1000,
                    VR_CROPTWIDTH[0],VR_CROPTWIDTH[1]],
            interpolation='nearest',
            cmap='hot')
        axs[i].set_title(f"Saliency Map (std={std})")
        axs[i].axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"plots/saliency_map_comparison_{model_name}.png", dpi=300)
    plt.show()

    # # axs[1].imshow(saliency_map_reshaped, aspect="auto", 
    #     extent=[TS_CROPTWIDTH[0]/1000,TS_CROPTWIDTH[1]/1000,
    #             VR_CROPTWIDTH[0],VR_CROPTWIDTH[1]],
    #     vmin=vmin, vmax=vmax,
    #     origin="lower",
    #     interpolation='nearest',
    #     cmap="hot")