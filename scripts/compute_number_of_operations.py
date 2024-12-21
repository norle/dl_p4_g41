from pathlib import Path
import torch
from torchvision.transforms import transforms
from fvcore.nn import FlopCountAnalysis, flop_count_table

from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name

# Constants
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

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

MODEL_LIST = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16]

FLOP_LIST = []

# Add other models as needed
MODEL_DIR = Path("scripts/models")
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data")
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)
INPUT_SHAPE = (1, 6, 74, 918)  # Batch size 1, 6 channels, 74x918 resolution


def compute_flops(model, input_tensor):
    """
    Compute the number of FLOPs for a forward pass through the model.

    Args:
        model (torch.nn.Module): The model to analyze.
        input_tensor (torch.Tensor): A sample input tensor.

    Returns:
        float: Total number of FLOPs in the model.
        str: Human-readable FLOP count table for detailed breakdown.
    """
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()
    flop_table = flop_count_table(flops)
    return total_flops, flop_table


if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    for model_name in MODEL_LIST:
        # Load the model
        model_path = MODEL_DIR / model_name
        model = torch.load(model_path, map_location=DEVICE)
        print(f"Loaded model from {model_path}")

        # Move model to device
        model.to(DEVICE)

        # Prepare input tensor for FLOP calculation
        input_tensor = torch.randn(INPUT_SHAPE, device=DEVICE)

        # Calculate FLOPs
        total_flops, flop_table = compute_flops(model, input_tensor)
        FLOP_LIST.append(total_flops)
    
    for i in range(len(MODEL_LIST)):
        print(f"{MODEL_LIST[i]}: {FLOP_LIST[i]} flops")
