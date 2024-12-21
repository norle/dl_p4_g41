from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import sys

from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from fvcore.nn import FlopCountAnalysis

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

# # All
MODEL_LIST = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16]
inference_times = []
std_times = []
median_inference_times = []


## Constants
#DEVICE = (
#   "cuda"
#   if torch.cuda.is_available()
#   else "mps"
#   if torch.backends.mps.is_available()
#   else "cpu"
#)


DEVICE = "cpu"


DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data")
MODEL_DIR = Path("scripts/models")
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)
BATCH_SIZE = 500
NUM_WORKERS = 1

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    for model_name in MODEL_LIST:
        # Load the model
        model_path = MODEL_DIR / model_name
        model = torch.load(model_path, map_location=DEVICE)
        model.to(DEVICE).eval()  # Ensure model is on device and in eval mode
        print(f"Loaded model from {model_path}")

        # Prepare a single input tensor
        input_tensor = torch.randn(1, 6, 74, 918, device=DEVICE)

        # Warm-up
        with torch.no_grad():
            for _ in range(500):
                _ = model(input_tensor)

        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(100):
                torch.cuda.synchronize()  # Ensure synchronization if using GPU
                t0 = time.time()
                _ = model(input_tensor)
                t1 = time.time()
                torch.cuda.synchronize()  # Synchronize again
                times.append(t1 - t0)

        average_time = np.mean(times)
        std_time = np.std(times)
        median_time = np.median(times)
        inference_times.append(average_time)
        std_times.append(std_time)
        median_inference_times.append(median_time)
    
    for i in range(len(MODEL_LIST)):
        print(f"Average inference time for {MODEL_LIST[i]}: {inference_times[i]:.6f} sec (std = {std_times[i]:.6f}) | Median = {median_inference_times[i]:.6f}")


