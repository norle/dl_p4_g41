#%%
from torch.nn import Module
from models import SpectrVelCNNRegr, AttentionModel
from thop import profile
import torch

def print_model_complexity(model) -> None:
    """Check and print the number of parameters in the network

    Args:
        model (module): Pytorch model class
    """
    dummy_input = torch.randn(1, 6, 74, 918)  # Adjust the input size as needed
    flops, params = profile(model(), inputs=(dummy_input,))
    
    print(f"Number of parameters in model {model.__name__}: {params} = {'{:.2e}'.format(params)}")
    print(f"FLOPs: {flops} = {'{:.2e}'.format(flops)}")

# %%
if __name__ == "__main__":
    model = SpectrVelCNNRegr
    print_model_complexity(AttentionModel)
    #print_model_complexity(FullyConv_v1)
    #print_model_complexity(EfficientNet)

# %%
