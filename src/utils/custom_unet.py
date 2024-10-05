from diffusers import UNet2DConditionModel
import torch

class CustomUNet2DConditionModel(UNet2DConditionModel):
    def forward(self, *args, **kwargs):
        # Call the parent class's forward method
        output = super().forward(*args, **kwargs)
        
        # Extract the sample and h_vect
        sample = output.sample
        
        # The h_vect is the output of the middle block
        # We need to detach and clone it to avoid modifying the original tensor
        h_vect = self.mid_block.output[0].detach().clone()
        
        # Return both the sample and h_vect
        return sample, h_vect

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.mid_block.gradient_checkpointing = True