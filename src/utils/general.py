from typing import List
import torch
from PIL import Image
import os
import numpy as np

def ensure_directory(file_path: str) -> None:
    """Ensure that the directory for the given file path exists.

    Args:
    file_path (str): The full file path for which the directory should be ensured.

    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def normalize_img_for_imshow(image):
    """
    Normalizes an image array to a valid range for visualization with Matplotlib's imshow function.

    :param image: NumPy array. The input image with values either as floats in the range [0, 1]
                  or integers in the range [0, 255].
    :return: NumPy array. The normalized image with values clipped to the valid range.
    """
    # Check if the image is in float format
    if image.dtype in [np.float32, np.float64]:
        # Clip the image to the range [0, 1] to ensure all values are valid
        image = np.clip(image, 0, 1)
    # Check if the image is in integer format
    elif image.dtype in [np.int32, np.int64, np.uint8]:
        # Clip the image to the range [0, 255] for valid integer values
        image = np.clip(image, 0, 255)
    return image

def is_image_file(path: str) -> bool:
    """Check if the provided path ends with a '.jpg' or '.png' extension.
    
    Args:
    path (str): The file path to check.

    Returns:
    bool: True if the file is a JPEG or PNG image, False otherwise.
    """
    return path.lower().endswith(('.jpg', '.png'))

def remove_image_filename(path: str) -> str:
    """Remove the image file name from the path if it ends with a recognized image extension.
    
    Args:
    path (str): The full path to potentially an image file.
    
    Returns:
    str: The path without the image file name if it was an image, otherwise the original path.
    """
    if path.lower().endswith(('.jpg', '.png')):
        return os.path.dirname(path)
    return path

def tensors_to_pil(images: List[torch.Tensor]) -> List[Image.Image]:
    """Convert a list of torch tensors to a list of PIL images, handling data types and array shapes correctly.
    
    Args:
    images (List[torch.Tensor]): List of tensors representing images.
    
    Returns:
    List[Image.Image]: List of converted PIL images.
    """
    pil_images = []
    for img_tensor in images:
        # Move tensor to CPU and detach from computation graph
        img_tensor = img_tensor.cpu().detach()
        
        # Normalize the tensor to [0, 1] if necessary
        if img_tensor.max() > 1.0:
            img_tensor /= 255.0

        # Ensure the tensor is in the form (H, W, C) for RGB or (H, W) for grayscale
        if img_tensor.dim() == 3 and img_tensor.shape[0] in {1, 3}:  # C, H, W
            img_tensor = img_tensor.permute(1, 2, 0)  # H, W, C

        # Convert to numpy and ensure data type is uint8
        img_array = img_tensor.numpy()
        if img_tensor.shape[-1] == 1:  # Grayscale image case
            img_array = img_array.squeeze(-1)
        img_array = (img_array * 255).astype('uint8')

        # Create PIL image from numpy array
        pil_image = Image.fromarray(img_array)
        pil_images.append(pil_image)
    
    return pil_images