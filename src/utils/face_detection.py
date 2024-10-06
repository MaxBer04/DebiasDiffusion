"""
Face Detection Module for DebiasDiffusion

This module provides face detection functionality for the DebiasDiffusion project.
It uses a combination of InsightFace and face_recognition libraries to detect and
align faces in images.

The FaceDetector class offers methods to detect faces, align them, and handle
cases where multiple faces are present in an image.

Usage:
    from src.utils.face_detection import get_face_detector

    face_detector = get_face_detector(gpu_id=0)
    success, bbox, face_chip, aligned_face = face_detector.detect_and_align_face(image_tensor)

Note:
    This module requires the InsightFace and face_recognition libraries to be installed.
"""

import torch
import numpy as np
from torchvision.transforms import ToTensor
import face_recognition
from insightface.app import FaceAnalysis
from PIL import Image
from skimage import transform
import kornia
from typing import Tuple, Optional, List, Union

class FaceDetector:
    def __init__(self, gpu_id: int = 0):
        """
        Initialize the FaceDetector.

        Args:
            gpu_id (int): The ID of the GPU to use. Defaults to 0.
        """
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=['detection'],
            providers=['CUDAExecutionProvider'],
            provider_options=[{'device_id': gpu_id}]
        )
        self.face_app.prepare(ctx_id=gpu_id, det_size=(640, 640))
        
    def detect_and_align_face(
        self,
        image_tensor: torch.Tensor,
        size_face: int = 224,
        size_aligned_face: int = 112,
        fill_value: float = -1
    ) -> Tuple[bool, Optional[List[int]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Detect and align a face in the given image tensor.

        Args:
            image_tensor (torch.Tensor): Input image tensor.
            size_face (int): Size of the output face chip. Defaults to 224.
            size_aligned_face (int): Size of the aligned face. Defaults to 112.
            fill_value (float): Fill value for padding. Defaults to -1.

        Returns:
            Tuple containing:
            - bool: True if a face was detected, False otherwise.
            - Optional[List[int]]: Bounding box of the detected face, or None if no face detected.
            - Optional[torch.Tensor]: Face chip tensor, or None if no face detected.
            - Optional[torch.Tensor]: Aligned face tensor, or None if no face detected.
        """
        image_np = ((image_tensor * 0.5 + 0.5) * 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        
        faces_insight = self.face_app.get(image_np[:,:,[2,1,0]])  # Convert to BGR for InsightFace
        
        if len(faces_insight) == 0:
            faces_fr = face_recognition.face_locations(image_np, model="cnn", number_of_times_to_upsample=0)
            if len(faces_fr) == 0:
                return False, None, None, None
            
            face_fr = self.get_largest_face_FR(faces_fr, dim_max=image_np.shape[0], dim_min=0)
            bbox = np.array((face_fr[-1],) + face_fr[:-1])  # Convert to [left, top, right, bottom]
            bbox = self.expand_bbox(bbox, expand_coef=1.1, target_ratio=1)
            
            face_landmarks = face_recognition.face_landmarks(image_np, face_locations=[face_fr], model="large")[0]
            landmarks = np.array([
                np.mean(face_landmarks['left_eye'], axis=0),
                np.mean(face_landmarks['right_eye'], axis=0),
                face_landmarks['nose_bridge'][-1],
                face_landmarks['top_lip'][0],
                face_landmarks['top_lip'][6]
            ])
        else:
            face = self.get_largest_face_app(faces_insight, dim_max=image_np.shape[0], dim_min=0)
            bbox = self.expand_bbox(face.bbox, expand_coef=0.5, target_ratio=1)
            landmarks = face.kps
        
        face_chip = self.crop_face(image_tensor, bbox, target_size=[size_face, size_face], fill_value=fill_value)
        aligned_face_chip = self.align_face(image_tensor, landmarks, size=size_aligned_face)
        
        return True, bbox.tolist(), face_chip, aligned_face_chip

    def expand_bbox(self, bbox: np.ndarray, expand_coef: float, target_ratio: float) -> List[int]:
        """
        Expand the bounding box to achieve a target aspect ratio.

        Args:
            bbox (np.ndarray): Original bounding box [left, top, right, bottom].
            expand_coef (float): Expansion coefficient.
            target_ratio (float): Target aspect ratio (height / width).

        Returns:
            List[int]: Expanded bounding box [left, top, right, bottom].
        """
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        current_ratio = bbox_height / bbox_width
        if current_ratio > target_ratio:
            more_height = bbox_height * expand_coef
            more_width = (bbox_height + more_height) / target_ratio - bbox_width
        else:
            more_width = bbox_width * expand_coef
            more_height = (bbox_width + more_width) * target_ratio - bbox_height
        
        bbox_new = [
            int(round(bbox[0] - more_width*0.5)),
            int(round(bbox[1] - more_height*0.5)),
            int(round(bbox[2] + more_width*0.5)),
            int(round(bbox[3] + more_height*0.5))
        ]
        return bbox_new

    def crop_face(
        self,
        img_tensor: torch.Tensor,
        bbox: List[int],
        target_size: List[int],
        fill_value: float
    ) -> torch.Tensor:
        """
        Crop the face from the image tensor based on the bounding box.

        Args:
            img_tensor (torch.Tensor): Input image tensor.
            bbox (List[int]): Bounding box [left, top, right, bottom].
            target_size (List[int]): Target size of the cropped face [height, width].
            fill_value (float): Fill value for padding.

        Returns:
            torch.Tensor: Cropped and resized face tensor.
        """
        img_height, img_width = img_tensor.shape[-2:]
        
        left, top, right, bottom = bbox
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        face = img_tensor[:, top:bottom, left:right]
        face = torch.nn.functional.interpolate(face.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        
        return face

    def align_face(self, img_tensor: torch.Tensor, landmarks: np.ndarray, size: int = 112) -> torch.Tensor:
        """
        Align the face using facial landmarks.

        Args:
            img_tensor (torch.Tensor): Input image tensor.
            landmarks (np.ndarray): Facial landmarks.
            size (int): Size of the output aligned face. Defaults to 112.

        Returns:
            torch.Tensor: Aligned face tensor.
        """
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        dst = landmarks.astype(np.float32)
        tform = transform.SimilarityTransform()
        tform.estimate(dst, src)
        
        M = torch.tensor(tform.params[0:2, :]).unsqueeze(0).to(img_tensor.device).to(img_tensor.dtype)
        
        aligned_face = kornia.geometry.transform.warp_affine(
            img_tensor.unsqueeze(0), 
            M, 
            dsize=(size, size), 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ).squeeze(0)
        
        return aligned_face

    def get_largest_face_app(self, faces: List[Any], dim_max: int, dim_min: int) -> Any:
        """
        Get the largest face detected by InsightFace.

        Args:
            faces (List[Any]): List of detected faces.
            dim_max (int): Maximum dimension of the image.
            dim_min (int): Minimum dimension of the image.

        Returns:
            Any: The largest face object.
        """
        if len(faces) == 1:
            return faces[0]
        
        max_area = 0
        largest_face = None
        for face in faces:
            bbox = face.bbox
            area = (min(bbox[2], dim_max) - max(bbox[0], dim_min)) * (min(bbox[3], dim_max) - max(bbox[1], dim_min))
            if area > max_area:
                max_area = area
                largest_face = face
        return largest_face

    def get_largest_face_FR(self, faces: List[Tuple[int, int, int, int]], dim_max: int, dim_min: int) -> Tuple[int, int, int, int]:
        """
        Get the largest face detected by face_recognition.

        Args:
            faces (List[Tuple[int, int, int, int]]): List of detected faces.
            dim_max (int): Maximum dimension of the image.
            dim_min (int): Minimum dimension of the image.

        Returns:
            Tuple[int, int, int, int]: The largest face bounding box.
        """
        if len(faces) == 1:
            return faces[0]
        
        max_area = 0
        largest_face = None
        for face in faces:
            bbox = (face[-1],) + face[:-1]  # Convert to [left, top, right, bottom]
            area = (min(bbox[2], dim_max) - max(bbox[0], dim_min)) * (min(bbox[3], dim_max) - max(bbox[1], dim_min))
            if area > max_area:
                max_area = area
                largest_face = face
        return largest_face

def get_face_detector(gpu_id: int) -> FaceDetector:
    """
    Create and return a FaceDetector instance.

    Args:
        gpu_id (int): The ID of the GPU to use.

    Returns:
        FaceDetector: An instance of the FaceDetector class.
    """
    return FaceDetector(gpu_id)