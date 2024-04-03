from __future__ import annotations
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision

from einops import rearrange
from PIL import Image

from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "blur",
    "dilate_erode",
    "find_faces",
    "gaussian_blur",
    "get_data_dir",
    "get_frame_rate",
    "get_num_audio_samples",
    "human_size",
    "latent_friendly_image",
    "rectify_image",
    "reiterator",
    "remove_outlier_faces",
    "scale_image",
    "seed_everything",
    "track_faces",
]

def human_size(num_bytes: int) -> str:
    """
    Convert a number of bytes to a human-readable string
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} YB"

def latent_friendly_image(
    image: Union[List[Image.Image], Image.Image],
    nearest: int=8,
    resample=Image.NEAREST
) -> Union[List[Image.Image], Image.Image]:
    """
    Resize an image or list of images to be friendly to latent space optimization
    """
    if isinstance(image, list):
        return [latent_friendly_image(img, nearest) for img in image]
    width, height = image.size
    new_width = (width // nearest) * nearest
    new_height = (height // nearest) * nearest
    image = image.resize((new_width, new_height), resample=resample)
    return image

def scale_image(
    image: Union[List[Image.Image], Image.Image],
    scale: float=1.0,
    resample=Image.LANCZOS
) -> Union[List[Image.Image], Image.Image]:
    """
    Scale an image or list of images
    """
    if scale == 1.0:
        return image
    if isinstance(image, list):
        return [scale_image(img, scale) for img in image]
    width, height = image.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), resample=resample)
    return image

def dilate_erode(
    image: Union[Image, List[Image]],
    value: int
) -> Union[Image, List[Image]]:
    """
    Given an image, dilate or erode it.
    Values of >0 dilate, <0 erode. 0 Does nothing.
    :see: http://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    """
    if value == 0:
        return image
    if isinstance(image, list):
        return [
            dilate_erode(img, value)
            for img in image
        ]

    from PIL import Image
    import cv2
    import numpy as np

    arr = np.array(image.convert("L"))
    transform = cv2.dilate if value > 0 else cv2.erode
    value = abs(value)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    arr = transform(arr, kernel, iterations=1)
    return Image.fromarray(arr)

def blur(
    image: Union[Image, List[Image]],
    kernel_size: int
) -> Union[Image, List[Image]]:
    """
    Given an image, blur it.
    """
    if kernel_size == 0:
        return image
    if isinstance(image, list):
        return [
            blur(img, kernel_size)
            for img in image
        ]

    from PIL import Image
    import cv2
    import numpy as np
    arr = np.array(image)
    arr = cv2.blur(arr, (kernel_size, kernel_size))
    return Image.fromarray(arr)

def gaussian_blur(
    image: Union[Image, List[Image]],
    kernel_size: int
) -> Union[Image, List[Image]]:
    """
    Given an image, blur it with a Gaussian kernel.
    """
    if kernel_size == 0:
        return image
    if isinstance(image, list):
        return [
            gaussian_blur(img, kernel_size)
            for img in image
        ]

    from PIL import Image
    import cv2
    import numpy as np
    arr = np.array(image)
    arr = cv2.GaussianBlur(arr, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    return Image.fromarray(arr)

def rectify_image(
    image: Union[List[Image.Image], Image.Image],
    size: int,
    resample=Image.LANCZOS,
    method: Literal["smallest", "largest"]="largest"
) -> Union[List[Image.Image], Image.Image]:
    """
    Scale an image or list of images
    """
    if isinstance(image, list):
        return [rectify_image(img, size) for img in image]

    width, height = image.size

    if width > height and method == "largest":
        new_width = size
        new_height = int(size * height / width)
    else:
        new_width = int(size * width / height)
        new_height = size
    image = image.resize((new_width, new_height), resample=resample)
    return image

def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

def get_data_dir() -> str:
    return osp.realpath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "data"))

class reiterator:
    """
    Transparently memoize any iterator
    """
    memoized: List[Any]

    def __init__(self, iterable: Iterable[Any]) -> None:
        self.iterable = iterable
        self.memoized = []
        self.started = False
        self.finished = False

    def __iter__(self) -> Iterable[Any]:
        if not self.started:
            self.started = True
            last_index: Optional[int] = None
            for i, value in enumerate(self.iterable):
                yield value
                self.memoized.append(value)
                last_index = i
                if self.finished:
                    # Completed somewhere else
                    break
            if self.finished:
                if last_index is None:
                    last_index = 0
                for value in self.memoized[last_index+1:]:
                    yield value
            self.finished = True
            del self.iterable
        elif not self.finished:
            # Complete iterator
            self.memoized += [item for item in self.iterable]
            self.finished = True
            del self.iterable
            for item in self.memoized:
                yield item
        else:
            for item in self.memoized:
                yield item

import numpy as np
from scipy.ndimage import label, find_objects
from scipy.spatial.distance import cdist

def is_outlier(track, total_frames):
    """
    Determine if a face track is outlier based on visibility in consecutive frames.
    """
    for frame_idx, _ in track:
        print(f"FRAME IDX {frame_idx}")
    return False

def find_faces(mask: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Find faces in a binary mask image.
    """
    labeled_array, num_features = label(mask)
    face_regions = find_objects(labeled_array)
    return [
        (region[1].start, region[0].start, region[1].stop, region[0].stop) # l, t, r, b
        for region in face_regions
    ]

def track_faces(
    masks: List[Image.Image],
    distance_threshold: int=30
) -> List[List[Tuple[int, Tuple[int, int, int, int]]]]:
    """
    Track faces in a series of binary mask images.
    """
    face_tracks = []
    total_frames = len(masks)

    for frame_idx, frame in enumerate(masks):
        labeled_array, num_features = label(frame)
        face_regions = find_objects(labeled_array)

        if not face_tracks:
            face_tracks = [[(frame_idx, region)] for region in face_regions]
            continue
        if not face_regions:
            continue

        current_face_centers = [((region[0].start + region[0].stop) / 2, (region[1].start + region[1].stop) / 2) for region in face_regions]
        previous_face_centers = [((track[-1][1][0].start + track[-1][1][0].stop) / 2, (track[-1][1][1].start + track[-1][1][1].stop) / 2) for track in face_tracks if track]
        
        if previous_face_centers:
            distance_matrix = cdist(previous_face_centers, current_face_centers)
            matched_current_faces = set()
            for i, track in enumerate(face_tracks):
                if track:
                    closest_face_idx = np.argmin(distance_matrix[i])
                    if distance_matrix[i][closest_face_idx] < distance_threshold and closest_face_idx not in matched_current_faces:
                        track.append((frame_idx, face_regions[closest_face_idx]))
                        matched_current_faces.add(closest_face_idx)
            
            # Start new tracks for unmatched faces
            for idx, region in enumerate(face_regions):
                if idx not in matched_current_faces:
                    face_tracks.append([(frame_idx, region)])

    return [
        [
            (frame_idx, (region[1].start, region[0].start, region[1].stop, region[0].stop)) # l, t, r, b
            for frame_idx, region in track
        ]
        for track in face_tracks
    ]

def is_outlier(track: List[Tuple[int, Tuple[int, int, int, int]]]) -> bool:
    """
    Determine if a face track is outlier based on visibility in consecutive frames.
    """
    min_idx = min(track, key=lambda x: x[0])[0]
    max_idx = max(track, key=lambda x: x[0])[0]

    total_frames = max_idx - min_idx + 1
    visible_frames = len(track)

    return visible_frames < 16 or visible_frames / total_frames < 0.5

def remove_outlier_faces(frames: List[Image.Image], distance_threshold: int=30) -> None:
    """
    Removes outlier faces from a series of binary mask images, allowing for faces that appear partway.
    """
    faces = track_faces(frames, distance_threshold)
    for track in faces:
        if is_outlier(track):
            for frame_idx, region in track:
                # Set the outlier face region to black
                left, top, right, bottom = region
                frames[frame_idx].paste(
                    Image.new("L", (right - left, bottom - top), 0),
                    (left, top)
                )

def get_num_audio_samples(
    audio_path: str,
    sample_rate: int=16000,
    fps: int=30
) -> int:
    """
    Get the number of audio samples in a given audio file.
    """
    import math
    import audioread
    with audioread.audio_open(audio_path) as f:
        total_samples = int(f.duration * f.samplerate)
        adjusted_samples = total_samples*(sample_rate/f.samplerate)
        return math.ceil(adjusted_samples/sample_rate*fps)

def get_frame_rate(video_path: str) -> float:
    """
    Returns the frame rate of the given video file using moviepy.
    
    Parameters:
    video_path (str): The path to the video file.
    
    Returns:
    float: The frame rate of the video.
    """
    from moviepy.editor import VideoFileClip
    with VideoFileClip(video_path) as video:
        return video.fps  # fps stands for frames per second
