# Inspired by https://github.com/huggingface/diffusers/blob/main/examples/community/stable_diffusion_mega.py
from __future__ import annotations

import gc
import os
import re
import json
import torch
import logging
import numpy as np

from PIL import Image
from typing import Union, List, Dict, Any, Optional, Callable
from contextlib import nullcontext, ExitStack

from huggingface_hub import hf_hub_download

from transformers import (
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DiffusionPipeline
)

from transformers.modeling_utils import no_init_weights
from diffusers.utils import is_accelerate_available, is_xformers_available

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

from aniportrait.utils import (
    PoseHelper,
    Video,
    adain_color_fix,
    dilate_erode,
    empty_cache,
    find_faces,
    gaussian_blur,
    get_data_dir,
    get_num_audio_samples,
    iterate_state_dict,
    rectify_image,
    remove_outlier_faces,
    scale_image,
    wavelet_color_fix,
)

from aniportrait.audio_models.audio2mesh import Audio2MeshModel
from aniportrait.models.unet_2d_condition import UNet2DConditionModel
from aniportrait.models.unet_3d_condition import UNet3DConditionModel
from aniportrait.models.pose_guider import PoseGuiderModel

from aniportrait.pipelines.pipeline_pose2img import Pose2ImagePipeline, Pose2ImagePipelineOutput
from aniportrait.pipelines.pipeline_pose2vid import Pose2VideoPipeline, Pose2VideoPipelineOutput
from aniportrait.pipelines.pipeline_pose2vid_long import Pose2LongVideoPipeline

logger = logging.getLogger(__name__)

__all__ = ["AniPortraitPipeline"]


class AniPortraitPipeline(DiffusionPipeline):
    """
    Combines all the pipelines into one and shares models.
    """
    vae_slicing: bool = False
    cpu_offload_gpu_id: Optional[int] = None

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuiderModel,
        audio_mesher: Audio2MeshModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler
        ]
    ) -> None:
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
            audio_mesher=audio_mesher
        )
        self.pose_helper = PoseHelper()

    @classmethod
    def from_single_file(
        cls,
        file_path_or_repository: str,
        filename: str="aniportrait.safetensors",
        config_filename: str="config.json",
        variant: Optional[str]=None,
        subfolder: Optional[str]=None,
        device: Optional[Union[str, torch.device]]=None,
        torch_dtype: Optional[torch.dtype]=None,
        cache_dir: Optional[str]=None,
    ) -> AniPortraitPipeline:
        """
        Loads the pipeline from a single file.
        """
        if variant is not None:
            filename, ext = os.path.splitext(filename)
            filename = f"{filename}.{variant}{ext}"

        if device is None:
            device = "cpu"
        else:
            device = str(device)

        if os.path.isdir(file_path_or_repository):
            model_dir = file_path_or_repository
            if subfolder:
                model_dir = os.path.join(model_dir, subfolder)
            file_path = os.path.join(model_dir, filename)
            config_path = os.path.join(model_dir, config_filename)
        elif os.path.isfile(file_path_or_repository):
            file_path = file_path_or_repository
            if os.path.isfile(config_filename):
                config_path = config_filename
            else:
                config_path = os.path.join(os.path.dirname(file_path), config_filename)
                if not os.path.exists(config_path) and subfolder:
                    config_path = os.path.join(os.path.dirname(file_path), subfolder, config_filename)
        elif re.search(r"^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$", file_path_or_repository):
            file_path = hf_hub_download(
                file_path_or_repository,
                filename,
                subfolder=subfolder,
                cache_dir=cache_dir,
            )
            try:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                )
            except:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    cache_dir=cache_dir,
                )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File {config_path} not found.")

        with open(config_path, "r") as f:
            aniportrait_config = json.load(f)

        # Create the scheduler
        scheduler = DDIMScheduler(**aniportrait_config["scheduler"])

        # Create the models
        context = ExitStack()
        if is_accelerate_available():
            context.enter_context(no_init_weights())
            context.enter_context(init_empty_weights())

        with context:
            # UNets
            reference_unet = UNet2DConditionModel.from_config(aniportrait_config["reference_unet"])
            denoising_unet = UNet3DConditionModel.from_config(aniportrait_config["denoising_unet"])

            # VAE
            vae = AutoencoderKL.from_config(aniportrait_config["vae"])

            # Image encoder
            image_encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(**aniportrait_config["image_encoder"]))

            # Pose Guider
            pose_guider = PoseGuiderModel.from_config(aniportrait_config["pose_guider"])

            # Audio Mesher
            audio_mesher = Audio2MeshModel.from_config(aniportrait_config["audio_mesher"])

        # Load the weights
        logger.debug("Models created, loading weights...")
        state_dicts = {}
        for key, value in iterate_state_dict(file_path):
            try:
                module, _, key = key.partition(".")
                if is_accelerate_available():
                    if module == "reference_unet":
                        set_module_tensor_to_device(reference_unet, key, device=device, value=value)
                    elif module == "denoising_unet":
                        set_module_tensor_to_device(denoising_unet, key, device=device, value=value)
                    elif module == "vae":
                        set_module_tensor_to_device(vae, key, device=device, value=value)
                    elif module == "image_encoder":
                        set_module_tensor_to_device(image_encoder, key, device=device, value=value)
                    elif module == "pose_guider":
                        set_module_tensor_to_device(pose_guider, key, device=device, value=value)
                    elif module == "audio2mesh":
                        set_module_tensor_to_device(audio_mesher, key, device=device, value=value)
                    else:
                        raise ValueError(f"Unknown module: {module}")
                else:
                    if module not in state_dicts:
                        state_dicts[module] = {}
                    state_dicts[module][key] = value
            except (AttributeError, KeyError, ValueError) as ex:
                logger.warning(f"Skipping module {module} key {key} due to {type(ex)}: {ex}")
        if not is_accelerate_available():
            try:
                reference_unet.load_state_dict(state_dicts["reference_unet"])
                denoising_unet.load_state_dict(state_dicts["denoising_unet"])
                vae.load_state_dict(state_dicts["vae"])
                image_encoder.load_state_dict(state_dicts["image_encoder"], strict=False)
                pose_guider.load_state_dict(state_dicts["pose_guider"])
                audio_mesher.load_state_dict(state_dicts["audio2mesh"])
                del state_dicts
                gc.collect()
            except KeyError as ex:
                raise RuntimeError(f"File did not provide a state dict for {ex}")

        # Create the pipeline
        pipeline = cls(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            audio_mesher=audio_mesher,
            scheduler=scheduler,
        )

        if torch_dtype is not None:
            pipeline.to(torch_dtype)

        pipeline.to(device)

        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("XFormers is not available, falling back to PyTorch attention")
        return pipeline

    @classmethod
    def from_pretrained(
        cls,
        *args: Any,
        **kwargs: Any
    ) -> AniPortraitPipeline:
        """
        Loads the pipeline from the pretrained model.
        """
        context = ExitStack()
        if is_accelerate_available():
            context.enter_context(no_init_weights()) # Prevent wav2vec2 zero-initialization
        with context:
            return super().from_pretrained(*args, **kwargs)

    def enable_vae_slicing(self) -> None:
        """
        Enables VAE slicing.
        """
        self.vae_slicing = True

    def disable_vae_slicing(self) -> None:
        """
        Disables VAE slicing.
        """
        self.vae_slicing = False

    def enable_model_cpu_offload(self, gpu_id: int=0) -> None:
        """
        Offloads the models to the CPU modelly.
        """
        self.cpu_offload_gpu_id = gpu_id

    def disable_model_cpu_offload(self):
        """
        Disables the model CPU offload.
        """
        self.cpu_offload_gpu_id = None

    def get_default_face_landmarks(self) -> np.ndarray:
        """
        Gets the default set of face landmarks.
        """
        return np.load(os.path.join(get_data_dir(), "face_landmarks.npy"))

    def get_default_translation_matrix(self) -> np.ndarray:
        """
        Gets the default translation matrix.
        """
        return np.load(os.path.join(get_data_dir(), "translation_matrix.npy"))

    def get_default_pose_sequence(self) -> np.ndarray:
        """
        Gets the default pose sequence.
        """
        return np.load(os.path.join(get_data_dir(), "pose_sequence.npy"))

    @torch.no_grad()
    def img2mask(
        self,
        reference_image: Image.Image,
        width: Optional[int]=None,
        height: Optional[int]=None,
        tile_size: int=512,
        tile_stride: int=256,
        dilate: Optional[int]=31,
        gaussian_kernel_size: Optional[int]=15,
    ) -> Image.Image:
        """
        Generates a mask image from a reference image.
        """
        image_width, image_height = reference_image.size
        if tile_size and tile_stride:
            if image_width < tile_size and image_height < tile_size:
                reference_image = rectify_image(reference_image, tile_size)
                detect_image_width, detect_image_height = reference_image.size
            else:
                detect_image_width, detect_image_height = image_width, image_height

            # Mediapipe likes square images, so we do it in tiles
            # this isn't perfect but it's good enough
            width_tiles = (detect_image_width - tile_size) // tile_stride + 1
            height_tiles = (detect_image_height - tile_size) // tile_stride + 1

            mask = Image.new("L", (detect_image_width, detect_image_height), 0)

            for i in range(width_tiles):
                for j in range(height_tiles):
                    x0 = i * tile_stride
                    y0 = j * tile_stride
                    x1 = min(detect_image_width, x0 + tile_size)
                    y1 = min(detect_image_height, y0 + tile_size)
                    tile_mask = self.pose_helper.image_to_mask(
                        reference_image.crop((x0, y0, x1, y1)),
                        width=x1-x0,
                        height=y1-y0
                    )
                    if tile_mask is not None:
                        mask.paste(
                            Image.new("L", tile_mask.size, (255,)),
                            (x0, y0),
                            mask=tile_mask
                        )
            mask = mask.resize((image_width, image_height))
        else:
            mask = self.pose_helper.image_to_mask(
                reference_image,
                width=width,
                height=height
            )

        if dilate:
            mask = dilate_erode(mask, dilate)
        if gaussian_kernel_size:
            mask = gaussian_blur(mask, gaussian_kernel_size)

        return mask

    @torch.no_grad()
    def vid2mask(
        self,
        reference_images: List[Image.Image],
        width: Optional[int]=None,
        height: Optional[int]=None,
        tile_size: int=512,
        tile_stride: int=256,
        dilate: Optional[int]=31,
        gaussian_kernel_size: Optional[int]=15,
        first_frame: bool=True,
        recenter_every_n_frames: int=5,
        face_tracking_distance_threshold: Optional[int]=30, # pixels
        face_padding: int=96 # pixels
    ) -> List[Image.Image]:
        """
        Generates a list of mask images from a list of reference images.
        """
        if first_frame:
            # Find faces in the first frame then track them
            first_frame = self.img2mask(
                reference_images[0],
                width=width,
                height=height,
                tile_size=tile_size,
                tile_stride=tile_stride,
                dilate=None,
                gaussian_kernel_size=None
            )

            # We paste using a mask in case any faces overlap in bounding box
            white = Image.new("L", first_frame.size, 255)
            frames = [first_frame] + [Image.new("L", first_frame.size, 0) for _ in range(len(reference_images) - 1)]
            faces = find_faces(first_frame)

            for x0, y0, x1, y1 in faces:
                # Scale image so face is `tile_size - face_padding * 2` pixels wide or tall
                face_width = x1 - x0
                face_height = y1 - y0
                target_face_size = tile_size - face_padding * 2
                scale = target_face_size / max(face_width, face_height)

                # Create tiled square about face
                start_x = int(max(0, (x0 + x1) / 2 - (tile_size / 2 * scale)))
                start_y = int(max(0, (y0 + y1) / 2 - (tile_size / 2 * scale)))
                end_x = int(min(first_frame.size[0], start_x + tile_size * scale))
                end_y = int(min(first_frame.size[1], start_y + tile_size * scale))
                found_masks = 0

                # Iterate through reference frames
                for i, reference_image in enumerate(reference_images[1:]):
                    # Crop and resize to detection size
                    ref = reference_image.crop((start_x, start_y, end_x, end_y))
                    ref = ref.resize((tile_size, tile_size))
                    mask = self.img2mask(
                        ref,
                        tile_size=None,
                        tile_stride=None,
                        dilate=None,
                        gaussian_kernel_size=None
                    )
                    if mask:
                        # Paste mask onto frame
                        mask = mask.resize((end_x-start_x, end_y-start_y))
                        found_masks += 1
                        frames[i+1].paste(
                            white.resize((end_x-start_x, end_y-start_y)),
                            (start_x, start_y),
                            mask=mask
                        )
                        # If we haven't recentered in <n> frames, do so now
                        if found_masks % recenter_every_n_frames == 0:
                            mask_faces = find_faces(mask)
                            if mask_faces:
                                mask_x0, mask_y0, mask_x1, mask_y1 = mask_faces[0]
                                mask_x0 += start_x
                                mask_y0 += start_y
                                mask_x1 += start_x
                                mask_y1 += start_y
                                start_x = max(0, (mask_x0 + mask_x1) // 2 - tile_size // 2)
                                start_y = max(0, (mask_y0 + mask_y1) // 2 - tile_size // 2)
                                end_x = min(first_frame.size[0], start_x + tile_size)
                                end_y = min(first_frame.size[1], start_y + tile_size)

            if dilate:
                frames = [dilate_erode(mask, dilate) for mask in frames]
            if gaussian_kernel_size:
                frames = [gaussian_blur(mask, gaussian_kernel_size) for mask in frames]
        else:
            frames = [
                self.img2mask(
                    reference_image,
                    width=width,
                    height=height,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                    dilate=dilate,
                    gaussian_kernel_size=gaussian_kernel_size
                )
                for reference_image in reference_images
            ]

        if face_tracking_distance_threshold:
            # Remove faces that appear/disappear between frames
            remove_outlier_faces(
                frames,
                distance_threshold=face_tracking_distance_threshold
            )

        return frames

    @torch.no_grad()
    def img2pose(
        self,
        reference_image: Image.Image,
        width: Optional[int]=None,
        height: Optional[int]=None
    ) -> Image.Image:
        """
        Generates a pose image from a reference image.
        """
        return self.pose_helper.image_to_pose(reference_image, width=width, height=height)

    @torch.no_grad()
    def vid2pose(
        self,
        reference_images: List[Image.Image],
        retarget_image: Optional[Image.Image]=None,
        width: Optional[int]=None,
        height: Optional[int]=None
    ) -> List[Image.Image]:
        """
        Generates a list of pose images from a list of reference images.
        """
        if retarget_image is not None:
            return self.pose_helper.images_to_pose_with_retarget(
                reference_images,
                retarget_image,
                width=width,
                height=height
            )
        return [
            self.img2pose(reference_image, width=width, height=height)
            for reference_image in reference_images
        ]

    @torch.no_grad()
    def audio2pose(
        self,
        audio_path: str,
        fps: int=30,
        reference_image: Optional[Image.Image]=None,
        pose_reference_images: Optional[List[Image.Image]]=None,
        width: Optional[int]=None,
        height: Optional[int]=None,
        include_mask: bool=False,
        leading_seconds_silence: float=0.0,
        trailing_seconds_silence: float=0.0,
        pitch_shift: float=0.0,
    ) -> Union[List[Image.Image], Tuple[List[Image.Image], List[Image.Image]]]:
        """
        Generates a pose image from an audio clip.
        """
        self.pose_helper.set_fps(fps)
        if reference_image is not None:
            image_width, image_height = reference_image.size
            if width is None:
                width = image_width
            if height is None:
                height = image_height
            landmarks = self.pose_helper.get_landmarks(reference_image)
            if not landmarks:
                raise ValueError("No face landmarks found in the reference image.")
            face_landmarks = landmarks["lmks3d"]
            translation_matrix = landmarks["trans_mat"]
        else:
            face_landmarks = self.get_default_face_landmarks()
            translation_matrix = self.get_default_translation_matrix()

        if pose_reference_images is not None:
            pose_sequence = self.pose_helper.images_to_pose_sequence(pose_reference_images)
        else:
            pose_sequence = self.get_default_pose_sequence()

        if width is None:
            width = 256
        if height is None:
            height = 256

        prediction = self.audio_mesher.infer_from_path(
            audio_path,
            fps=fps,
            leading_seconds_silence=leading_seconds_silence,
            trailing_seconds_silence=trailing_seconds_silence,
            pitch_shift=pitch_shift,
        )
        prediction = prediction.squeeze().detach().cpu().numpy()
        prediction = prediction.reshape(prediction.shape[0], -1, 3)

        pose_images = self.pose_helper.pose_sequence_to_images(
            prediction + face_landmarks,
            pose_sequence,
            translation_matrix,
            width,
            height
        )

        if include_mask:
            masks = self.pose_helper.pose_sequence_to_masks(
                prediction + face_landmarks,
                pose_sequence,
                translation_matrix,
                width,
                height
            )
            return pose_images, masks
        return pose_images

    @torch.no_grad()
    def pose2img(
        self,
        reference_image: Image.Image,
        pose_image: Image.Image,
        num_inference_steps: int,
        guidance_scale: float,
        eta: float=0.0,
        reference_pose_image: Optional[Image.Image]=None,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        width: Optional[int]=None,
        height: Optional[int]=None,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates an image from a reference image and pose image.
        """
        pipeline = Pose2ImagePipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler
        )

        if self.vae_slicing:
            pipeline.enable_vae_slicing()
        if self.cpu_offload_gpu_id is not None:
            pipeline.enable_model_cpu_offload(self.cpu_offload_gpu_id)

        pipeline.vae.to(device=self._execution_device)

        if reference_pose_image is None:
            reference_pose_image = self.img2pose(reference_image)

        image_width, image_height = reference_image.size
        if width is None:
            width = image_width
        if height is None:
            height = image_height

        result = pipeline(
            ref_image=reference_image,
            pose_image=pose_image,
            ref_pose_image=reference_pose_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generation=generation,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs
        )
        empty_cache()
        return result

    @torch.no_grad()
    def pose2vid(
        self,
        reference_image: Union[Image.Image, List[Image.Image]],
        pose_images: List[Image.Image],
        num_inference_steps: int,
        guidance_scale: float,
        eta: float=0.0,
        reference_pose_image: Optional[Image.Image]=None,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        width: Optional[int]=None,
        height: Optional[int]=None,
        video_length: Optional[int]=None,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates a video from a reference image and a list of pose images.
        """
        pipeline = Pose2VideoPipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler
        )

        if self.vae_slicing:
            pipeline.enable_vae_slicing()
        if self.cpu_offload_gpu_id is not None:
            pipeline.enable_model_cpu_offload(self.cpu_offload_gpu_id)

        pipeline.vae.to(device=self._execution_device)

        if reference_pose_image is None:
            reference_pose_image = self.img2pose(reference_image[0] if isinstance(reference_image, list) else reference_image)

        if isinstance(reference_image, list):
            image_width, image_height = reference_image[0].size
        else:
            image_width, image_height = reference_image.size

        if width is None:
            width = image_width
        if height is None:
            height = image_height
        if video_length is None:
            video_length = len(pose_images)

        result = pipeline(
            ref_image=reference_image,
            pose_images=pose_images,
            ref_pose_image=reference_pose_image,
            width=width,
            height=height,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generation=generation,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs
        )
        empty_cache()
        return result

    @torch.no_grad()
    def pose2vid_long(
        self,
        reference_image: Union[Image.Image, List[Image.Image]],
        pose_images: List[Image.Image],
        num_inference_steps: int,
        guidance_scale: float,
        eta: float=0.0,
        reference_pose_image: Optional[Image.Image]=None,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        context_schedule: str="uniform",
        context_frames: int=16,
        context_overlap: int=4,
        context_batch_size: int=1,
        interpolation_factor: int=1,
        width: Optional[int]=None,
        height: Optional[int]=None,
        video_length: Optional[int]=None,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates a long video from a reference image and a list of pose images.
        """
        pipeline = Pose2LongVideoPipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler
        )
        if self.vae_slicing:
            pipeline.enable_vae_slicing()
        if self.cpu_offload_gpu_id is not None:
            pipeline.enable_model_cpu_offload(self.cpu_offload_gpu_id)

        pipeline.vae.to(device=self._execution_device)

        if reference_pose_image is None:
            reference_pose_image = self.img2pose(reference_image[0] if isinstance(reference_image, list) else reference_image)

        if isinstance(reference_image, list):
            image_width, image_height = reference_image[0].size
        else:
            image_width, image_height = reference_image.size

        if width is None:
            width = image_width
        if height is None:
            height = image_height
        if video_length is None:
            video_length = len(pose_images)

        result = pipeline(
            ref_image=reference_image,
            pose_images=pose_images,
            ref_pose_image=reference_pose_image,
            width=width,
            height=height,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generation=generation,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            context_schedule=context_schedule,
            context_frames=context_frames,
            context_overlap=context_overlap,
            context_batch_size=context_batch_size,
            interpolation_factor=interpolation_factor,
            **kwargs
        )
        empty_cache()
        return result

    @torch.no_grad()
    def img2img(
        self,
        reference_image: Image.Image,
        pose_reference_image: Image.Image,
        num_inference_steps: int,
        guidance_scale: float,
        eta: float=0.0,
        reference_pose_image: Optional[Image.Image]=None,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        width: Optional[int]=None,
        height: Optional[int]=None,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates an image from a reference image and pose image.
        """
        pipeline = Pose2ImagePipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler
        )

        if self.vae_slicing:
            pipeline.enable_vae_slicing()
        if self.cpu_offload_gpu_id is not None:
            pipeline.enable_sequential_cpu_offload(self.cpu_offload_gpu_id)

        pipeline.vae.to(device=self._execution_device)

        pose_image = self.img2pose(pose_reference_image)
        if reference_pose_image is None:
            reference_pose_image = self.img2pose(reference_image)

        image_width, image_height = reference_image.size
        if width is None:
            width = image_width
        if height is None:
            height = image_height

        result = pipeline(
            ref_image=reference_image,
            pose_image=pose_image,
            ref_pose_image=reference_pose_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generation=generation,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs
        )
        empty_cache()
        return result

    @torch.no_grad()
    def audio2vid(
        self,
        audio: str,
        reference_image: Image.Image,
        num_inference_steps: int=25,
        guidance_scale: float=3.5,
        fps: int=30,
        eta: float=0.0,
        reference_pose_image: Optional[Image.Image]=None,
        pose_reference_images: Optional[List[Image.Image]]=None,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        context_schedule: str="uniform",
        context_frames: int=16,
        context_overlap: int=4,
        context_batch_size: int=1,
        interpolation_factor: int=1,
        width: Optional[int]=None,
        height: Optional[int]=None,
        video_length: Optional[int]=None,
        use_long_video: bool=True,
        pose_filename: Optional[str]=None,
        leading_seconds_silence: float=0.0,
        trailing_seconds_silence: float=0.0,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates a video from an audio clip.
        """
        pose_images = self.audio2pose(
            audio,
            fps=fps,
            reference_image=reference_image,
            pose_reference_images=pose_reference_images,
            leading_seconds_silence=leading_seconds_silence,
            trailing_seconds_silence=trailing_seconds_silence
        )

        if pose_filename:
            # Save pose video for debugging/information
            Video(pose_images).save(pose_filename, rate=fps, overwrite=True)

        if use_long_video and len(pose_images) > 16 and (video_length is None or video_length > 16):
            return self.pose2vid_long(
                reference_image=reference_image,
                pose_images=pose_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                reference_pose_image=reference_pose_image,
                generation=generation,
                output_type=output_type,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
                context_schedule=context_schedule,
                context_frames=context_frames,
                context_overlap=context_overlap,
                context_batch_size=context_batch_size,
                interpolation_factor=interpolation_factor,
                width=width,
                height=height,
                video_length=video_length,
                **kwargs
            )

        return self.pose2vid(
            reference_image=reference_image,
            pose_images=pose_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            reference_pose_image=reference_pose_image,
            generation=generation,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            width=width,
            height=height,
            video_length=video_length,
            **kwargs
        )

    @torch.no_grad()
    def vid2vid(
        self,
        reference_image: Image.Image,
        pose_reference_images: List[Image.Image],
        reference_pose_image: Optional[Image.Image]=None,
        num_inference_steps: int=25,
        guidance_scale: float=3.5,
        eta: float=0.0,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        context_schedule: str="uniform",
        context_frames: int=16,
        context_overlap: int=4,
        context_batch_size: int=1,
        interpolation_factor: int=1,
        width: Optional[int]=None,
        height: Optional[int]=None,
        video_length: Optional[int]=None,
        use_long_video: bool=True,
        pose_filename: Optional[str]=None,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates a video from an audio clip.
        """
        image_width, image_height = reference_image.size
        if width is None:
            width = image_width
        if height is None:
            height = image_height

        pose_images = self.vid2pose(
            pose_reference_images,
            retarget_image=reference_image,
            width=width,
            height=height
        )
        if pose_filename:
            # Save pose video for debugging/information
            Video(pose_images).save(pose_filename, rate=kwargs.get("fps", 30), overwrite=True)

        if reference_pose_image is None:
            reference_pose_image = self.img2pose(reference_image, width=width, height=height)

        if use_long_video and len(pose_images) > 16 and (video_length is None or video_length > 16):
            return self.pose2vid_long(
                reference_image=reference_image,
                pose_images=pose_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                reference_pose_image=reference_pose_image,
                generation=generation,
                output_type=output_type,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
                context_schedule=context_schedule,
                context_frames=context_frames,
                context_overlap=context_overlap,
                context_batch_size=context_batch_size,
                interpolation_factor=interpolation_factor,
                width=width,
                height=height,
                video_length=video_length,
                **kwargs
            )

        return self.pose2vid(
            reference_image=reference_image,
            pose_images=pose_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            reference_pose_image=reference_pose_image,
            generation=generation,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            width=width,
            height=height,
            video_length=video_length,
            **kwargs
        )

    @classmethod
    def composite_images(
        cls,
        background: Image.Image,
        pose: Image.Image,
        mask: Optional[Image.Image]=None,
        mask_opacity: int=16
    ) -> Image.Image:
        """
        Composites a pose image onto a background image, optionally with a mask.
        """
        alpha = Image.new("L", background.size, 128)
        image = background.copy()
        if mask is not None:
            image.paste(mask, (0, 0), mask=alpha)
        image.paste(pose, (0, 0), mask=alpha)
        return image

    @torch.no_grad()
    def audiovid2vid(
        self,
        audio: str,
        reference_image: List[Image.Image],
        num_inference_steps: int=25,
        guidance_scale: float=3.5,
        fps: int=30,
        eta: float=0.0,
        reference_pose_image: Optional[Image.Image]=None,
        pose_reference_images: Optional[List[Image.Image]]=None,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        context_schedule: str="uniform",
        context_frames: int=16,
        context_overlap: int=4,
        context_batch_size: int=1,
        interpolation_factor: int=1,
        width: Optional[int]=None,
        height: Optional[int]=None,
        video_length: Optional[int]=None,
        use_long_video: bool=True,
        paste_back: bool=False,
        paste_back_color_fix: Optional[Literal["wavelet", "adain"]]="wavelet",
        crop_to_face: bool=False,
        crop_to_face_target_size: Optional[int]=512,
        crop_to_face_padding: Optional[int]=64,
        mask_filename: Optional[str]=None,
        pose_filename: Optional[str]=None,
        combined_filename: Optional[str]=None,
        mask_dilate: Optional[int]=31,
        mask_gaussian_kernel_size: Optional[int]=15,
        mask_first_frame: bool=True,
        leading_seconds_silence: float=0.0,
        trailing_seconds_silence: float=0.0,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates a video from an audio clip and a video clip (lipsync).
        """
        paste_back = paste_back and output_type == "pil"
        paste_back_bbox = None
        inference_scale = 1.0

        # Standardize number of frames so we don't do extra work
        if video_length is None:
            video_length = get_num_audio_samples(audio, fps=fps) + int((leading_seconds_silence + trailing_seconds_silence) * fps)

        # Slice image arrays to standard length
        reference_image = reference_image[:video_length]
        if pose_reference_images is not None:
            pose_reference_images = pose_reference_images[:video_length]

        # Copy background images for pasting later
        background_image = [image.copy() for image in reference_image]
        if crop_to_face or paste_back:
            # Create masks to crop to face and/or paste
            mask_images = self.vid2mask(
                reference_image,
                width=width,
                height=height,
                dilate=mask_dilate,
                gaussian_kernel_size=mask_gaussian_kernel_size,
                first_frame=mask_first_frame
            )
            if mask_filename:
                # Save mask video for debugging/information
                Video(mask_images).save(mask_filename, rate=fps, overwrite=True)

            # Find bounding box for cropping/pasting
            mask_width, mask_height = mask_images[0].size
            (x0, y0, x1, y1) = (None, None, None, None)
            for i, mask_image in enumerate(mask_images):
                bbox = mask_image.getbbox()
                if bbox is None:
                    continue
                if x0 is None:
                    x0, y0, x1, y1 = bbox
                else:
                    bx0, by0, bx1, by1 = bbox
                    x0 = min(x0, bx0)
                    y0 = min(y0, by0)
                    x1 = max(x1, bx1)
                    y1 = max(y1, by1)

            if x0 is None or y0 is None or x1 is None or y1 is None:
                raise ValueError("No faces found in video.")

            # Scale image so face is `tile_size - face_padding * 2` pixels wide or tall
            face_width = x1 - x0
            face_height = y1 - y0
            target_face_size = crop_to_face_target_size - crop_to_face_padding * 2
            inference_scale = target_face_size / max(face_width, face_height)

            # Create tiled square about face
            start_x = int(max(0, (x0 + x1) / 2 - ((crop_to_face_target_size / 2) / inference_scale)))
            start_y = int(max(0, (y0 + y1) / 2 - ((crop_to_face_target_size / 2) / inference_scale)))
            end_x = int(min(mask_images[0].size[0], start_x + crop_to_face_target_size / inference_scale))
            end_y = int(min(mask_images[0].size[1], start_y + crop_to_face_target_size / inference_scale))
            paste_back_bbox = (start_x, start_y, end_x, end_y)

            # Crop all images
            reference_image = [image.crop(paste_back_bbox) for image in reference_image]
            mask_images = [image.crop(paste_back_bbox) for image in mask_images]
            if pose_reference_images is not None:
                pose_reference_images = [image.crop(paste_back_bbox) for image in pose_reference_images]

            # Scale all images
            if inference_scale != 1.0:
                reference_image = [
                    image.resize((crop_to_face_target_size, crop_to_face_target_size))
                    for image in reference_image
                ]
                mask_images = [
                    image.resize((crop_to_face_target_size, crop_to_face_target_size))
                    for image in mask_images
                ]
                if pose_reference_images is not None:
                    pose_reference_images = [
                        image.resize((crop_to_face_target_size, crop_to_face_target_size))
                        for image in pose_reference_images
                    ]

        # Generate pose images using audio and references
        pose_images = self.audio2pose(
            audio,
            fps=fps,
            reference_image=reference_image[0],
            pose_reference_images=pose_reference_images,
            leading_seconds_silence=leading_seconds_silence,
            trailing_seconds_silence=trailing_seconds_silence
        )

        # We could be one-off due to rounding, ensure everything is correct here
        video_length = min(video_length, len(pose_images), len(reference_image))
        background_image = background_image[:video_length]
        pose_images = pose_images[:video_length]
        reference_image = reference_image[:video_length]
        if pose_reference_images is not None:
            pose_reference_images = pose_reference_images[:video_length]

        if pose_filename:
            # Save pose video for debugging/information
            Video(pose_images).save(pose_filename, rate=fps, overwrite=True)

        if combined_filename:
            # Save combined video for debugging/information
            combined_images = []
            for i, background in enumerate(background_image[:len(pose_images)]):
                pose = pose_images[i]
                mask = None
                if crop_to_face or paste_back:
                    mask = mask_images[i]
                    if crop_to_face:
                        x1, y1, x2, y2 = paste_back_bbox
                        mask_image = Image.new("L", background.size, 0)
                        mask_image.paste(mask.resize((x2 - x1, y2 - y1)), paste_back_bbox)
                        mask = mask_image
                        pose_image = Image.new("RGB", background.size, (0, 0, 0))
                        pose_image.paste(pose_images[i].resize((x2 - x1, y2 - y1)), paste_back_bbox)
                        pose = pose_image
                combined_images.append(self.composite_images(background, pose, mask))
            Video(combined_images).save(combined_filename, rate=fps, overwrite=True)

        if use_long_video and len(pose_images) > 16 and (video_length is None or video_length > 16):
            result = self.pose2vid_long(
                reference_image=reference_image,
                pose_images=pose_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                reference_pose_image=reference_pose_image,
                generation=generation,
                output_type=output_type,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
                context_schedule=context_schedule,
                context_frames=context_frames,
                context_overlap=context_overlap,
                context_batch_size=context_batch_size,
                interpolation_factor=interpolation_factor,
                width=width,
                height=height,
                video_length=video_length,
                **kwargs
            )
        else:
            result = self.pose2vid(
                reference_image=reference_image,
                pose_images=pose_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                reference_pose_image=reference_pose_image,
                generation=generation,
                output_type=output_type,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
                width=width,
                height=height,
                video_length=video_length,
                **kwargs
            )

        result_length = len(result.videos)

        if paste_back or crop_to_face:
            # Paste back images
            result_length = min(result_length, len(background_image), len(reference_image))

            result.videos = result.videos[:result_length]
            mask_images = mask_images[:result_length]
            background_image = background_image[:result_length]
            reference_image = reference_image[:result_length]

            for i, (reference, background, frame, mask) in enumerate(zip(reference_image, background_image, result.videos, mask_images)):
                reference = reference.resize(frame.size)
                if paste_back_color_fix == "wavelet":
                    frame = wavelet_color_fix(frame, reference)
                elif paste_back_color_fix == "adain":
                    frame = adain_color_fix(frame, reference)
                if crop_to_face:
                    x0, y0, x1, y1 = paste_back_bbox
                    actual_width = x1 - x0
                    actual_height = y1 - y0
                    frame = frame.resize((actual_width, actual_height))
                    mask = mask.resize((actual_width, actual_height))
                    background.paste(frame, paste_back_bbox, mask=mask)
                    result.videos[i] = background
                else:
                    reference.paste(frame.resize(mask.size), (0, 0), mask=mask)
                    result.videos[i] = reference
        return result
