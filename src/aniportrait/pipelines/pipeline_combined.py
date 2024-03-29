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

from aniportrait.utils import PoseHelper, get_data_dir, iterate_state_dict, empty_cache

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

    def enable_sequential_cpu_offload(self, gpu_id: int=0) -> None:
        """
        Offloads the models to the CPU sequentially.
        """
        self.cpu_offload_gpu_id = gpu_id

    def disable_sequential_cpu_offload(self):
        """
        Disables the sequential CPU offload.
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
        height: Optional[int]=None
    ) -> List[Image.Image]:
        """
        Generates a pose image from an audio clip.
        """
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

        prediction = self.audio_mesher.infer_from_path(audio_path, fps=fps)
        prediction = prediction.squeeze().detach().cpu().numpy()
        prediction = prediction.reshape(prediction.shape[0], -1, 3)

        return self.pose_helper.pose_sequence_to_images(
            prediction + face_landmarks,
            pose_sequence,
            translation_matrix,
            width,
            height
        )

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
            pipeline.enable_sequential_cpu_offload(self.cpu_offload_gpu_id)

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
        reference_image: Image.Image,
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
            pipeline.enable_sequential_cpu_offload(self.cpu_offload_gpu_id)

        if reference_pose_image is None:
            reference_pose_image = self.img2pose(reference_image)

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
        reference_image: Image.Image,
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
            pipeline.enable_sequential_cpu_offload(self.cpu_offload_gpu_id)

        if reference_pose_image is None:
            reference_pose_image = self.img2pose(reference_image)

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
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates a video from an audio clip.
        """
        pose_images = self.audio2pose(
            audio,
            fps=fps,
            reference_image=reference_image,
            pose_reference_images=pose_reference_images
        )

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
