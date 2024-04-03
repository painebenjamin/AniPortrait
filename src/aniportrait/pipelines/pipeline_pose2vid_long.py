from __future__ import annotations
# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import math

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, deprecate, is_accelerate_available, logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from transformers import CLIPImageProcessor

from aniportrait.models.mutual_self_attention import ReferenceAttentionControl
from aniportrait.pipelines.pipeline_pose2vid import Pose2VideoPipelineOutput
from aniportrait.utils import get_tensor_interpolation_method, get_context_scheduler

class Pose2LongVideoPipeline(DiffusionPipeline):
    _optional_components = []
    _exclude_from_cpu_offload = ["pose_guider"]

    model_cpu_offload_seq = "image_encoder->reference_unet->denoising_unet->vae"

    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        pose_guider,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ]
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=True,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0]), desc="Decoding Latents"):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    def images_from_video(
        self,
        video: torch.Tensor,
        rescale: bool=False
    ) -> List[Image.Image]:
        """
        Convert a video tensor to a list of PIL images
        """
        import numpy as np
        import torchvision
        from einops import rearrange
        video = rearrange(video, "b c t h w -> t b c h w")
        height, width = video.shape[-2:]
        outputs = []

        for x in video:
            x = torchvision.utils.make_grid(x, nrow=1)  # (c h w)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            x = Image.fromarray(x)
            outputs.append(x)

        return outputs

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        pose_images,
        ref_pose_image,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float=0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=1,
        context_schedule="uniform",
        context_frames=16,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        clip_processing_size=224,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        animated_reference = isinstance(ref_image, list)
        if animated_reference:
            ref_image = ref_image[:video_length]
            num_ref_images = len(ref_image)
            if num_ref_images < video_length:
                # Reflect the animation - first add mirror
                ref_image += [ref_image[i].copy() for i in range(num_ref_images - 2, 0, -1)]
                while len(ref_image) < video_length:
                    ref_image += [image.copy() for image in ref_image]
                ref_image = ref_image[:video_length]

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
            animated_reference=animated_reference,
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            animated_reference=animated_reference,
        )

        num_channels_latents = self.denoising_unet.config.in_channels

        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        if animated_reference:
            encoder_hidden_states = torch.cat([
                self.image_encoder(
                    self.clip_image_processor.preprocess(image.resize((224, 224)), return_tensors="pt").pixel_values.to(device, dtype=self.image_encoder.dtype)
                ).image_embeds.unsqueeze(1)
                for image in tqdm(ref_image, desc="Encoding Reference Images (CLIP)")
            ], dim=1) # (1, f, c, l)
            latent_shape = (
                video_length,
                num_channels_latents,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            ref_image_latents = torch.zeros(
                latent_shape,
                device=device,
                dtype=self.vae.dtype
            )
            num_ref_images = len(ref_image)
            for i, ref_image_frame in tqdm(list(enumerate(ref_image)), desc="Encoding Reference Images (VAE)"):
                ref_image_tensor = self.ref_image_processor.preprocess(
                    ref_image_frame, height=height, width=width
                )
                ref_image_tensor = ref_image_tensor.to(
                    dtype=self.vae.dtype, device=device
                )
                ref_image_latents[i] = self.vae.encode(ref_image_tensor).latent_dist.mean
            if num_ref_images < video_length:
                ref_image_latents[num_ref_images:] = ref_image_latents[num_ref_images - 1].unsqueeze(0).repeat(video_length - num_ref_images, 1, 1, 1)
            ref_image_latents = ref_image_latents.unsqueeze(1) * 0.18215  # (f, b, 4, h, w)
        else:
            # Prepare clip image embeds
            clip_image = self.clip_image_processor.preprocess(
                (ref_image[0] if isinstance(ref_image, list) else ref_image).resize(
                    (clip_processing_size, clip_processing_size)
                ), return_tensors="pt"
            ).pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(device, dtype=self.image_encoder.dtype)
            ).image_embeds
            ref_image_tensor = self.ref_image_processor.preprocess(
                ref_image, height=height, width=width
            )  # (bs, c, width, height)
            ref_image_tensor = ref_image_tensor.to(
                dtype=self.vae.dtype, device=device
            )
            ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
            ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
            encoder_hidden_states = clip_image_embeds.unsqueeze(1)

        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            self.image_encoder.dtype,
            device,
            generator,
        )

        # Prepare a list of pose condition images
        pose_cond_tensor_list = []
        for pose_image in pose_images:
            pose_cond_tensor = self.cond_image_processor.preprocess(
                pose_image, height=height, width=width
            )
            pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
            pose_cond_tensor_list.append(pose_cond_tensor)

        pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=2)  # (bs, c, t, h, w)
        
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        
        ref_pose_tensor = self.cond_image_processor.preprocess(
            ref_pose_image, height=height, width=width
        )
        ref_pose_tensor = ref_pose_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )

        # Reference writer
        for ref_latents in tqdm([ref_image_latents] if not animated_reference else ref_image_latents, desc="Forwarding Reference Images"):
            self.reference_unet(
                ref_latents.repeat(
                    (2 if do_classifier_free_guidance else 1), 1, 1, 1
                ),
                torch.zeros_like(timesteps[0]),
                # t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )

        reference_control_reader.update(reference_control_writer)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        context_scheduler = get_context_scheduler(context_schedule)
        for i, t in tqdm(list(enumerate(timesteps)), desc="Sampling"):
            noise_pred = torch.zeros(
                (
                    latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                    *latents.shape[1:],
                ),
                device=latents.device,
                dtype=latents.dtype,
            )
            counter = torch.zeros(
                (1, 1, latents.shape[2], 1, 1),
                device=latents.device,
                dtype=latents.dtype,
            )

            context_queue = list(
                context_scheduler(
                    0,
                    num_inference_steps,
                    latents.shape[2],
                    context_frames,
                    context_stride,
                    0,
                )
            )
            num_context_batches = math.ceil(len(context_queue) / context_batch_size)

            context_queue = list(
                context_scheduler(
                    0,
                    num_inference_steps,
                    latents.shape[2],
                    context_frames,
                    context_stride,
                    context_overlap,
                )
            )

            num_context_batches = math.ceil(len(context_queue) / context_batch_size)
            global_context = []
            for i in range(num_context_batches):
                global_context.append(
                    context_queue[
                        i * context_batch_size : (i + 1) * context_batch_size
                    ]
                )

            for context in global_context:
                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents[:, :, c] for c in context])
                    .to(device)
                    .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                b, c, f, h, w = latent_model_input.shape
                
                pose_cond_input = (
                    torch.cat([pose_cond_tensor[:, :, c] for c in context])
                    .to(device)
                    .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                )
                pose_fea = self.pose_guider(pose_cond_input, ref_pose_tensor)

                pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states[:b, context].mean(1) if animated_reference else encoder_hidden_states[:b],
                    pose_cond_fea=pose_fea,
                    context_frames=context,
                    return_dict=False,
                )[0]

                for j, c in enumerate(context):
                    noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                    counter[:, :, c] = counter[:, :, c] + 1

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        reference_control_reader.clear()
        reference_control_writer.clear()

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)

        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type not in ["np", "numpy"]:
            images = torch.from_numpy(images)
            if output_type == "pil":
                images = self.images_from_video(images)

        # Free model hooks
        self.maybe_free_model_hooks()

        if not return_dict:
            return images

        return Pose2VideoPipelineOutput(videos=images)
