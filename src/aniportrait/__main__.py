from __future__ import annotations

import os
import sys
import click
import traceback

from PIL import Image

from typing import Optional

@click.command()
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False))
@click.option("--video", "-v", type=click.Path(exists=True, dir_okay=False), help="Video file to drive the animation.")
@click.option("--audio", "-a", type=click.Path(exists=True, dir_okay=False), help="Audio file to drive the animation.")
@click.option("--frame-rate", "-fps", type=int, default=None, help="Video FPS. Also controls the sampling rate of the audio. Will default to the video FPS if a video file is provided, or 30 if not.", show_default=True)
@click.option("--guidance-scale", "-cfg", type=float, default=3.5, help="Guidance scale for the diffusion process.", show_default=True)
@click.option("--num-inference-steps", "-ns", type=int, default=20, help="Number of diffusion steps.", show_default=True)
@click.option("--context-frames", "-cf", type=int, default=16, help="Number of context frames to use.", show_default=True)
@click.option("--context-overlap", "-co", type=int, default=4, help="Number of context frames to overlap.", show_default=True)
@click.option("--num-frames", "-nf", type=int, default=None, help="An explicit number of frames to use. When not passed, use the length of the audio or video")
@click.option("--seed", "-s", type=int, default=None, help="Random seed.")
@click.option("--width", "-w", type=int, default=None, help="Output video width. Defaults to the input image width.")
@click.option("--height", "-h", type=int, default=None, help="Output video height. Defaults to the input image height.")
@click.option("--model", "-m", type=str, default="benjamin-paine/aniportrait", help="HuggingFace model name.")
@click.option("--no-half", "-nh", is_flag=True, default=False, help="Do not use half precision.", show_default=True)
@click.option("--no-offload", "-no", is_flag=True, default=False, help="Do not offload to the CPU to preserve GPU memory.", show_default=True)
@click.option("--gpu-id", "-g", type=int, default=0, help="GPU ID to use.")
@click.option("--model-single-file", "-sf", is_flag=True, default=False, help="Download and use a single file instead of a directory.")
@click.option("--config-file", "-cf", type=str, default="config.json", help="Config file to use when using the model-single-file option. Accepts a path or a filename in the same directory as the single file. Will download from the repository passed in the model option if not provided.", show_default=True)
@click.option("--model-filename", "-mf", type=str, default="aniportrait.safetensors", help="The model file to download when using the model-single-file option.", show_default=True)
@click.option("--remote-subfolder", "-rs", type=str, default=None, help="Remote subfolder to download from when using the model-single-file option.")
@click.option("--cache-dir", "-cd", type=click.Path(exists=True, file_okay=False), help="Cache directory to download to. Default uses the huggingface cache.", default=None)
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False), help="Output file.", default="output.mp4", show_default=True)
@click.option("--paste-back", "-pb", is_flag=True, default=False, help="Paste the original background back in.", show_default=True)
@click.option("--paste-back-color-fix", "-pbcf", type=click.Choice(["adain", "wavelet"]), default="wavelet", help="Color fix method to use when pasting back.", show_default=True)
@click.option("--crop-to-face", "-ctf", is_flag=True, default=False, help="Crop the input to the face prior to execution, then merge the cropped result with the uncropped image. Implies --paste-back.", show_default=True)
@click.option("--pose-output", "-pop", type=click.Path(exists=False, dir_okay=False), help="When passed, save the pose image(s) to this file.", default=None, show_default=True)
@click.option("--mask-output", "-mop", type=click.Path(exists=False, dir_okay=False), help="When passed, save the mask image(s) to this file.", default=None, show_default=True)
@click.option("--combined-output", "-cop", type=click.Path(exists=False, dir_okay=False), help="When passed, save the combined image(s) to this file.", default=None, show_default=True)
@click.option("--mask-blur", "-mb", type=int, default=15, help="Amount of blur to apply to the mask when using cropping or pasting.", show_default=True)
@click.option("--mask-dilate", "-md", type=int, default=31, help="Amount of dilation to apply to the mask when using cropping or pasting.", show_default=True)
@click.option("--mask-slow", "-ms", is_flag=True, default=False, help="Use a slower, more accurate mask generation method.", show_default=True)
@click.option("--leading-seconds-silence", "-lss", type=float, default=0.0, help="Seconds of silence to add to the beginning of the audio.", show_default=True)
@click.option("--trailing-seconds-silence", "-tss", type=float, default=0.0, help="Seconds of silence to add to the end of the audio.", show_default=True)
def main(
    input_image: str,
    video: Optional[str]=None,
    audio: Optional[str]=None,
    frame_rate: Optional[int]=None,
    guidance_scale: float=3.5,
    num_inference_steps: int=20,
    context_frames: int=16,
    context_overlap: int=4,
    num_frames: Optional[int]=None,
    seed: Optional[int]=None,
    width: Optional[int]=None,
    height: Optional[int]=None,
    model: str="benjamin-paine/aniportrait",
    no_half: bool=False,
    no_offload: bool=False,
    gpu_id: int=0,
    model_single_file: bool=False,
    config_file: str="config.json",
    model_filename: str="aniportrait.safetensors",
    remote_subfolder: Optional[str]=None,
    cache_dir: Optional[str]=None,
    output: str="output.mp4",
    paste_back: bool=False,
    paste_back_color_fix: str="wavelet",
    crop_to_face: bool=False,
    pose_output: Optional[str]=None,
    mask_output: Optional[str]=None,
    combined_output: Optional[str]=None,
    mask_blur: int=15,
    mask_dilate: int=31,
    mask_slow: bool=False,
    leading_seconds_silence: float=0.0,
    trailing_seconds_silence: float=0.0,
) -> None:
    """
    Run AniPortrait on an input image with a video, and/or audio file. When only a video file is provided, a video-to-video (face reenactment) animation is performed. When only an audio file is provided, an audio-to-video (lip-sync) animation is performed. When both a video and audio file are provided, a video-to-video animation is performed with the audio as guidance for the face and mouth movements.
    """
    if not video and not audio:
        raise ValueError("You must provide either a video or audio file (or both.)")

    if os.path.exists(output):
        base, ext = os.path.splitext(os.path.basename(output))
        dirname = os.path.dirname(output)
        suffix = 1
        while os.path.exists(os.path.join(dirname, f"{base}-{suffix}{ext}")):
            suffix += 1
        new_output_filename = f"{base}-{suffix}{ext}"
        click.echo(f"Output file {output} already exists. Writing to {new_output_filename} instead.")
        output = os.path.join(dirname, new_output_filename)

    import torch
    from aniportrait.pipelines import AniPortraitPipeline
    from aniportrait.utils import Video, Audio, human_size, get_num_audio_samples, get_frame_rate

    device = (
        torch.device("cuda", index=gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if no_half:
        variant = None
        torch_dtype = None
    else:
        variant = "fp16"
        torch_dtype = torch.float16

    if model_single_file:
        pipeline = AniPortraitPipeline.from_single_file(
            model,
            filename=model_filename,
            config_filename=config_file,
            variant=variant,
            subfolder=remote_subfolder,
            cache_dir=cache_dir,
            device=device,
            torch_dtype=torch_dtype,
        )
    else:
        pipeline = AniPortraitPipeline.from_pretrained(
            model,
            variant=variant,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        )

    if not no_offload:
        pipeline.enable_model_cpu_offload(gpu_id=gpu_id)

    if torch_dtype is not None:
        pipeline.to(torch_dtype)
    pipeline.to(device)

    if frame_rate is None and video is not None:
        frame_rate = get_frame_rate(video)
    else:
        frame_rate = 30

    if audio and num_frames is None:
        num_frames = get_num_audio_samples(audio, fps=frame_rate) + int((leading_seconds_silence + trailing_seconds_silence) * frame_rate)

    if mask_blur is not None:
        if mask_blur % 2 != 1:
            mask_blur += 1
            click.echo(f"Mask blur must be an odd number. Adjusting to {mask_blur}.")

    video_container = None
    _, ext = os.path.splitext(input_image)
    if ext.lower() in [".mp4", ".mov", ".avi", ".webm"]:
        video_container = Video.from_file(
            input_image,
            image_format="RGB",
            maximum_frames=num_frames,
        )
        input_image = video_container.frames_as_list
    else:
        input_image = Image.open(input_image).convert("RGB")

    if audio:
        pose_reference_images = None
        if video:
            pose_video_container = Video.from_file(
                video,
                image_format="RGB",
                maximum_frames=num_frames,
            )
            pose_reference_images = pose_video_container.frames_as_list
            print(f"Loaded {len(pose_reference_images)} frames from {video}")

        kwargs: Dict[str, Any] = {}
        if isinstance(input_image, list):
            pipeline_callable = pipeline.audiovid2vid
            kwargs["paste_back"] = paste_back
            kwargs["paste_back_color_fix"] = paste_back_color_fix
            kwargs["crop_to_face"] = crop_to_face
            kwargs["mask_filename"] = mask_output
            kwargs["combined_filename"] = combined_output
            kwargs["mask_blur"] = mask_blur
            kwargs["mask_dilate"] = mask_dilate
            kwargs["mask_first_frame"] = not mask_slow
        else:
            pipeline_callable = pipeline.audio2vid

        result = pipeline_callable(
            audio,
            input_image,
            frame_rate=frame_rate,
            guidance_scale=guidance_scale,
            fps=frame_rate,
            pose_reference_images=pose_reference_images,
            num_inference_steps=num_inference_steps,
            context_frames=context_frames,
            context_overlap=context_overlap,
            num_frames=num_frames,
            seed=seed,
            width=width,
            height=height,
            video_length=num_frames,
            leading_seconds_silence=leading_seconds_silence,
            trailing_seconds_silence=trailing_seconds_silence,
            pose_filename=pose_output,
            **kwargs
        )
    else:
        video_container = Video.from_file(
            video,
            image_format="RGB",
            maximum_frames=num_frames,
        )
        pose_reference_images = video_container.frames_as_list
        print(f"Loaded {len(pose_reference_images)} frames from {video}")

        result = pipeline.vid2vid(
            input_image,
            pose_reference_images=pose_reference_images,
            num_inference_steps=num_inference_steps,
            fps=frame_rate,
            guidance_scale=guidance_scale,
            context_frames=context_frames,
            context_overlap=context_overlap,
            video_length=num_frames,
            seed=seed,
            width=width,
            height=height,
            pose_filename=pose_output,
        )

    # Save result
    if video_container is None:
        video_container = Video(result.videos, frame_rate=frame_rate)
    else:
        video_container.frames = result.videos

    if audio:
        audio_helper = Audio.from_file(audio)
        audio_frames = audio_helper.frames_as_list
        num_channels = len(audio_frames[0])
        audio_frames = [(0,) * num_channels] * int(leading_seconds_silence * audio_helper.rate) + audio_frames + [(0,) * num_channels] * int(trailing_seconds_silence * audio_helper.rate)
        video_container.audio = Audio(audio_frames, rate=audio_helper.rate)

    bytes_written = video_container.save(output, rate=frame_rate)
    click.echo(f"Wrote {len(result.videos)} frames to {output} ({human_size(bytes_written)})")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as ex:
        sys.stderr.write(f"{ex}\r\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        sys.exit(5)
