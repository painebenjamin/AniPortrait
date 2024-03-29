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
@click.option("--gpu-id", "-g", type=int, default=0, help="GPU ID to use.")
@click.option("--model-single-file", "-sf", is_flag=True, default=False, help="Download and use a single file instead of a directory.")
@click.option("--config-file", "-cf", type=str, default="config.json", help="Config file to use when using the model-single-file option. Accepts a path or a filename in the same directory as the single file. Will download from the repository passed in the model option if not provided.", show_default=True)
@click.option("--model-filename", "-mf", type=str, default="aniportrait.safetensors", help="The model file to download when using the model-single-file option.", show_default=True)
@click.option("--remote-subfolder", "-rs", type=str, default=None, help="Remote subfolder to download from when using the model-single-file option.")
@click.option("--cache-dir", "-c", type=click.Path(exists=True, file_okay=False), help="Cache directory to download to. Default uses the huggingface cache.", default=None)
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False), help="Output file.", default="output.mp4", show_default=True)
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
    gpu_id: int=0,
    model_single_file: bool=False,
    config_file: str="config.json",
    model_filename: str="aniportrait.safetensors",
    remote_subfolder: Optional[str]=None,
    cache_dir: Optional[str]=None,
    output: str="output.mp4",
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
    from aniportrait.utils import Video

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
        pipeline = AniPortraitPipeline.from_model_single_file(
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

    if torch_dtype is not None:
        pipeline.to(torch_dtype)
    pipeline.to(device)

    input_image = Image.open(input_image).convert("RGB")

    if audio:
        pose_reference_images = None
        if video:
            pose_reference_video = Video.from_file(
                video,
                image_format="RGB",
                maximum_frames=num_frames,
            )
            pose_reference_images = pose_reference_video.frames_as_list
            print(f"Loaded {len(pose_reference_images)} frames from {video}")
            if frame_rate is None:
                frame_rate = pose_reference_video.frame_rate
        elif frame_rate is None:
            frame_rate = 30
        result = pipeline.audio2vid(
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
        )
    else:
        pose_reference_video = Video.from_file(
            video,
            image_format="RGB",
            maximum_frames=num_frames,
        )
        pose_reference_images = pose_reference_video.frames_as_list
        print(f"Loaded {len(pose_reference_images)} frames from {video}")
        if frame_rate is None:
            frame_rate = pose_reference_video.frame_rate
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
        )
    Video(
        result.videos,
        audio=audio if audio else video,
        frame_rate=frame_rate
    ).save(output)
    click.echo(f"Wrote {len(result.videos)} frames to {output}")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as ex:
        sys.stderr.write(f"{ex}\r\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        sys.exit(5)
