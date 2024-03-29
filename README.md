# AniPortrait

AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animations

Huawei Wei, Zejun Yang, Zhisheng Wang

Tencent Games Zhiji, Tencent

![zhiji_logo](asset/zhiji_logo.png)

Here we propose AniPortrait, a novel framework for generating high-quality animation driven by 
audio and a reference portrait image. You can also provide a video to achieve face reenacment.

<a href='https://arxiv.org/abs/2403.17694'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

## Pipeline

![pipeline](asset/pipeline.png)

## TODO List

- [x] Now our paper is available on arXiv.

- [x] Update the code to generate pose_temp.npy for head pose control.

- [ ] We will release audio2pose pre-trained weight for audio2video after futher optimization. You can choose head pose template in `./configs/inference/head_pose_temp` as substitution.

## Various Generated Videos

### Self driven

<table class="center">
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/Zejun-Yang/AniPortrait/assets/21038147/82c0f0b0-9c7c-4aad-bf0e-27e6098ffbe1" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/Zejun-Yang/AniPortrait/assets/21038147/51a502d9-1ce2-48d2-afbe-767a0b9b9166" muted="false"></video>
    </td>
</tr>
</table>

### Face reenacment

<table class="center">
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/Zejun-Yang/AniPortrait/assets/21038147/849fce22-0db1-4257-a75f-a5dc655e6b9e" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/Zejun-Yang/AniPortrait/assets/21038147/d4e0add6-20a2-4f4b-808c-530a6f4d3331" muted="false"></video>
    </td>
</tr>
</table>

### Audio driven

<table class="center">
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/Zejun-Yang/AniPortrait/assets/21038147/63171e5a-e4c1-4383-8f20-9764524928d0" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/Zejun-Yang/AniPortrait/assets/21038147/6fd74024-ba19-4f6b-b37a-10df5cf2c934" muted="false"></video>
    </td>
</tr>

<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/Zejun-Yang/AniPortrait/assets/21038147/9e516cc5-bf09-4d45-b5e3-820030764982" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/Zejun-Yang/AniPortrait/assets/21038147/7c68148b-8022-453f-be9a-c69590038197" muted="false"></video>
    </td>
</tr>
</table>

## Installation

### Build environment

We recommend a python version >=3.10 and cuda version =11.7. Then build environment as follows:

```shell
pip install git+https://github.com/painebenjamin/aniportrait.git
```

## Inference

You can now use the command line utility `aniportrait`. See these for examples in this repository:

### Face Reenactment

```sh
aniportrait configs/inference/ref_images/solo.png --video configs/inference/video/Aragaki_song.mp4 --num-frames 64 --width 512 --height 512
```
*Note: remove `--num-frames 64` to match the length of the video.*

### Audio Driven

```sh
aniportrait configs/inference/ref_images/lyl.png --audio configs/inference/video/lyl.wav --num-frames 96 --width 512 --height 512
```
*Note: remove `--num-frames 64` to match the length of the audio.*

### Help

For help, run `aniportrait --help`.

```sh
Usage: aniportrait [OPTIONS] INPUT_IMAGE

  Run AniPortrait on an input image with a video, and/or audio file. - When
  only a video file is provided, a video-to-video (face reenactment) animation
  is performed. - When only an audio file is provided, an audio-to-video (lip-
  sync) animation is performed. - When both a video and audio file are
  provided, a video-to-video animation is performed with the audio as guidance
  for the face and mouth movements.

Options:
  -v, --video FILE                Video file to drive the animation.
  -a, --audio FILE                Audio file to drive the animation.
  -fps, --frame-rate INTEGER      Video FPS. Also controls the sampling rate
                                  of the audio. Will default to the video FPS
                                  if a video file is provided, or 30 if not.
  -cfg, --guidance-scale FLOAT    Guidance scale for the diffusion process.
                                  [default: 3.5]
  -ns, --num-inference-steps INTEGER
                                  Number of diffusion steps.  [default: 20]
  -cf, --context-frames INTEGER   Number of context frames to use.  [default:
                                  16]
  -co, --context-overlap INTEGER  Number of context frames to overlap.
                                  [default: 4]
  -nf, --num-frames INTEGER       An explicit number of frames to use. When
                                  not passed, use the length of the audio or
                                  video
  -s, --seed INTEGER              Random seed.
  -w, --width INTEGER             Output video width. Defaults to the input
                                  image width.
  -h, --height INTEGER            Output video height. Defaults to the input
                                  image height.
  -m, --model TEXT                HuggingFace model name.
  -nh, --no-half                  Do not use half precision.
  -g, --gpu-id INTEGER            GPU ID to use.
  -sf, --single-file              Download and use a single file instead of a
                                  directory.
  -cf, --config-file TEXT         Config file to use when using the single-
                                  file option. Accepts a path or a filename in
                                  the same directory as the single file. Will
                                  download from the repository passed in the
                                  model option if not provided.  [default:
                                  config.json]
  -mf, --model-filename TEXT      The model file to download when using the
                                  single-file option.  [default:
                                  aniportrait.safetensors]
  -rs, --remote-subfolder TEXT    Remote subfolder to download from when using
                                  the single-file option.
  -c, --cache-dir DIRECTORY       Cache directory to download to. Default uses
                                  the huggingface cache.
  -o, --output FILE               Output file.  [default: output.mp4]
  --help                          Show this message and exit.
```

## Training

### Data preparation
Download [VFHQ](https://liangbinxie.github.io/projects/vfhq/) and [CelebV-HQ](https://github.com/CelebV-HQ/CelebV-HQ) 

Extract keypoints from raw videos and write training json file (here is an example of processing VFHQ): 

```shell
python -m scripts.preprocess_dataset --input_dir VFHQ_PATH --output_dir SAVE_PATH --training_json JSON_PATH
```

Update lines in the training config file: 

```yaml
data:
  json_path: JSON_PATH
```

### Stage1

Run command:

```shell
accelerate launch train_stage_1.py --config ./configs/train/stage1.yaml
```

### Stage2

Put the pretrained motion module weights `mm_sd_v15_v2.ckpt` ([download link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)) under `./pretrained_weights`. 

Specify the stage1 training weights in the config file `stage2.yaml`, for example:

```yaml
stage1_ckpt_dir: './exp_output/stage1'
stage1_ckpt_step: 30000 
```

Run command:

```shell
accelerate launch train_stage_2.py --config ./configs/train/stage2.yaml
```

## Acknowledgements

We first thank the authors of [EMO](https://github.com/HumanAIGC/EMO), and part of the images and audios in our demos are from EMO. Additionally, we would like to thank the contributors to the [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [majic-animate](https://github.com/magic-research/magic-animate), [animatediff](https://github.com/guoyww/AnimateDiff) and [Open-AnimateAnyone](https://github.com/guoqincode/Open-AnimateAnyone) repositories, for their open research and exploration.

## Citation

```
@misc{wei2024aniportrait,
      title={AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animations}, 
      author={Huawei Wei and Zejun Yang and Zhisheng Wang},
      year={2024},
      eprint={2403.17694},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
