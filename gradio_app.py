#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
import os
import json
import tempfile
import shutil
from pathlib import Path
import torch
import sys
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import mediapy
import time
import traceback
import gc

# Add project path
path_to_insert = "humo"
if path_to_insert not in sys.path:
    sys.path.insert(0, path_to_insert)

from common.config import load_config, create_object
from common.distributed import get_device, get_global_rank, init_torch
from common.logger import get_logger
from humo.models.utils.utils import tensor_to_video


SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
}


class HuMoGradioApp:
    def __init__(self):
        self.generator = None
        self.config = None
        self.temp_dir = tempfile.mkdtemp()
        self.progress = 0
        self.is_generating = False
        self.model_type = None
        self.logger = get_logger(self.__class__.__name__)

    def load_model(self, model_type: str = "1.7B"):
        """Load HuMo model"""
        try:
            gc.collect()
            torch.cuda.empty_cache()

            if model_type == "1.7B":
                model_path = "./weights/HuMo/HuMo-1.7B"
                config_path = "humo/configs/inference/generate_1_7B.yaml"
                self.model_type = "1.7B"
            elif model_type == "17B":
                model_path = "./weights/HuMo/HuMo-17B"
                config_path = "humo/configs/inference/generate.yaml"
                self.model_type = "17B"
            else:
                return f"❌ Unsupported model type: {model_type}"

            if not os.path.exists(model_path):
                return f"❌ Model path does not exist: {model_path}"

            if not os.path.exists(config_path):
                return f"❌ Config file does not exist: {config_path}"

            self.config = load_config(config_path, [])
            self.config.dit.checkpoint_dir = model_path

            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_world_size() == 1:
                self.config.dit.sp_size = 1
                self.config.generation.sequence_parallel = 1
                print("Single GPU mode: sequence parallelism disabled")

            init_torch(cudnn_benchmark=False)
            self.generator = create_object(self.config)
            self.generator.configure_models()

            return f"✅ {model_type} model loaded successfully!"
        except Exception as e:
            traceback.print_exc()
            return f"❌ Failed to load model: {str(e)}"

    def update_progress(self) -> float:
        """Update progress bar"""
        if not self.is_generating:
            return 0
        self.progress += 1
        return min(99, self.progress)

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        ref_image: Optional[Image.Image] = None,
        audio_file: Optional[str] = None,
        mode: str = "TA",
        frames: int = 97,
        height: int = 720,
        width: int = 1280,
        scale_i: float = 5.0,
        scale_a: float = 5.5,
        scale_t: float = 5.0,
        sampling_steps: int = 50,
        seed: int = 666666,
        fps: int = 25
    ) -> Tuple[str, str, float]:
        """Generate video"""
        start_time = time.time()

        try:
            self.progress = 0
            self.is_generating = True

            if self.generator is None:
                self.is_generating = False
                return "", "❌ Please load the model first", 0

            if not prompt.strip():
                self.is_generating = False
                return "", "❌ Please enter a text prompt", 0

            output_dir = os.path.join(self.temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            self.logger.info("Creating test case file")
            test_case = {
                "case_1": {
                    "img_paths": [],
                    "audio_path": None,
                    "prompt": prompt
                }
            }

            if ref_image is not None and mode == "TIA":
                img_path = os.path.join(self.temp_dir, "ref_image.png")
                ref_image.save(img_path)
                test_case["case_1"]["img_paths"] = [img_path]
            elif mode == "TIA" and ref_image is None:
                self.is_generating = False
                return "", "❌ TIA mode requires a reference image", 0

            if audio_file is not None:
                self.logger.info(f"Received audio file: {audio_file}")
                if os.path.isfile(audio_file):
                    audio_path = os.path.join(self.temp_dir, "audio.wav")
                    shutil.copy2(audio_file, audio_path)
                    test_case["case_1"]["audio_path"] = audio_path
                else:
                    test_case["case_1"]["audio_path"] = None

            test_case_path = os.path.join(self.temp_dir, "test_case.json")
            with open(test_case_path, 'w', encoding='utf-8') as f:
                json.dump(test_case, f, ensure_ascii=False, indent=2)

            config_updates = {
                'mode': mode,
                'frames': frames,
                'height': height,
                'width': width,
                'scale_i': scale_i,
                'scale_a': scale_a,
                'scale_t': scale_t,
                'seed': seed,
                'fps': fps,
                'positive_prompt': test_case_path,
                'output': {'dir': output_dir}
            }

            if negative_prompt.strip():
                config_updates['sample_neg_prompt'] = negative_prompt

            self.generator.update_generation_config(**config_updates)
            self.generator.update_config(diffusion_timesteps_sampling_steps=sampling_steps)

            self.logger.info("Starting video generation...")
            generation_start_time = time.time()
            self.generator.inference_loop()
            generation_duration = time.time() - generation_start_time

            self.is_generating = False
            self.progress = 100

            output_files = list(Path(output_dir).glob("*.mp4"))
            if output_files:
                video_path = os.path.abspath(str(output_files[0]))
                total_duration = time.time() - start_time
                success_message = (
                    f"✅ Video generated successfully! "
                    f"Generation time: {generation_duration:.2f}s, Total: {total_duration:.2f}s"
                )
                return video_path, success_message, 100
            else:
                return "", "❌ No generated video file found", 0

        except Exception as e:
            total_duration = time.time() - start_time
            self.logger.error(f"Video generation failed, total time: {total_duration:.2f}s")
            traceback.print_exc()
            self.is_generating = False
            return "", f"❌ Video generation failed: {str(e)}", 0

    def cleanup(self):
        """Clean temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


app = HuMoGradioApp()


def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="HuMo Video Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # HuMo Multimodal Video Generator

        Generate high-quality human videos from text, image, and audio inputs.
        """)

        with gr.Row():
            with gr.Column(scale=4):
                model_type = gr.Dropdown(
                    choices=["1.7B", "17B"],
                    value="1.7B",
                    label="Model Type",
                    info="1.7B: Fast, lightweight | 17B: High quality, requires more VRAM"
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("Load Model", variant="primary")

        model_status = gr.Textbox(label="Model Status", interactive=False)

        with gr.Tabs() as tabs:
            with gr.TabItem("Basic Settings"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(
                            label="Text Prompt (Required)",
                            placeholder="Describe the video you want to generate...",
                            lines=3
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt (Optional)",
                            placeholder="Describe what you don’t want...",
                            lines=2,
                            value="Overexposed, static, low quality, blurry details, artifacts, distorted faces, extra fingers, messy background"
                        )
                        mode = gr.Radio(
                            choices=["TA", "TIA"],
                            value="TA",
                            label="Generation Mode",
                            info="TA: Text + Audio | TIA: Text + Image + Audio"
                        )
                        with gr.Row():
                            ref_image = gr.Image(
                                label="Reference Image (for TIA mode)",
                                type="pil",
                                visible=False
                            )
                            audio_file = gr.Audio(
                                label="Audio File",
                                type="filepath"
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### Example Prompts")
                        example_prompts = gr.Dataset(
                            components=[prompt],
                            samples=[
                                ["A person dancing energetically to upbeat music"],
                                ["Someone speaking confidently in a business meeting"],
                                ["A person playing guitar with passion"]
                            ],
                            label="Click to load example prompts"
                        )
                        generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.TabItem("Advanced Settings"):
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Video Settings", open=True):
                            with gr.Row():
                                frames = gr.Slider(minimum=-1, maximum=200, value=-1, step=1,
                                                   label="Number of Frames",
                                                   info="-1 to auto-calculate from audio; otherwise manual frame count")
                                fps = gr.Slider(minimum=15, maximum=60, value=25, step=1,
                                                label="Frame Rate (FPS)",
                                                info="Higher FPS = smoother video")
                            resolution = gr.Dropdown(
                                choices=[
                                    "720P Portrait (720x1280)",
                                    "720P Landscape (1280x720)",
                                    "480P Portrait (480x832)",
                                    "480P Landscape (832x480)",
                                    "1024P Square (1024x1024)"
                                ],
                                value="720P Portrait (720x1280)",
                                label="Resolution",
                                info="720P = better quality; 480P = faster generation"
                            )
                        with gr.Accordion("Generation Parameters", open=True):
                            with gr.Row():
                                scale_i = gr.Slider(1.0, 10.0, 5.0, 0.1, label="Image Guidance Strength")
                                scale_a = gr.Slider(1.0, 10.0, 5.5, 0.1, label="Audio Guidance Strength")
                                scale_t = gr.Slider(1.0, 10.0, 5.0, 0.1, label="Text Guidance Strength")
                            with gr.Row():
                                sampling_steps = gr.Slider(20, 100, 50, 5, label="Sampling Steps")
                                seed = gr.Number(label="Random Seed", value=-1, precision=0,
                                                 info="-1 = random seed; set fixed seed for reproducible results")

        progress_bar = gr.Progress()

        with gr.Row():
            with gr.Column():
                output_video = gr.Video(label="Generated Video", height=400)
                status_text = gr.Textbox(label="Status", interactive=False, lines=2)

        def load_model_wrapper(model_type):
            return app.load_model(model_type)

        def generate_video_wrapper(prompt, negative_prompt, ref_image, audio_file, mode, frames, resolution,
                                   scale_i, scale_a, scale_t, sampling_steps, seed, fps, progress=gr.Progress()):
            progress(0, desc="Preparing...")

            if "720P Landscape" in resolution:
                height, width = 720, 1280
            elif "720P Portrait" in resolution:
                height, width = 1280, 720
            elif "480P Landscape" in resolution:
                height, width = 480, 832
            elif "480P Portrait" in resolution:
                height, width = 832, 480
            elif "1024P Square" in resolution:
                height, width = 1024, 1024
            else:
                height, width = 720, 1280

            video_path, status, final_progress = app.generate_video(
                prompt, negative_prompt, ref_image, audio_file, mode, frames, height, width,
                scale_i, scale_a, scale_t, sampling_steps, seed, fps
            )

            progress(final_progress / 100)
            return video_path, status

        def update_ref_image_visibility(mode):
            return gr.update(visible=(mode == "TIA"))

        def load_example_prompt(evt: gr.SelectData):
            return evt.value[0]

        load_btn.click(fn=load_model_wrapper, inputs=[model_type], outputs=[model_status])
        mode.change(fn=update_ref_image_visibility, inputs=[mode], outputs=[ref_image])
        example_prompts.select(fn=load_example_prompt, inputs=[], outputs=[prompt])
        generate_btn.click(
            fn=generate_video_wrapper,
            inputs=[prompt, negative_prompt, ref_image, audio_file, mode, frames, resolution,
                    scale_i, scale_a, scale_t, sampling_steps, seed, fps],
            outputs=[output_video, status_text]
        )

        with gr.Accordion("Help", open=False):
            gr.Markdown("""
            ## Quick Start

            1. Select the model type.
               - 1.7B: lightweight and faster.
               - 17B: higher quality, requires more VRAM.
            2. Load the model.
            3. Choose the generation mode.
               - TA: text + audio.
               - TIA: text + image + audio.
            4. Enter your prompt.
            5. Upload optional reference image or audio.
            6. Click Generate Video.

            ## Tips
            - **Positive Prompts**: Use detailed and specific descriptions.
            - **Negative Prompts**: Describe what you don’t want (e.g., blurry, static, low quality).
            - **Guidance Strengths**: Adjust text/audio/image influence balance.
            - **Sampling Steps**: 30–50 recommended.
            - **Resolution**: 720P for quality, 480P for speed.

            ## Troubleshooting
            - Check model paths: `./weights/HuMo/HuMo-1.7B` or `./weights/HuMo/HuMo-17B`.
            - TIA mode requires a reference image.
            - Audio must be in WAV format.
            - Reduce resolution, steps, or frames to improve performance.
            """)

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", share=True, debug=True, show_error=True)
