import random

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from .base import NodeBase, first_image, register_node
from .io_nodes import OutOfMemoryError, free_memory

# A reasonable subset of diffusers' interchangeable schedulers. Each is
# swapped onto the cached pipeline via `SchedulerClass.from_config(pipe.scheduler.config)`
# right before sampling, so the same cached pipeline instance can be reused
# across KSampler nodes/runs that pick different samplers without reloading
# the underlying model weights.
SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm_multistep": DPMSolverMultistepScheduler,
    "pndm": PNDMScheduler,
    "ddim": DDIMScheduler,
    "lms": LMSDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
}

MAX_SEED = 2**31 - 1


@register_node("CLIP Text Encode")
class CLIPTextEncode(NodeBase):
    """Wraps the prompt text as CONDITIONING. There's no separate embedding
    step here: KSampler passes this text straight to the loaded pipeline's
    own `prompt=`/`negative_prompt=` arguments, which do their own
    tokenization/CLIP-encoding internally. This node exists so the graph
    keeps the familiar checkpoint -> conditioning -> sampler shape. The same
    node class is used for both the positive and negative prompt - wire a
    second instance into KSampler's optional `negative` input for a real
    negative prompt, the same way ComfyUI itself wires two CLIPTextEncode
    nodes into a KSampler.
    """

    CATEGORY = "conditioning"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True})}}

    def encode(self, text):
        return ({"text": text},)


@register_node("KSampler")
class KSamplerStub(NodeBase):
    """Runs real Stable Diffusion sampling via the diffusers pipeline that
    LoadCheckpoint attaches to `model`. `latent_image` only supplies the
    output width/height (it's a plain RGB PIL.Image, not a real VAE latent),
    so this is txt2img-only for now - no image-to-image or inpainting.

    `sampler` swaps the pipeline's scheduler (see SCHEDULERS above). `seed`
    reproduces the same output when >= 0; a negative seed (the default, -1)
    picks a fresh random seed each run. `negative` is an optional second
    CONDITIONING input (wire a second CLIPTextEncode into it) passed through
    as the pipeline's `negative_prompt=`. `batch_size` generates more than
    one image per run via `num_images_per_prompt=`; the output is always a
    list of one or more PIL images. The seed actually used (after resolving
    -1 to a random one) is reported in the node's UI output as `{"seed": N}`,
    which lands in the prompt's history/`execution_success` message and in
    Save/Preview Image's embedded PNG metadata.

    For a distilled few-step checkpoint like the default
    `stabilityai/sd-turbo`, use 1-4 steps and cfg 0: diffusers only applies
    classifier-free guidance (and the negative prompt) when
    guidance_scale > 1, and turbo models are trained without it.
    """

    CATEGORY = "sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("IMAGE",),
                "sampler": (list(SCHEDULERS.keys()), {"default": "euler"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": MAX_SEED}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            },
        }

    def sample(
        self,
        model,
        positive,
        latent_image,
        sampler="euler",
        seed=-1,
        steps=20,
        cfg=7.0,
        batch_size=1,
        negative=None,
        on_progress=None,
    ):
        pipe = model["pipeline"]

        scheduler_cls = SCHEDULERS.get(sampler)
        if scheduler_cls is not None and not isinstance(pipe.scheduler, scheduler_cls):
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)

        text = positive.get("text", "")
        negative_text = negative.get("text", "") if negative else None

        # latent_image may be a batch (e.g. another KSampler's output); only
        # its size matters here.
        width, height = first_image(latent_image).size
        # SD pipelines require dimensions that are multiples of 8.
        width = max(8, (width // 8) * 8)
        height = max(8, (height // 8) * 8)

        seed = int(seed)
        if seed < 0:
            seed = random.randint(0, MAX_SEED)
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

        total_steps = int(steps)

        def _report_progress(pipe_self, step, timestep, callback_kwargs):
            # diffusers' callback_on_step_end contract: called after each
            # denoising step with (pipeline, step_index, timestep,
            # callback_kwargs), and must return callback_kwargs unmodified
            # (or with any changes to the tensors it requested) for the loop
            # to continue normally. `step` is 0-based; report 1-based so a
            # UI can show e.g. "3 / 20" rather than "2 / 20".
            on_progress(step + 1, total_steps)
            return callback_kwargs

        out_of_memory = False
        try:
            result = pipe(
                prompt=text,
                negative_prompt=negative_text,
                width=width,
                height=height,
                num_inference_steps=total_steps,
                guidance_scale=float(cfg),
                num_images_per_prompt=int(batch_size),
                generator=generator,
                callback_on_step_end=_report_progress if on_progress is not None else None,
            )
        except OutOfMemoryError:
            # Handled outside the except block: the in-flight exception's
            # traceback keeps the pipeline call's frames (and their GPU
            # tensors) alive, so memory can't be freed until it's gone.
            out_of_memory = True
        if out_of_memory:
            # Return the failed run's activations to the driver; the cached
            # pipeline's weights stay loaded so a smaller retry doesn't have
            # to reload the checkpoint.
            free_memory()
            raise RuntimeError(
                f"Out of GPU memory while sampling {width}x{height} (batch_size={int(batch_size)}). "
                "Try a smaller image size or batch_size, or close other GPU applications."
            )
        # Always a list (even for batch_size == 1) so every downstream IMAGE
        # consumer sees one consistent batch shape (see nodes/base.py's
        # as_image_list/map_images helpers).
        return {"seed": seed, "result": (list(result.images),)}
