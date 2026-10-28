import inspect
import logging
import random
import threading
import time
import weakref
from collections import OrderedDict

from PIL import Image

from . import controlnet, embeddings, ip_adapter, latent_preview, models, prompt_weights, regional, runtime, wildcards
from .base import NodeBase, as_image_list, first_image, register_node
from .io_nodes import MODEL_INFO_KEY, NODE_INFO_KEY, SEED_INFO_KEY

logger = logging.getLogger(__name__)

# Looked up (lazily, by class name) on diffusers at sample time, so this
# module doesn't import diffusers at startup (report item P6).
SCHEDULERS = {
    "euler": "EulerDiscreteScheduler",
    "euler_ancestral": "EulerAncestralDiscreteScheduler",
    "dpm_multistep": "DPMSolverMultistepScheduler",
    "pndm": "PNDMScheduler",
    "ddim": "DDIMScheduler",
    "lms": "LMSDiscreteScheduler",
    "heun": "HeunDiscreteScheduler",
}

# Sigma / timestep schedules (report item B8): "karras" is the most used
# A1111/Forge variant, "trailing" what SDXL-turbo-style models want.
SCHEDULE_TYPES = ["normal", "karras", "trailing"]

MAX_SEED = 2**31 - 1

# Above this many output pixels the VAE decodes in tiles (bounded memory,
# but tiling can leave faint seams, so smaller images decode in one piece).
# A decode that still runs out of memory is retried once with tiling on.
VAE_TILING_MIN_PIXELS = 1536 * 1536

# How often a live latent preview is sent at most.
PREVIEW_INTERVAL_S = 0.25


def free_memory() -> None:
    runtime.free_memory()


@register_node("CLIP Text Encode")
class CLIPTextEncode(NodeBase):
    """The prompt text as CONDITIONING. Weighting with (word:1.2), (word),
    [word] is supported for SD 1.x/2.x, SDXL, SD3 and Flux, and prompts
    longer than 77 CLIP tokens are encoded in chunks instead of being cut
    off (SD 1.x/2.x, SDXL, SD3; Flux reads up to 512 T5 tokens). `clip_skip`
    applies to the CLIP encoders of SD 1.x/2.x, SDXL and SD3 (Flux has
    none). The encoding itself happens in the sampler, where the model is
    known (backend/nodes/prompt_weights.py). Wire a second instance into the
    sampler's `negative` input for a negative prompt (Flux ignores it).

    Wildcards: `{red|green|blue}` picks one alternative and `__animal__` one
    line of wildcards/animal.txt in the data folder, chosen per image seed, so
    a batch varies and each image reproduces from its own seed."""

    CATEGORY = "conditioning"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    DESCRIPTION = ("Prompt text; supports (word:1.2) weighting, long prompts and wildcards "
                   "({a|b|c}, __file__ from the wildcards folder), resolved per seed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"text": ("STRING", {"default": "", "multiline": True})},
            "optional": {
                "clip_skip": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 12,
                        "tooltip": "Skip the last N CLIP layers (1 = A1111/ComfyUI 'clip skip 2')",
                    },
                )
            },
        }

    @classmethod
    def IS_CHANGED(cls, text="", **_):
        """F21: the wildcard files the text names, so editing one re-runs the
        sampler instead of reusing its cached output."""
        stamps = []
        for name in wildcards.referenced_files(text):
            try:
                stat = wildcards.wildcard_file(name).stat()
                stamps.append((name, stat.st_mtime_ns, stat.st_size))
            except (OSError, ValueError):
                stamps.append((name, None, None))
        return stamps

    def encode(self, text, clip_skip=0):
        # Wildcards ({a|b}, __name__) stay in the text: the sampler resolves
        # them per image seed (nodes/wildcards.py, report item F21).
        return ({"text": text, "clip_skip": int(clip_skip)},)


# ---- LoRA (report item B6) ----


@register_node("Load LoRA")
class LoraLoader(NodeBase):
    """Adds a LoRA from models/loras/ to the MODEL at `strength`. Chain
    several for several LoRAs. Nothing is changed in the shared, cached
    pipeline here: the sampler activates exactly this MODEL's LoRAs for its
    run (and none for a MODEL without them). Needs the `peft` package."""

    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    DESCRIPTION = "Applies a LoRA (models/loras/*.safetensors) to a model."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (models.list_models("loras"), {"tooltip": "A .safetensors file in models/loras/"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, lora_name="", **_):
        path = models.local_model_path("loras", lora_name)
        if path is not None and path.is_file():
            stat = path.stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        return float("nan")

    def apply(self, model, lora_name, strength=1.0):
        path = models.local_model_path("loras", lora_name)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"LoRA '{lora_name}' is not in models/loras/")
        loras = list(model.get("loras", [])) + [(lora_name, float(strength))]
        return ({**model, "loras": loras},)


_lora_lock = threading.Lock()


def _adapter_name(lora_name: str, path) -> str:
    stat = path.stat()
    safe = "".join(c if c.isalnum() else "_" for c in lora_name)
    return f"{safe}_{stat.st_mtime_ns % 10**9}"


def apply_loras(pipe, loras) -> None:
    """Make exactly `loras` ([(name, strength)]) active on `pipe` for this
    run: each LoRA file is loaded once per pipeline (as a named adapter)
    and then just switched on/off with set_adapters()/disable_lora(), so the
    cached pipeline never accumulates a previous run's LoRAs."""
    loaded = getattr(pipe, "_pixelflow_loras", None)
    if not loras:
        if loaded:
            pipe.disable_lora()
        return
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(f"{type(pipe).__name__} does not support LoRAs")
    try:
        import peft  # noqa: F401
    except ImportError:
        raise RuntimeError("LoRA support needs the 'peft' package: pip install peft") from None
    with _lora_lock:
        loaded = getattr(pipe, "_pixelflow_loras", None)
        if loaded is None:
            loaded = set()
            setattr(pipe, "_pixelflow_loras", loaded)
        names, weights = [], []
        for lora_name, strength in loras:
            path = models.local_model_path("loras", lora_name)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"LoRA '{lora_name}' is not in models/loras/")
            adapter = _adapter_name(lora_name, path)
            if adapter not in loaded:
                logger.info("loading LoRA %s", path)
                pipe.load_lora_weights(str(path), adapter_name=adapter)
                loaded.add(adapter)
            if adapter in names:  # the same LoRA chained twice: strengths add up
                weights[names.index(adapter)] += float(strength)
            else:
                names.append(adapter)
                weights.append(float(strength))
        pipe.enable_lora()
        pipe.set_adapters(names, adapter_weights=weights)


# ---- shared sampling machinery (KSampler, KSampler (Image), KSampler (Inpaint)) ----

# img2img/inpainting pipelines derived from a loaded txt2img pipeline via
# from_pipe(), which shares its components (UNet, VAE, text encoder) - no
# extra weights are loaded. Keyed weakly by the base pipeline, so a derived
# pipeline goes away together with its base when the pipeline cache drops it.
_DERIVED_FACTORIES = {"img2img": "AutoPipelineForImage2Image", "inpaint": "AutoPipelineForInpainting"}
_derived_pipelines: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_derived_lock = threading.Lock()


def derived_pipeline(pipe, kind: str):
    import diffusers

    with _derived_lock:
        per_pipe = _derived_pipelines.setdefault(pipe, {})
        derived = per_pipe.get(kind)
        if derived is None:
            derived = getattr(diffusers, _DERIVED_FACTORIES[kind]).from_pipe(pipe)
            derived.set_progress_bar_config(disable=True)
            per_pipe[kind] = derived
        return derived


def _sampler_inputs(image_input=None):
    """INPUT_TYPES shared by the three samplers; `image_input` adds the
    source-image inputs of the img2img/inpaint variants."""
    required = {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
    }
    required.update(image_input or {"latent_image": ("IMAGE",)})
    required.update(
        {
            "sampler": (list(SCHEDULERS.keys()), {"default": "euler"}),
            "seed": (
                "INT",
                {
                    "default": -1,
                    "min": -1,
                    "max": MAX_SEED,
                    "control_after_generate": True,
                    "tooltip": "-1 = random; image i of a batch uses seed + i",
                },
            ),
            "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
        }
    )
    return {
        "required": required,
        "optional": {
            "negative": ("CONDITIONING",),
            "scheduler": (list(SCHEDULE_TYPES), {"default": "normal", "tooltip": "Sigma schedule: normal, karras or trailing"}),
        },
        "hidden": {"unique_id": "UNIQUE_ID"},
    }


def _sampler_is_changed(seed=-1, **_):
    # A random seed (-1, or a seed fed by a link) must re-sample every run;
    # a fixed seed with unchanged inputs reproduces the same image, so the
    # engine may reuse the previous result (engine.graph.OutputCache).
    try:
        return float("nan") if int(seed) < 0 else ""
    except (TypeError, ValueError):
        return float("nan")


def _set_vae_tiling(pipe, enabled: bool) -> None:
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    if enabled and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    elif not enabled and hasattr(vae, "disable_tiling"):
        vae.disable_tiling()


def _can_tile(pipe) -> bool:
    return hasattr(getattr(pipe, "vae", None), "enable_tiling")


def _snap8(size):
    """SD pipelines need dimensions that are multiples of 8."""
    width, height = size
    return max(8, (width // 8) * 8), max(8, (height // 8) * 8)


def make_scheduler(pipe, sampler: str, schedule: str = "normal", base_config=None, base_class=None):
    """A fresh scheduler for this run, always derived from the checkpoint's
    *own* scheduler config (kept when it was loaded), never from whatever a
    previous run swapped in - so options like karras don't leak between
    runs that share the cached pipeline. A flow-matching model (Flux, SD3,
    Z-Image: `base_class` FlowMatch...) keeps its own scheduler class: the
    samplers in SCHEDULERS are for noise-prediction models only."""
    import diffusers

    config = dict(base_config or getattr(pipe.scheduler, "config", {}) or {})
    class_name = SCHEDULERS.get(sampler)
    if base_class and str(base_class).startswith("FlowMatch"):
        if sampler != "euler":
            logger.info("%s is a flow-matching model; sampler '%s' ignored", type(pipe).__name__, sampler)
        class_name = base_class
    scheduler_cls = getattr(diffusers, class_name) if class_name else type(pipe.scheduler)
    overrides = {}
    if schedule == "karras":
        overrides["use_karras_sigmas"] = True
    elif schedule == "trailing":
        overrides["timestep_spacing"] = "trailing"
    if schedule != "normal":
        accepted = set(inspect.signature(scheduler_cls.__init__).parameters)
        dropped = [k for k in overrides if k not in accepted]
        if dropped:
            logger.info("%s has no %s option; using its normal schedule", scheduler_cls.__name__, schedule)
        overrides = {k: v for k, v in overrides.items() if k in accepted}
    # As kwargs, not merged into the dict: from_config() resets keys listed in
    # the config's _use_default_values back to their defaults.
    return scheduler_cls.from_config(config, **overrides)


def _supported_kwargs(pipe, kwargs):
    """Drop call arguments this pipeline class doesn't take (e.g. Flux has
    no negative prompt in some versions), instead of failing."""
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    unsupported = [k for k in kwargs if k not in params]
    if unsupported:
        logger.debug("%s does not take %s; dropped", type(pipe).__name__, unsupported)
    return {k: v for k, v in kwargs.items() if k in params}


def _texts(positive, negative):
    return positive.get("text", ""), (negative.get("text", "") if negative else None)


def _wildcard_runs(positive, negative, seed, count):
    """F21: when either prompt has wildcards, one (seed, positive, negative)
    per output image -- image i resolved with seed + i, exactly what a re-run
    of that image alone (Save Image embeds its seed, batch_size 1) resolves --
    plus the base seed; None when there are no wildcards (one batched call)."""
    text, negative_text = _texts(positive, negative)
    if not wildcards.has_wildcards(text) and not wildcards.has_wildcards(negative_text):
        return None
    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, MAX_SEED)
    runs = []
    for i in range(int(count)):
        image_seed = (seed + i) % (MAX_SEED + 1)
        image_positive = dict(positive, text=wildcards.resolve(text, image_seed))
        image_negative = (dict(negative, text=wildcards.resolve(negative_text, image_seed, "negative"))
                          if negative else None)
        runs.append((image_seed, image_positive, image_negative))
    return seed, runs


def _run_progress(on_progress, index, count):
    """Progress of run `index` of `count` single-image runs as one bar."""
    if on_progress is None:
        return None

    def report(current, total, **info):
        on_progress(index * total + current, count * total, **info)

    report.wants_previews = getattr(on_progress, "wants_previews", False)
    return report


def _merge_runs(seed, outputs):
    images = [image for output in outputs for image in output["result"][0]]
    seeds = [s for output in outputs for s in output["seeds"]]
    return {"seed": seed, "seeds": seeds, "result": (images,)}


def _repeat_rows(tensor, repeat: int):
    """Repeat a [1, ...] embedding `repeat` times along the batch dimension."""
    return tensor if repeat == 1 else tensor.repeat(repeat, *([1] * (tensor.dim() - 1)))


# Encoded prompts, reused while the prompt, the model and its LoRAs stay the same: a re-run after
# changing only the seed, steps or sampler skips the text encoders (T5-XXL takes a while, and much
# longer when it's offloaded to the CPU). Keyed by the base pipeline (held weakly: an evicted model's
# entries just stop matching) and cleared with the models by /api/free.
EMBED_CACHE_SIZE = 16
_embed_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_embed_cache_lock = threading.Lock()


def clear_prompt_cache() -> None:
    with _embed_cache_lock:
        _embed_cache.clear()


def _cached_embeds(base_pipe, key: tuple, encode):
    """encode() once per `key` for `base_pipe`; the result is a dict of tensors."""
    full_key = (id(base_pipe),) + key
    with _embed_cache_lock:
        hit = _embed_cache.get(full_key)
        if hit is not None and hit[0]() is base_pipe:
            _embed_cache.move_to_end(full_key)
            return hit[1]
    embeds = encode()
    if embeds is not None:
        with _embed_cache_lock:
            _embed_cache[full_key] = (weakref.ref(base_pipe), embeds)
            while len(_embed_cache) > EMBED_CACHE_SIZE:
                _embed_cache.popitem(last=False)
    return embeds


def _reference_embeds(pipe, family, text, negative_text):
    """The pipeline's own encode_prompt() for a plain prompt -- exactly what pipe() would compute
    from the text -- as call kwargs, or None when this pipeline has no encode_prompt we know how to
    call (then the text goes to pipe() as before)."""
    encode = getattr(pipe, "encode_prompt", None)
    if encode is None:
        return None
    device = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", None)
    try:
        if family == prompt_weights.FAMILY_SD:
            pe, npe = encode(text, device, 1, True, negative_prompt=negative_text)
            return {"prompt_embeds": pe, "negative_prompt_embeds": npe}
        if family in (prompt_weights.FAMILY_SDXL, prompt_weights.FAMILY_SD3):
            if family == prompt_weights.FAMILY_SDXL:
                result = encode(text, None, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True,
                                negative_prompt=negative_text)
            else:
                result = encode(text, None, None, device=device, num_images_per_prompt=1,
                                do_classifier_free_guidance=True, negative_prompt=negative_text)
            pe, npe, pooled, npooled = result
            return {"prompt_embeds": pe, "negative_prompt_embeds": npe, "pooled_prompt_embeds": pooled,
                    "negative_pooled_prompt_embeds": npooled}
        if family == prompt_weights.FAMILY_FLUX:
            pe, pooled, _text_ids = encode(text, None, device=device, num_images_per_prompt=1)
            return {"prompt_embeds": pe, "pooled_prompt_embeds": pooled}
    except (TypeError, ValueError, AttributeError) as exc:  # another signature: keep the text path
        logger.debug("%s.encode_prompt not usable for caching (%s)", type(pipe).__name__, exc)
    return None


def _lora_signature(model) -> tuple:
    return tuple((str(name), float(strength)) for name, strength in (model or {}).get("loras", []) or ())


def _prompt_kwargs(model, pipe, positive, negative, repeat: int):
    """Prompt embeddings (`prompt_embeds`, plus `pooled_prompt_embeds` for SDXL/SD3/Flux, and
    negatives): from our own encoder when weighting, long prompts or clip skip need it
    (prompt_weights.py, report item B9), else from the pipeline's own encode_prompt() -- either way
    cached between runs (_cached_embeds). Plain `prompt`/`negative_prompt` text only for a pipeline
    we can't encode for. The model's LoRAs are applied first: a LoRA can change the text encoder."""
    text, negative_text = _texts(positive, negative)
    clip_skip = int(positive.get("clip_skip", 0) or 0)
    arch = (model or {}).get("arch") or models.pipeline_meta(model.get("pipeline") if model else None).get("arch")
    family = prompt_weights.encoding_family(pipe, arch)
    base_pipe = (model or {}).get("pipeline", pipe)
    if model:
        apply_loras(base_pipe, model.get("loras", []))
    # B22: embedding:name -> its token, loaded into the shared tokenizer/text encoder on first use.
    text = embeddings.prepare(pipe, family, text)
    negative_text = embeddings.prepare(pipe, family, negative_text)
    key_base = (family, text, negative_text, clip_skip, _lora_signature(model))
    # clip_skip goes through our own encoder too: diffusers' SD 1.x clip_skip
    # path reaches for CLIPTextModel.text_model, which transformers 5 removed.
    # Multi-vector embeddings count with all their vectors toward the 75-token window.
    expanded = [embeddings.expand_vectors(pipe, text), embeddings.expand_vectors(pipe, negative_text)]
    if prompt_weights.needs_custom_encoding(pipe, family, expanded, clip_skip):
        embeds = _cached_embeds(
            base_pipe, ("ours",) + key_base,
            lambda: prompt_weights.encode_conditioning(pipe, family, expanded[0], expanded[1], clip_skip),
        )
        return {key: _repeat_rows(value, repeat) for key, value in embeds.items()}
    if family is not None and not clip_skip:
        embeds = _cached_embeds(
            base_pipe, ("pipeline",) + key_base, lambda: _reference_embeds(pipe, family, text, negative_text)
        )
        if embeds is not None:
            return {key: _repeat_rows(value, repeat) for key, value in embeds.items()}
    if family != prompt_weights.FAMILY_SD:
        # Plain text: either nothing of ours is needed (diffusers' own
        # encoding is the reference then), or the pipeline is one we have no
        # encoder for; either way no weighting syntax reaches the model.
        text = prompt_weights.plain_text(text)
        negative_text = prompt_weights.plain_text(negative_text) if negative_text is not None else None
    kwargs = {
        "prompt": [text] * repeat if repeat > 1 else text,
        "negative_prompt": ([negative_text] * repeat if repeat > 1 else negative_text) if negative_text is not None else None,
    }
    if clip_skip and family is None:
        kwargs["clip_skip"] = clip_skip
    return kwargs


def _run_diffusion(
    pipe,
    sampler,
    seed,
    call_kwargs,
    width,
    height,
    batch_size,
    total_steps,
    on_progress,
    *,
    images_out=None,
    schedule="normal",
    model=None,
    node_id=None,
    conditioning=None,
):
    """Build the scheduler, resolve the seed, and call `pipe(**call_kwargs)`
    with per-image generators, progress (and live previews), and
    out-of-memory handling. Returns {"seed": N, "seeds": [...], "result":
    ([images],)} - the node output shape all three samplers share. Image i
    of the batch is generated from seed + i, and carries that seed in its
    .info so Save Image can embed it."""
    torch = runtime.torch()
    meta = models.pipeline_meta((model or {}).get("pipeline")) or models.pipeline_meta(pipe)
    pipe.scheduler = make_scheduler(pipe, sampler, schedule, meta.get("scheduler_config"), meta.get("scheduler_class"))
    apply_loras((model or {}).get("pipeline", pipe), (model or {}).get("loras", []))
    # B20: exactly this MODEL's IP-Adapter (or none) on the shared UNet, and its image argument.
    call_kwargs = {**call_kwargs, **ip_adapter.activate(pipe, (model or {}).get("ip_adapter"))}
    # F19: the positive prompt's regions, encoded once for this run (after the LoRAs, which can
    # change the text encoder).
    regions = _prepared_regions(model, pipe, conditioning, width, height)
    unet_config = getattr(getattr(pipe, "unet", None), "config", None)
    cfg_active = float(call_kwargs.get("guidance_scale", 0.0)) > 1.0 and getattr(unet_config, "time_cond_proj_dim", None) is None

    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, MAX_SEED)
    images_out = int(images_out or batch_size)
    seeds = [(seed + i) % (MAX_SEED + 1) for i in range(images_out)]
    arch = meta.get("arch") or (model or {}).get("arch") or models.ARCH_SD

    last_preview = [0.0]

    def _report_progress(pipe_self, step, timestep, callback_kwargs):
        # diffusers' callback_on_step_end contract: called after each
        # denoising step with (pipeline, step_index, timestep,
        # callback_kwargs), and must return callback_kwargs. `step` is
        # 0-based; report 1-based so a UI can show e.g. "3 / 20".
        current = min(step + 1, total_steps)
        info = {}
        latents = callback_kwargs.get("latents")
        now = time.monotonic()
        if latents is not None and (now - last_preview[0] >= PREVIEW_INTERVAL_S or current == total_steps):
            last_preview[0] = now
            try:
                image = latent_preview.latents_to_image(latents, arch, width, height)
            except Exception:  # noqa: BLE001 - a preview must never break sampling
                image = None
            if image is not None:
                info["preview"] = latent_preview.to_data_url(image)
        on_progress(current, total_steps, **info)
        return callback_kwargs

    previews = on_progress is not None and getattr(on_progress, "wants_previews", False)
    tile = width * height > VAE_TILING_MIN_PIXELS
    for _attempt in range(2):
        _set_vae_tiling(pipe, tile)
        # Fresh generators per attempt, so a retry samples the same images.
        device = getattr(pipe, "_execution_device", None) or pipe.device
        generator_device = device if getattr(device, "type", str(device)) != "mps" else "cpu"
        generators = [torch.Generator(device=generator_device).manual_seed(s) for s in seeds]
        kwargs = dict(call_kwargs)
        kwargs["generator"] = generators
        if on_progress is not None:
            kwargs["callback_on_step_end"] = _report_progress
            if previews:
                kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        out_of_memory = False
        try:
            with regional.regional_unet(pipe, regions, cfg_active):
                result = pipe(**_supported_kwargs(pipe, kwargs))
            images = list(result.images)
            for index, image in enumerate(images):
                image.info[SEED_INFO_KEY] = seeds[index] if index < len(seeds) else seed
                if node_id is not None:
                    image.info[NODE_INFO_KEY] = node_id
                if model and model.get("name"):
                    image.info[MODEL_INFO_KEY] = model["name"]
            # Always a list (even for batch_size == 1) so every downstream
            # IMAGE consumer sees one consistent batch shape (see
            # nodes/base.py's as_image_list/map_images helpers).
            return {"seed": seed, "seeds": seeds, "result": (images,)}
        except runtime.oom_error():
            # Handled outside the except block: the in-flight exception's
            # traceback keeps the pipeline call's frames (and their GPU
            # tensors) alive, so memory can't be freed until it's gone.
            out_of_memory = True
        if out_of_memory:
            free_memory()
            if tile or not _can_tile(pipe):
                break
            logger.warning("out of memory sampling %dx%d; retrying once with VAE tiling", width, height)
            tile = True
    raise RuntimeError(
        f"Out of GPU memory while sampling {width}x{height} (batch_size={int(batch_size)}). "
        "Try a smaller image size or batch_size, or close other GPU applications."
    )


def _prepared_regions(model, pipe, conditioning, width, height):
    """F19: each region of the positive conditioning as {"embeds", "pooled", "mask"} -- its prompt
    encoded like the main one (weighting, long prompts, clip skip), its mask at the image size."""
    positive, negative = conditioning or (None, None)
    regions = regional.region_conditioning(positive)
    if not regions:
        return []
    arch = (model or {}).get("arch") or models.pipeline_meta(pipe).get("arch")
    if arch not in (models.ARCH_SD, models.ARCH_SDXL) or getattr(pipe, "unet", None) is None:
        raise RuntimeError(f"Regional prompts work with SD 1.x/2.x and SDXL models, not this one ({arch or type(pipe).__name__})")
    prepared = []
    for region in regions:
        region_positive = {"text": region["text"], "clip_skip": positive.get("clip_skip", 0)}
        embeds = _prompt_kwargs(model, pipe, region_positive, negative, 1)
        if "prompt_embeds" not in embeds:
            raise RuntimeError("Regional prompts need a model whose prompts PixelFlow encodes itself (SD 1.x/2.x, SDXL)")
        prepared.append({"embeds": embeds["prompt_embeds"], "pooled": embeds.get("pooled_prompt_embeds"),
                         "mask": regional.region_mask(region, width, height)})
    return prepared


def _source_images(image):
    """An IMAGE input as a list of RGB images all sized like the first one
    (snapped to multiples of 8) - one img2img/inpaint batch."""
    images = [img.convert("RGB") for img in as_image_list(image)]
    size = _snap8(images[0].size)
    return [img if img.size == size else img.resize(size, Image.LANCZOS) for img in images], size


def _denoise_steps(steps, denoise) -> int:
    """How many steps an img2img/inpaint run actually takes (diffusers skips
    the first (1 - strength) share of the schedule)."""
    effective = min(int(int(steps) * float(denoise)), int(steps))
    if effective < 1:
        raise ValueError(
            f"denoise {float(denoise):g} x steps {int(steps)} leaves no sampling steps - "
            "raise denoise or steps (their product must be at least 1)"
        )
    return effective


@register_node("KSampler")
class KSampler(NodeBase):
    """Text-to-image sampling with the pipeline LoadCheckpoint attaches to
    `model`. `latent_image` only supplies the output width/height (it's a
    plain RGB image, not a real VAE latent) - use KSampler (Image) /
    KSampler (Inpaint) to start from an existing image.

    `sampler` picks the scheduler and `scheduler` its sigma schedule
    (normal/karras/trailing). `seed` reproduces the same output when >= 0;
    -1 picks a random seed. Image i of a batch uses seed + i, so it can be
    reproduced on its own (Save Image embeds each image's own seed). The
    seeds used are reported as {"seed": N, "seeds": [...]}.

    For a distilled few-step checkpoint like the default
    `stabilityai/sd-turbo`, use 1-4 steps and cfg 0: diffusers only applies
    classifier-free guidance (and the negative prompt) when
    guidance_scale > 1, and turbo models are trained without it.
    """

    CATEGORY = "sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    DESCRIPTION = "Generates images from a prompt (txt2img)."

    @classmethod
    def INPUT_TYPES(cls):
        return _sampler_inputs()

    @classmethod
    def IS_CHANGED(cls, **inputs):
        return _sampler_is_changed(**inputs)

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
        scheduler="normal",
        unique_id=None,
        on_progress=None,
    ):
        runs = _wildcard_runs(positive, negative, seed, batch_size)
        if runs:
            base_seed, items = runs
            return _merge_runs(base_seed, [
                self.sample(model, item_positive, latent_image, sampler, item_seed, steps, cfg, 1, item_negative,
                            scheduler, unique_id, _run_progress(on_progress, index, len(items)))
                for index, (item_seed, item_positive, item_negative) in enumerate(items)
            ])
        pipe = model["pipeline"]
        # latent_image may be a batch (e.g. another KSampler's output); only
        # its size matters here.
        width, height = _snap8(first_image(latent_image).size)
        controls = controlnet.controls_of(positive)
        if controls:  # B7: the ControlNet variant of the pipeline, base weights shared
            pipe = controlnet.control_pipeline(pipe, "text2img", controls)
        call_kwargs = {
            **_prompt_kwargs(model, pipe, positive, negative, 1),
            "width": width,
            "height": height,
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg),
            "num_images_per_prompt": int(batch_size),
        }
        if controls:
            call_kwargs.update(controlnet.control_kwargs("text2img", controls, (width, height)))
        return _run_diffusion(
            pipe,
            sampler,
            seed,
            call_kwargs,
            width,
            height,
            batch_size,
            int(steps),
            on_progress,
            schedule=scheduler,
            model=model,
            node_id=unique_id,
            conditioning=(positive, negative),
        )


# The class was called KSamplerStub before; saved prompts/PNG metadata using
# that name keep working through base.NODE_ALIASES.
KSamplerStub = KSampler


@register_node("KSampler (Image)")
class KSamplerImage(NodeBase):
    """img2img: re-samples an existing IMAGE (Load Image, or another
    sampler's output for a "hires fix": KSampler -> Resize -> this at ~0.4
    denoise). `denoise` is diffusers' `strength`: 0 keeps the image, 1
    ignores it; only about steps x denoise steps actually run. Uses an
    img2img pipeline derived from the loaded checkpoint with from_pipe(), so
    it shares the already-loaded weights. A batch IMAGE input is re-sampled
    image by image (all resized to the first one's size), `batch_size`
    variations each. Seeds, sampler, negative prompt and outputs work as in
    KSampler."""

    CATEGORY = "sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    DESCRIPTION = "Re-samples an existing image (img2img / hires fix)."

    @classmethod
    def INPUT_TYPES(cls):
        return _sampler_inputs(
            {
                "image": ("IMAGE",),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        )

    @classmethod
    def IS_CHANGED(cls, **inputs):
        return _sampler_is_changed(**inputs)

    def sample(
        self,
        model,
        positive,
        image,
        denoise=0.6,
        sampler="euler",
        seed=-1,
        steps=20,
        cfg=7.0,
        batch_size=1,
        negative=None,
        scheduler="normal",
        unique_id=None,
        on_progress=None,
    ):
        sources = as_image_list(image)
        runs = _wildcard_runs(positive, negative, seed, len(sources) * int(batch_size))
        if runs:
            base_seed, items = runs
            return _merge_runs(base_seed, [
                self.sample(model, item_positive, [sources[index // int(batch_size)]], denoise, sampler, item_seed,
                            steps, cfg, 1, item_negative, scheduler, unique_id,
                            _run_progress(on_progress, index, len(items)))
                for index, (item_seed, item_positive, item_negative) in enumerate(items)
            ])
        effective_steps = _denoise_steps(steps, denoise)
        controls = controlnet.controls_of(positive)
        if controls:
            pipe = controlnet.control_pipeline(model["pipeline"], "img2img", controls)
        else:
            pipe = derived_pipeline(model["pipeline"], "img2img")
        images, (width, height) = _source_images(image)
        call_kwargs = {
            **_prompt_kwargs(model, pipe, positive, negative, len(images)),
            "image": images,
            "strength": float(denoise),
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg),
            "num_images_per_prompt": int(batch_size),
        }
        if controls:
            call_kwargs.update(controlnet.control_kwargs("img2img", controls, (width, height)))
        return _run_diffusion(
            pipe,
            sampler,
            seed,
            call_kwargs,
            width,
            height,
            batch_size,
            effective_steps,
            on_progress,
            images_out=len(images) * int(batch_size),
            schedule=scheduler,
            model=model,
            node_id=unique_id,
            conditioning=(positive, negative),
        )


@register_node("KSampler (Inpaint)")
class KSamplerInpaint(NodeBase):
    """Inpainting: re-samples only the white area of `mask` (a MASK from
    Load Image's alpha channel or Load Mask), keeping the rest of `image`.
    Works with regular checkpoints (diffusers blends the latents outside the
    mask); dedicated inpainting checkpoints give cleaner seams. A single
    mask applies to every image of a batch; it's resized to the image size.
    Uses an inpainting pipeline derived with from_pipe() (no extra weights).
    `denoise` 1.0 repaints the masked area from scratch."""

    CATEGORY = "sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    DESCRIPTION = "Repaints the white area of a mask."

    @classmethod
    def INPUT_TYPES(cls):
        return _sampler_inputs(
            {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        )

    @classmethod
    def IS_CHANGED(cls, **inputs):
        return _sampler_is_changed(**inputs)

    def sample(
        self,
        model,
        positive,
        image,
        mask,
        denoise=1.0,
        sampler="euler",
        seed=-1,
        steps=20,
        cfg=7.0,
        batch_size=1,
        negative=None,
        scheduler="normal",
        unique_id=None,
        on_progress=None,
    ):
        sources = as_image_list(image)
        runs = _wildcard_runs(positive, negative, seed, len(sources) * int(batch_size))
        if runs:
            base_seed, items = runs
            source_masks = as_image_list(mask)
            return _merge_runs(base_seed, [
                self.sample(model, item_positive, [sources[index // int(batch_size)]],
                            [source_masks[0 if len(source_masks) == 1 else index // int(batch_size)]],
                            denoise, sampler, item_seed, steps, cfg, 1, item_negative, scheduler, unique_id,
                            _run_progress(on_progress, index, len(items)))
                for index, (item_seed, item_positive, item_negative) in enumerate(items)
            ])
        effective_steps = _denoise_steps(steps, denoise)
        controls = controlnet.controls_of(positive)
        if controls:
            pipe = controlnet.control_pipeline(model["pipeline"], "inpaint", controls)
        else:
            pipe = derived_pipeline(model["pipeline"], "inpaint")
        images, (width, height) = _source_images(image)
        masks = [m.convert("L") for m in as_image_list(mask)]
        if len(masks) == 1:
            masks = masks * len(images)
        elif len(masks) != len(images):
            raise ValueError(f"got {len(masks)} masks for {len(images)} images - use one mask or one per image")
        masks = [m if m.size == (width, height) else m.resize((width, height), Image.BILINEAR) for m in masks]
        call_kwargs = {
            **_prompt_kwargs(model, pipe, positive, negative, len(images)),
            "image": images,
            "mask_image": masks,
            "width": width,
            "height": height,
            "strength": float(denoise),
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg),
            "num_images_per_prompt": int(batch_size),
        }
        if controls:
            call_kwargs.update(controlnet.control_kwargs("inpaint", controls, (width, height)))
        return _run_diffusion(
            pipe,
            sampler,
            seed,
            call_kwargs,
            width,
            height,
            batch_size,
            effective_steps,
            on_progress,
            images_out=len(images) * int(batch_size),
            schedule=scheduler,
            model=model,
            node_id=unique_id,
            conditioning=(positive, negative),
        )
