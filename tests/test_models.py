"""Model discovery, architecture detection, downloads, precision and the
VRAM-budgeted pipeline cache (report items B4, B13, B19, P2, P4, P5)."""

import json

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.nodes import io_nodes, models, runtime

BASE = "http://127.0.0.1:8288"
GB = models.GB


# ---- discovery (B13) ----


def test_list_models_finds_files_and_diffusers_folders(isolated_data_dirs, monkeypatch):
    monkeypatch.setattr(models, "list_cached_hub_pipelines", lambda: ["someone/cached-pipe"])
    ckpts = isolated_data_dirs["checkpoints"]
    (ckpts / "sub").mkdir(parents=True)
    (ckpts / "a.safetensors").write_bytes(b"x")
    (ckpts / "sub" / "b.ckpt").write_bytes(b"x")
    (ckpts / "notes.txt").write_text("ignored")
    (ckpts / "my-diffusers" / "unet").mkdir(parents=True)
    (ckpts / "my-diffusers" / "model_index.json").write_text("{}")
    (ckpts / "my-diffusers" / "unet" / "inner.safetensors").write_bytes(b"x")
    names = models.list_models("checkpoints")
    assert names[:3] == ["a.safetensors", "my-diffusers", "sub/b.ckpt"]
    assert "someone/cached-pipe" in names and models.DEFAULT_CHECKPOINT in names
    assert "my-diffusers/unet/inner.safetensors" not in names

    loras = isolated_data_dirs["loras"]
    loras.mkdir(parents=True)
    (loras / "style.safetensors").write_bytes(b"x")
    assert models.list_models("loras") == ["style.safetensors"]
    with pytest.raises(KeyError):
        models.list_models("nope")


def test_models_endpoint_and_checkpoint_combo(isolated_data_dirs, monkeypatch):
    monkeypatch.setattr(models, "list_cached_hub_pipelines", lambda: [])
    isolated_data_dirs["checkpoints"].mkdir(parents=True)
    (isolated_data_dirs["checkpoints"] / "local.safetensors").write_bytes(b"x")
    client = TestClient(main.app, base_url=BASE)
    assert client.get("/api/models/checkpoints").json() == ["local.safetensors", models.DEFAULT_CHECKPOINT]
    assert client.get("/api/models/loras").json() == []
    assert client.get("/api/models/vae").json() == []  # a split-file kind now (B26)
    assert client.get("/api/models/nope").status_code == 404
    spec = client.get("/api/nodes").json()["LoadCheckpoint"]["input_types"]
    choices, config = spec["required"]["ckpt_name"]
    assert choices == ["local.safetensors", models.DEFAULT_CHECKPOINT]
    assert config["allow_custom"] is True and config["default"] == models.DEFAULT_CHECKPOINT
    assert spec["optional"]["dtype"][0] == ["auto", "fp16", "bf16", "fp32"]


def test_checkpoint_combo_still_accepts_any_hub_id():
    from backend.engine.graph import PromptValidationError, validate_prompt

    graph = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": "someone/not-in-the-list"}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
        "img": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#000"}},
        "s": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["img", 0],
                "sampler": "euler",
                "seed": 1,
                "steps": 2,
                "cfg": 1.0,
                "batch_size": 1,
            },
        },
        "out": {"class_type": "PreviewImage", "inputs": {"image": ["s", 0]}},
    }
    assert validate_prompt(graph)["ckpt"]["inputs"]["ckpt_name"] == "someone/not-in-the-list"
    graph["ckpt"]["inputs"]["ckpt_name"] = "   "
    with pytest.raises(PromptValidationError, match="empty"):
        validate_prompt(graph)


def test_local_model_path_refuses_traversal(isolated_data_dirs):
    assert models.local_model_path("checkpoints", "../../etc/passwd") is None
    assert models.local_model_path("checkpoints", "") is None


# ---- architecture (B4) ----


@pytest.mark.parametrize(
    "model_type,arch,cls",
    [
        ("v1", "sd", "StableDiffusionPipeline"),
        ("v2", "sd", "StableDiffusionPipeline"),
        ("xl_base", "sdxl", "StableDiffusionXLPipeline"),
        ("xl_refiner", "sdxl", "StableDiffusionXLImg2ImgPipeline"),
        ("sd35_medium", "sd3", "StableDiffusion3Pipeline"),
        ("flux-dev", "flux", "FluxPipeline"),
    ],
)
def test_single_file_types_map_to_pipeline_classes(model_type, arch, cls):
    assert models.arch_from_single_file_type(model_type) == (arch, cls)


def test_unsupported_single_file_type_is_a_clear_error():
    with pytest.raises(ValueError, match="unsupported checkpoint type 'wan-t2v-14B'"):
        models.arch_from_single_file_type("wan-t2v-14B")


@pytest.mark.parametrize(
    "class_name,arch",
    [
        ("StableDiffusionPipeline", "sd"),
        ("StableDiffusionXLPipeline", "sdxl"),
        ("StableDiffusion3Pipeline", "sd3"),
        ("FluxPipeline", "flux"),
        ("KandinskyPipeline", "other"),
    ],
)
def test_pipeline_class_names_map_to_architectures(class_name, arch):
    assert models.arch_from_pipeline_class(class_name) == arch


def test_single_safetensors_file_is_detected_from_its_keys(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    path = tmp_path / "xl.safetensors"
    # The key diffusers uses to recognise an SDXL base checkpoint.
    save_file({"conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias": torch.zeros(2)}, str(path))
    assert models.detect_single_file_arch(path) == ("sdxl", "StableDiffusionXLPipeline")


# ---- precision (P2) ----


@pytest.mark.parametrize(
    "option,arch,device,bf16,expected",
    [
        ("auto", "sd", "cuda", True, "fp16"),
        ("auto", "sdxl", "cuda", True, "fp16"),
        ("auto", "sd3", "cuda", True, "bf16"),
        ("auto", "flux", "cuda", False, "fp16"),
        ("bf16", "sd", "cuda", True, "bf16"),
        ("bf16", "sd", "cuda", False, "fp16"),
        ("fp32", "sd", "cuda", True, "fp32"),
        ("auto", "sd", "cpu", False, "fp32"),
        ("fp16", "sd", "cpu", False, "fp32"),
        ("bf16", "sd", "cpu", False, "bf16"),
        ("garbage", "sd", "cuda", True, "fp16"),
    ],
)
def test_dtype_resolution(option, arch, device, bf16, expected):
    assert models.resolve_dtype_name(option, arch, device, bf16) == expected


def test_new_diffusers_gets_the_dtype_keyword():
    # diffusers 0.40 deprecates torch_dtype= (report item T6).
    import diffusers

    major, minor = (int(p) for p in diffusers.__version__.split(".")[:2])
    expected = "dtype" if (major, minor) >= (0, 40) else "torch_dtype"
    assert models.dtype_kwarg_name() == expected


# ---- downloads (B19) ----


def test_select_pipeline_files_mirrors_from_pretrained():
    files = [
        "model_index.json",
        "README.md",
        "sd_turbo.safetensors",  # a root single-file copy: not needed
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.bin",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",  # only a variant: kept
        "text_encoder/model.safetensors",
        "text_encoder/flax_model.msgpack",
        "tokenizer/vocab.json",
        "samples/cat.png",  # not a component
    ]
    components = ["text_encoder", "tokenizer", "unet", "vae"]
    assert models.select_pipeline_files(files, components) == [
        "model_index.json",
        "text_encoder/model.safetensors",
        "tokenizer/vocab.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ]


def test_pipeline_components_skips_null_and_private_entries():
    index = {"_class_name": "X", "unet": ["diffusers", "UNet2DConditionModel"], "safety_checker": [None, None]}
    assert models.pipeline_components(index) == ["unet"]


def test_uncached_hub_model_offline_is_a_clear_error(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(models, "cached_snapshot", lambda repo: None)

    def offline(*args, **kwargs):
        raise huggingface_hub.errors.LocalEntryNotFoundError("no connection")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(RuntimeError, match="isn't downloaded yet and the Hugging Face Hub can't be reached") as info:
        models.ensure_hub_snapshot("someone/model")
    assert isinstance(info.value.__cause__, huggingface_hub.errors.LocalEntryNotFoundError)


def test_hub_download_reports_byte_progress(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(models, "cached_snapshot", lambda repo: None)
    index = tmp_path / "model_index.json"
    index.write_text(json.dumps({"unet": ["diffusers", "UNet2DConditionModel"]}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: str(index))
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo: ["model_index.json", "unet/w.safetensors"])
    seen = {}

    def fake_snapshot(repo_id, allow_patterns=None, tqdm_class=None):
        seen["patterns"] = allow_patterns
        bar = tqdm_class(total=1000, unit="B", desc="Downloading bytes")
        bar.update(400)
        bar.update(600)
        bar.close()
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    progress = []
    path = models.ensure_hub_snapshot("someone/model", lambda done, total, **info: progress.append((done, total, info)))
    assert path == tmp_path
    assert seen["patterns"] == ["model_index.json", "unet/w.safetensors"]
    assert progress[-1] == (1000, 1000, {"unit": "bytes"})


def test_cached_snapshot_needs_every_component(tmp_path, monkeypatch):
    import huggingface_hub

    snap = tmp_path / "snap"
    (snap / "unet").mkdir(parents=True)
    (snap / "model_index.json").write_text(json.dumps({"unet": ["d", "U"], "vae": ["d", "V"]}))
    (snap / "unet" / "w.safetensors").write_bytes(b"x")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: str(snap / name))
    assert models.cached_snapshot("r") is None, "vae is missing"
    (snap / "vae").mkdir()
    (snap / "vae" / "w.safetensors").write_bytes(b"x")
    assert models.cached_snapshot("r") == snap


# ---- the VRAM-budgeted cache (P4) and offload (P5) ----


class _FakePipe:
    def __init__(self, name):
        self.name = name
        self.device = "cpu"
        self.offloaded = False

    def to(self, device):
        self.device = device
        return self

    def enable_model_cpu_offload(self):
        self.offloaded = True


@pytest.fixture
def fake_cache(monkeypatch):
    """A fresh PipelineCache whose loads are fakes of a given size, on a
    pretend 16 GB GPU."""
    sizes = {}
    loads = []
    state = {"free": 16 * GB, "cuda": True}

    def fake_load(name, device, dtype_option, on_progress=None):
        loads.append((name, dtype_option))
        return models.CacheEntry(pipe=_FakePipe(name), nbytes=sizes.get(name, 2 * GB), placement="cpu", dtype="fp16")

    def used():
        return sum(e.nbytes for e in cache.entries.values() if e.placement == "gpu")

    cache = models.PipelineCache()
    monkeypatch.setattr(models, "_load_entry", fake_load)
    monkeypatch.setattr(runtime, "cuda_available", lambda: state["cuda"])
    monkeypatch.setattr(runtime, "device_memory", lambda: (state["free"] - used(), 16 * GB) if state["cuda"] else None)
    monkeypatch.setattr(runtime, "free_memory", lambda: None)
    monkeypatch.setenv("PIXELFLOW_VRAM_RESERVE_GB", "4")
    monkeypatch.setenv("PIXELFLOW_RAM_CACHE_GB", "8")
    monkeypatch.setattr(models, "MAX_CACHED_PIPELINES", 4)
    return cache, sizes, loads, state


def test_pipelines_share_the_gpu_while_they_fit(fake_cache):
    cache, sizes, loads, _ = fake_cache
    a = cache.get("a").pipe
    b = cache.get("b").pipe
    assert a.device == b.device == "cuda"
    assert cache.get("a").pipe is a and loads == [("a", "auto"), ("b", "auto")], "a cache hit, no reload"


def test_lru_pipelines_move_to_cpu_ram_when_vram_runs_out(fake_cache):
    cache, sizes, loads, _ = fake_cache
    sizes.update(a=5 * GB, b=5 * GB, c=5 * GB)  # budget: 16 - 4 reserve = 12 GB
    a = cache.get("a").pipe
    cache.get("b")
    cache.get("c")
    placements = {key[0]: e.placement for key, e in cache.entries.items()}
    assert placements == {"a": "cpu", "b": "gpu", "c": "gpu"}
    assert a.device == "cpu"
    # Re-using "a" brings it back from RAM (no reload) and demotes "b".
    assert cache.get("a").pipe is a and a.device == "cuda"
    assert {key[0]: e.placement for key, e in cache.entries.items()} == {"b": "cpu", "c": "gpu", "a": "gpu"}
    assert [name for name, _ in loads] == ["a", "b", "c"]


def test_cpu_tier_is_bounded_by_its_ram_budget(fake_cache, monkeypatch):
    cache, sizes, loads, _ = fake_cache
    monkeypatch.setenv("PIXELFLOW_RAM_CACHE_GB", "6")
    sizes.update(a=5 * GB, b=5 * GB, c=5 * GB, d=5 * GB)
    for name in "abcd":
        cache.get(name)
    assert [key[0] for key in cache.entries] == ["b", "c", "d"], "only one 5 GB pipeline fits the 6 GB RAM tier"
    assert cache.entries[("b", "cuda", "auto")].placement == "cpu"


def test_count_cap_and_dtype_in_the_cache_key(fake_cache, monkeypatch):
    cache, _, loads, _ = fake_cache
    monkeypatch.setattr(models, "MAX_CACHED_PIPELINES", 2)
    cache.get("a", "fp16")
    cache.get("a", "bf16")
    assert len(loads) == 2, "a different precision is a different pipeline"
    cache.get("b")
    assert [(k[0], k[2]) for k in cache.entries] == [("a", "bf16"), ("b", "auto")]


def test_model_larger_than_the_gpu_budget_uses_cpu_offload(fake_cache):
    cache, sizes, _, _ = fake_cache
    sizes["flux"] = 24 * GB
    entry = cache.get("flux")
    assert entry.placement == "offload" and entry.pipe.offloaded and entry.pipe.device == "cpu"


def test_cpu_only_machine_keeps_pipelines_in_ram(fake_cache):
    cache, _, _, state = fake_cache
    state["cuda"] = False
    entry = cache.get("a")
    assert entry.placement == "cpu" and entry.pipe.device == "cpu"
    assert list(cache.entries) == [("a", "cpu", "auto")]


def test_free_endpoint_unloads_everything_and_system_reports_it(fake_cache, monkeypatch):
    cache, _, _, _ = fake_cache
    monkeypatch.setattr(models, "PIPELINES", cache)
    cache.get("a")
    client = TestClient(main.app, base_url=BASE)
    system = client.get("/api/system").json()
    assert system["app"] == "pixelflow" and "torch" in system and "cuda" in system
    assert system["loaded_models"][0]["name"] == "a"
    assert {"data_dir", "models_dir", "log_dir", "version"} <= set(system)
    assert client.post("/api/free").json() == {"unloaded": 1}
    assert cache.entries == {}


def test_load_checkpoint_node_reports_arch_and_dtype(fake_cache, monkeypatch):
    cache, _, _, _ = fake_cache
    monkeypatch.setattr(models, "PIPELINES", cache)
    (model,) = io_nodes.LoadCheckpoint().load("a", dtype="fp16")
    assert model["name"] == "a" and model["dtype"] == "fp16" and model["loras"] == []
    assert model["pipeline"].device == "cuda"
