"""Render the fixed evaluation prompts with a mimicry LoRA (10 subjects x 4 seeds, 1024^2)."""

from pathlib import Path

from dazzle.evaluation.lora import BASE_MODEL
from dazzle.workspace import Workspace

SUBJECTS = (
    "a girl waiting for a train on a rainy platform",
    "a boy repairing a bicycle in a garage",
    "a girl holding a bouquet in a wheat field",
    "a boy reading under a streetlamp at night",
    "a girl riding a horse along a river",
    "a boy cooking ramen at a street stall",
    "a girl looking out of a classroom window",
    "a knight boy standing guard at a castle gate",
    "a girl feeding pigeons in a town square",
    "a boy climbing a tree to save a kite",
)
SEEDS = (1, 2, 3, 4)
NEGATIVE = "lowres, worst quality, bad anatomy, text, watermark"


def render_names() -> list[str]:
    return [f"p{i:02d}_s{s}.png" for i in range(len(SUBJECTS)) for s in SEEDS]


def render(ws: Workspace, jobs: list[tuple[Path, str, Path]]) -> None:
    """jobs: (lora weights, style, output dir). Existing images are kept."""
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_single_file(str(ws.models / BASE_MODEL), torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    for i, (weights, style, out) in enumerate(jobs):
        out.mkdir(parents=True, exist_ok=True)
        pipe.load_lora_weights(str(weights.parent), weight_name=weights.name, adapter_name=f"j{i}")
        pipe.set_adapters([f"j{i}"], adapter_weights=[1.0])
        for pi, subject in enumerate(SUBJECTS):
            for seed in SEEDS:
                dst = out / f"p{pi:02d}_s{seed}.png"
                if dst.exists():
                    continue
                pipe(
                    prompt=f"zq{style}, {subject}",
                    negative_prompt=NEGATIVE,
                    width=1024,
                    height=1024,
                    num_inference_steps=28,
                    guidance_scale=6.0,
                    generator=torch.Generator("cuda").manual_seed(seed),
                ).images[0].save(dst)
        pipe.unload_lora_weights()
