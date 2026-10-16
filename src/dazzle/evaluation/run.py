"""Evaluate protected training sets end to end and write runs/<run-id>/report.json.

    python -m dazzle.evaluation.run --run-id baseline-eval --generator bivri/dazzle-generator \\
        --filters jpeg85 vae resize0.5
    python -m dazzle.evaluation.run --run-id proxy-check --import-set decoy=/abs/path --proxy-only

Stages: build sets (generator output, then each transformation filter) -> quality (LPIPS, PSNR)
-> proxy -> LoRA per style -> renders -> E/F. Every stage skips work whose output exists, so an
interrupted run resumes with the same command.

Run layout: sets/<variant>/<style>/, loras/<variant>/, renders/<variant>/<style>/, report.json,
manifest.json.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dazzle
from dazzle.workspace import STYLES, Workspace


def code_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = os.environ.get("DAZZLE_GIT_SHA", "unknown")
    return f"{dazzle.__version__}+{sha}"


def build_protected(ws: Workspace, run: Path, args, device: str) -> list[str]:
    from dazzle.evaluation import filters
    from dazzle.generator import DazzleGenerator, levels_to_epsilon
    from dazzle.protect import protect_image

    variants = ["protected"] + [f"protected+{f}" for f in args.filters]
    gen = DazzleGenerator.from_pretrained(args.generator, revision=args.revision).to(device).eval()
    eps = None if args.epsilon_levels is None else levels_to_epsilon(args.epsilon_levels)
    fns = {f: filters.build(f, device) for f in args.filters}
    for style in args.styles:
        for src in ws.train_images(style):
            outs = {v: run / "sets" / v / style / src.name for v in variants}
            if all(p.exists() for p in outs.values()):
                continue
            prot = protect_image(Image.open(src), gen, eps)
            for v, dst in outs.items():
                dst.parent.mkdir(parents=True, exist_ok=True)
                (prot if v == "protected" else fns[v.split("+", 1)[1]](prot)).save(dst)
                if src.with_suffix(".txt").exists():
                    shutil.copy(src.with_suffix(".txt"), dst.with_suffix(".txt"))
    return variants


def import_sets(ws: Workspace, run: Path, items: list[str], styles: list[str]) -> list[str]:
    names = []
    for item in items:
        name, path = item.split("=", 1)
        for style in styles:
            dst = run / "sets" / name / style
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(Path(path) / style, dst)
            for src in ws.train_images(style):  # captions come from the clean set
                cap = src.with_suffix(".txt")
                if cap.exists() and not (dst / cap.name).exists():
                    shutil.copy(cap, dst / cap.name)
        names.append(name)
    return names


def quality(ws: Workspace, run: Path, variants: list[str], styles: list[str], device: str) -> dict:
    import lpips

    from dazzle.protect import to_tensor

    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    res = {}
    with torch.inference_mode():
        for v in variants:
            vals = {"lpips": [], "psnr": []}
            for style in styles:
                for src in ws.train_images(style):
                    x = to_tensor(Image.open(src)).to(device)
                    y = to_tensor(Image.open(run / "sets" / v / style / src.name)).to(device)
                    vals["lpips"].append(lp(x, y).item())
                    vals["psnr"].append(10 * torch.log10(4.0 / (x - y).pow(2).mean()).item())
            res[v] = {k: float(np.mean(a)) for k, a in vals.items()}
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--generator", help="Hub repo id or save_pretrained directory")
    p.add_argument("--revision", help="Hub revision (tag or commit) of --generator")
    p.add_argument("--epsilon-levels", type=float, help="override the generator default (8-bit levels)")
    p.add_argument("--filters", nargs="*", default=[], help="e.g. jpeg85 vae resize0.5")
    p.add_argument("--import-set", nargs="*", default=[], metavar="NAME=DIR", help="evaluate existing sets")
    p.add_argument("--styles", nargs="+", default=list(STYLES))
    p.add_argument("--proxy-only", action="store_true", help="stop after quality and proxy")
    p.add_argument("--seed", type=int, default=42, help="LoRA training seed")
    args = p.parse_args()
    if not args.generator and not args.import_set:
        p.error("need --generator or --import-set")

    ws, device = Workspace.from_env(), "cuda" if torch.cuda.is_available() else "cpu"
    run = ws.run(args.run_id)
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"code": code_version(), "args": vars(args), "workspace": str(ws.root)}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))

    variants = build_protected(ws, run, args, device) if args.generator else []
    variants += import_sets(ws, run, args.import_set, args.styles)
    report = {"manifest": manifest, "quality": quality(ws, run, variants, args.styles, device)}

    from dazzle.evaluation.csd import CSD
    from dazzle.evaluation.proxy import specificity_drop

    csd = CSD(device)
    refs = {s: csd.embed(ws.ref_images(s)) for s in STYLES}
    clean = {s: csd.embed(ws.train_images(s)) for s in args.styles}
    report["proxy"] = {
        v: {
            s: specificity_drop(
                clean[s], csd.embed([run / "sets" / v / s / f.name for f in ws.train_images(s)]), refs, s
            )
            for s in args.styles
        }
        for v in variants
    }
    (run / "report.json").write_text(json.dumps(report, indent=2))
    if args.proxy_only:
        print(json.dumps(report["proxy"], indent=2))
        return

    from dazzle.evaluation.lora import train_lora
    from dazzle.evaluation.render import render, render_names
    from dazzle.evaluation.score import effect_ratio

    jobs = []
    for v in variants:
        for s in args.styles:
            weights = train_lora(ws, run / "sets" / v, s, run / "loras" / v, seed=args.seed)
            print(f"lora {v}/{s} done", flush=True)
            jobs.append((weights, s, run / "renders" / v / s))
    render(ws, jobs)

    names = render_names()
    rr = ws.reference_renders
    base = {s: csd.embed([rr / s / "base" / n for n in names]) for s in args.styles}
    clean_lora = {s: csd.embed([rr / s / "lora" / n for n in names]) for s in args.styles}
    report["effect_ratio"] = {
        v: effect_ratio(
            base, clean_lora, {s: csd.embed([run / "renders" / v / s / n for n in names]) for s in args.styles}, refs
        )
        for v in variants
    }
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["effect_ratio"], indent=2))


if __name__ == "__main__":
    main()
