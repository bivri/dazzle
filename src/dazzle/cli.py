"""`dazzle` command line: protect image files, export the generator to ONNX."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dazzle.generator import DazzleGenerator, levels_to_epsilon
from dazzle.protect import protect_image

DEFAULT_MODEL = "bivri/dazzle-generator"


def _load(args) -> DazzleGenerator:
    return DazzleGenerator.from_pretrained(args.model, revision=args.revision).to(args.device).eval()


def cmd_protect(args) -> None:
    gen = _load(args)
    eps = None if args.epsilon_levels is None else levels_to_epsilon(args.epsilon_levels)
    args.out.mkdir(parents=True, exist_ok=True)
    for src in args.images:
        dst = args.out / (src.stem + ".png")
        protect_image(Image.open(src), gen, eps).save(dst)
        print(dst)


def cmd_export_onnx(args) -> None:
    gen = _load(args).cpu()
    h = torch.export.Dim("height", min=1) * 8
    w = torch.export.Dim("width", min=1) * 8
    torch.onnx.export(
        gen,
        (torch.zeros(1, 3, 512, 512),),
        str(args.out),
        input_names=["image"],
        output_names=["delta"],
        dynamic_shapes={"x": {2: h, 3: w}},
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    print(args.out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dazzle", description="Protective patterns against unauthorized style mimicry.")
    sub = p.add_subparsers(required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", default=DEFAULT_MODEL, help="Hub repo id or local save_pretrained directory")
        sp.add_argument("--revision", help="Hub revision (tag or commit)")
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    sp = sub.add_parser("protect", help="write protected PNG copies of images")
    sp.add_argument("images", nargs="+", type=Path)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--epsilon-levels", type=float, help="pattern bound in 8-bit levels; default from the model")
    common(sp)
    sp.set_defaults(fn=cmd_protect)

    sp = sub.add_parser("export-onnx", help="export the generator (input image in [-1, 1], output delta)")
    sp.add_argument("--out", type=Path, required=True)
    common(sp)
    sp.set_defaults(fn=cmd_export_onnx)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
