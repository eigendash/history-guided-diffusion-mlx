"""Compare history lengths, guidance variants and long rollouts on held-out clips.

    python scripts/evaluate.py            # needs artifacts/dfot.safetensors and bd.safetensors

Prints three tables and writes them to artifacts/results.md, plus a rollout strip
artifacts/rollout.png (rows: truth, then a few variants).
"""

import argparse
import struct
import sys
import zlib
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgd.data import BallWorld, from_model, to_model  # noqa: E402
from hgd.guidance import fractional, plain, rollout, sample, temporal, vanilla  # noqa: E402
from hgd.metrics import chunk_metrics, horizon_metrics  # noqa: E402
from hgd.model import Config, VideoDiT  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"
T = 8


def set_art(path):
    global ART
    ART = Path(path)


def load(name):
    model = VideoDiT(Config())
    model.load_weights(str(ART / f"{name}.safetensors"))
    mx.eval(model.parameters())
    return model


def generate(model, truth, hist, guidance, steps, seed=0):
    mx.random.seed(seed)
    clean = mx.array(to_model(truth))
    keep = (mx.arange(truth.shape[1]) < hist)[None, :, None, None, None]
    clean = mx.where(keep, clean, 0.0)
    out = sample(model, clean, list(range(hist, truth.shape[1])), guidance, steps)
    return from_model(out)


def fmt(x, digits=3):
    return "n/a" if x != x else f"{x:.{digits}f}"


def chunk_table(rows):
    lines = ["| model | history | guidance | pixel MSE | centroid err (px) | ball seen | motion ratio |", "|---|---:|---|---:|---:|---:|---:|"]
    for name, hist, label, m in rows:
        lines.append(
            f"| {name} | {hist} | {label} | {m['mse']:.4f} | {fmt(m['centroid'], 2)} | {m['visible']:.3f} | {m['motion']:.2f} |"
        )
    return "\n".join(lines)


def horizon_table(rows):
    bins = [f"{lo}-{hi - 1}" for lo, hi in (r["frames"] for r in rows[0][2])]
    head = "| model | guidance | " + " | ".join(f"err {b}" for b in bins) + " | ball seen | mass ratio (last bin) |"
    lines = [head, "|---|---|" + "---:|" * (len(bins) + 2)]
    for name, label, hs in rows:
        errs = " | ".join(fmt(h["centroid"], 2) for h in hs)
        seen = float(np.mean([h["visible"] for h in hs]))
        lines.append(f"| {name} | {label} | {errs} | {seen:.3f} | {fmt(hs[-1]['mass'], 2)} |")
    return "\n".join(lines)


def write_png(path, grid):
    img = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
    raw = b"".join(b"\x00" + row.tobytes() for row in img)

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    head = struct.pack(">IIBBBBB", img.shape[1], img.shape[0], 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=128)
    ap.add_argument("--rollout-clips", type=int, default=64)
    ap.add_argument("--rollout-frames", type=int, default=36)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--art", default=None, help="directory holding dfot.safetensors and bd.safetensors")
    args = ap.parse_args()
    if args.art:
        set_art(args.art)

    dfot, bd = load("dfot"), load("bd")
    world = BallWorld(seed=12345)
    truth, _ = world.sample(args.clips, T)
    out = []

    # 1. How many history frames, no guidance.
    rows = []
    for name, model in (("DFoT", dfot), ("binary dropout", bd)):
        for hist in (2, 4, 6):
            gen = generate(model, truth, hist, plain(T, set(range(hist))), args.steps)
            rows.append((name, hist, "none", chunk_metrics(gen, truth, hist)))
    out += ["### History length, no guidance", chunk_table(rows)]
    print(out[-2], out[-1], sep="\n", flush=True)

    # 2. Guidance variants.
    rows = []
    for hist in (4, 6):
        h, short = set(range(hist)), {hist - 2, hist - 1}
        variants = {
            "none": plain(T, h),
            "vanilla w=1.5": vanilla(T, h, 1.5),
            "vanilla w=3": vanilla(T, h, 3.0),
            "fractional w=1.5 k=0.5": fractional(T, h, 1.5, 0.5),
            "fractional w=3 k=0.5": fractional(T, h, 3.0, 0.5),
            "temporal long+short": temporal(T, [(h, 0.5), (short, 0.5)]),
            "temporal long+short w=1.5": temporal(T, [(h, 0.75), (short, 0.75)]),
        }
        for label, guidance in variants.items():
            gen = generate(dfot, truth, hist, guidance, args.steps)
            rows.append(("DFoT", hist, label, chunk_metrics(gen, truth, hist)))
        for label, guidance in (("none", plain(T, h)), ("vanilla w=1.5", vanilla(T, h, 1.5)), ("vanilla w=3", vanilla(T, h, 3.0))):
            gen = generate(bd, truth, hist, guidance, args.steps)
            rows.append(("binary dropout", hist, label, chunk_metrics(gen, truth, hist)))
    out += ["", "### Guidance variants", chunk_table(rows)]
    print(*out[-2:], sep="\n", flush=True)

    # 3. Long rollout with a sliding window (history 4, four new frames per round).
    world = BallWorld(seed=777)
    long_truth, _ = world.sample(args.rollout_clips, args.rollout_frames)
    start = mx.array(to_model(long_truth[:, :4]))
    hist = 4
    variants = [
        ("DFoT", dfot, "none", lambda n, h: plain(n, set(h))),
        ("DFoT", dfot, "vanilla w=2", lambda n, h: vanilla(n, set(h), 2.0)),
        ("DFoT", dfot, "fractional w=2 k=0.5", lambda n, h: fractional(n, set(h), 2.0, 0.5)),
        ("DFoT", dfot, "temporal long+short", lambda n, h: temporal(n, [(set(h), 0.5), ({h[-2], h[-1]}, 0.5)])),
        ("binary dropout", bd, "none", lambda n, h: plain(n, set(h))),
        ("binary dropout", bd, "vanilla w=2", lambda n, h: vanilla(n, set(h), 2.0)),
    ]
    rows, strips = [], [long_truth[0]]
    for name, model, label, build in variants:
        mx.random.seed(0)
        video = from_model(rollout(model, start, args.rollout_frames, T, hist, build, args.steps))
        rows.append((name, label, horizon_metrics(video, long_truth, hist, 8)))
        if name == "DFoT" and label in ("none", "temporal long+short"):
            strips.append(video[0])
    out += ["", f"### Rollout to {args.rollout_frames} frames (centroid error in pixels by frame range)", horizon_table(rows)]
    print(*out[-2:], sep="\n", flush=True)

    ART.mkdir(exist_ok=True)
    (ART / "results.md").write_text("\n".join(out) + "\n")
    cell = lambda v: np.concatenate(list(v), axis=1)  # noqa: E731
    write_png(ART / "rollout.png", np.concatenate([cell(s) for s in strips], axis=0))
    print(f"\nwrote {ART / 'results.md'} and {ART / 'rollout.png'}")


if __name__ == "__main__":
    main()
