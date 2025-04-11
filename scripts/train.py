"""Train a video diffusion model on the bouncing-ball world.

    python scripts/train.py --mode dfot   # independent per-frame noise levels
    python scripts/train.py --mode bd     # binary-dropout baseline

Checkpoints go to artifacts/<mode>.safetensors.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgd.data import BallWorld, to_model  # noqa: E402
from hgd.diffusion import binary_dropout_levels, independent_levels, training_loss  # noqa: E402
from hgd.model import Config, VideoDiT  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"
LEVELS = {"dfot": independent_levels, "bd": binary_dropout_levels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=LEVELS, default="dfot")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config()
    mx.random.seed(args.seed)
    model = VideoDiT(cfg)
    mx.eval(model.parameters())
    warmup = optim.linear_schedule(0.0, args.lr, 200)
    decay = optim.cosine_decay(args.lr, args.steps - 200, args.lr * 0.05)
    opt = optim.AdamW(learning_rate=optim.join_schedules([warmup, decay], [200]), weight_decay=0.01)
    world = BallWorld(args.seed)
    sample_levels = LEVELS[args.mode]

    def loss_fn(model, x0, levels, weight):
        return training_loss(model, x0, levels, weight)

    step = nn.value_and_grad(model, loss_fn)
    start, running = time.perf_counter(), None
    for i in range(1, args.steps + 1):
        frames, _ = world.sample(args.batch, cfg.frames)
        x0 = mx.array(to_model(frames))
        levels, weight = sample_levels(args.batch, cfg.frames)
        loss, grads = step(model, x0, levels, weight)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        running = float(loss) if running is None else 0.98 * running + 0.02 * float(loss)
        if i % 500 == 0 or i == 1:
            print(f"[{args.mode}] step {i:5d}  loss {running:.4f}  elapsed {time.perf_counter() - start:.0f}s", flush=True)

    ART.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else ART / f"{args.mode}.safetensors"
    model.save_weights(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
