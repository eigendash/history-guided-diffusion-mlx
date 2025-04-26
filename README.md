# history-guided-diffusion-mlx

A small video diffusion model trained with independent noise levels per frame, plus the history guidance schemes that this training makes possible, written for MLX and run on a toy video domain.

The ideas come from *History-Guided Video Diffusion* (ICML 2025, arXiv:2502.06764). This is an independent implementation written from the paper at a scale of about 1M parameters, on 16x16 videos of one bouncing ball. It is not a reproduction of the paper's experiments, and none of the numbers below are comparable to its FVD results.

## The idea in brief

A video model that conditions on history usually treats the history as a separate input: clean frames go through their own encoder, and the generated frames are denoised as a block at one noise level. The paper instead feeds every frame to one transformer and gives each frame its own noise level during training. A clean frame is history, a pure-noise frame is masked out, and anything between is a partly hidden frame. Conditioning on any subset of the frames then comes for free at sampling time: set the levels of the frames you want to condition on to zero and the rest to the current level.

Because masking is just noise, the unconditional score is the model run with all history replaced by noise, and scores for different slices of history can be combined. The paper builds three guidance rules on this. Vanilla guidance is classifier-free guidance with the history as the condition. Fractional guidance adds a term where the history is only partly noised, which keeps its coarse content and leaves fine detail unconstrained. Temporal guidance sums scores conditioned on different subsets of the history, such as the full context and the last few frames.

## What is here

`hgd/diffusion.py` has the cosine noise schedule, per-frame noising, and the two training-time level samplers: independent uniform levels per frame (the paper's recipe) and a binary-dropout baseline with a random-length clean prefix, a shared noise level for the rest, and the whole history masked 15% of the time. The network predicts v rather than the noise. That is affine in the score like noise prediction is, but it avoids dividing by a vanishing alpha near full noise.

`hgd/model.py` is a transformer over all tokens of a clip. Each token is modulated by the noise level of its own frame, attention runs in both directions in time, and positions are fixed sinusoids.

`hgd/guidance.py` expresses every variant as a base score plus weighted differences, `v_masked + sum_i w_i (v_i - v_masked)`, where each term is a vector of per-frame history levels. Plain conditioning, vanilla, fractional and temporal guidance are all lists of such terms. Combining v directly is the same as combining scores, since the weights sum to one. The sampler is DDIM with x0 clipping, and `rollout` extends a clip with a sliding window.

`hgd/data.py` is the toy world: one soft ball moving at constant velocity, bouncing off the walls, with a vertical band across the middle that hides it completely. A single frame does not reveal the velocity, and while the ball is behind the band only older frames say where it is heading.

## Setup of the experiment

Clips are 8 frames. Both models have 4 transformer blocks of width 128 and 4x4 patches, trained for 12,000 steps at batch size 64 (AdamW, peak learning rate 8e-4, cosine decay). Evaluation uses 128 held-out clips for single-chunk prediction and 64 for rollouts, sampled with 30 DDIM steps. Given the first `h` frames, the model fills in the rest and is scored against the true continuation by pixel MSE, by the distance between the detected ball centres where the true ball is visible (a miss costs 8 px; the canvas is 16 px wide, so about 6 px is what random placement gives), by agreement on whether the ball is visible, and by a motion ratio, the mean frame-to-frame change of the sample divided by that of the truth. Values below one indicate a tendency to freeze.

Everything is one training seed and one evaluation set. I did not compute confidence intervals, so differences of a few tenths of a pixel should not be trusted.

### History length

| model | history | pixel MSE | centroid error (px) | ball seen | motion ratio |
|---|---:|---:|---:|---:|---:|
| per-frame levels | 2 | 0.0154 | 3.95 | 0.786 | 0.98 |
| per-frame levels | 4 | 0.0102 | 2.42 | 0.842 | 0.89 |
| per-frame levels | 6 | 0.0054 | 1.09 | 0.902 | 0.80 |
| binary dropout | 2 | 0.0063 | 1.90 | 0.878 | 0.90 |
| binary dropout | 4 | 0.0029 | 0.90 | 0.941 | 0.98 |
| binary dropout | 6 | 0.0014 | 0.52 | 0.965 | 1.01 |

Both models use more history well. The binary-dropout baseline is clearly better at this conditional task: at six history frames its centroid error is half that of the per-frame model. That is the opposite of what the paper reports at scale. A plausible reason is that the baseline spends all its capacity on exactly the task being scored, a clean prefix and one shared noise level, while the per-frame model must also handle every other combination of levels. I did not test that explanation, and a larger model or longer training could change the ordering.

### Guidance

Centroid error in pixels (lower is better), with the motion ratio in brackets.

| model | history | none | vanilla w=1.5 | vanilla w=3 | fractional w=1.5 | fractional w=3 | temporal | temporal w=1.5 |
|---|---:|---|---|---|---|---|---|---|
| per-frame levels | 4 | 2.42 (0.89) | 2.01 (0.89) | 1.77 (0.96) | 2.17 (0.92) | 2.19 (0.97) | 2.45 (0.93) | 2.15 (0.92) |
| per-frame levels | 6 | 1.09 (0.80) | 0.98 (0.79) | 0.95 (0.90) | 1.05 (0.92) | 1.05 (1.02) | 1.33 (0.83) | 1.04 (0.82) |
| binary dropout | 4 | 0.90 (0.98) | 0.68 (1.00) | 0.61 (1.04) | | | | |
| binary dropout | 6 | 0.52 (1.01) | 0.35 (1.00) | 0.33 (1.01) | | | | |

Fractional uses a history noise level of 0.5. Temporal combines the full history with its last two frames, at weight 0.5 each ("temporal") or 0.75 each ("temporal w=1.5").

Vanilla guidance helps both models and helps more as the weight grows, up to the largest weight tried. On the per-frame model at four frames, centroid error falls from 2.42 to 1.77 px. The baseline improves too, from 0.90 to 0.61 px, so in this setting I do not see a guidance advantage for the per-frame model.

Fractional guidance did not improve centroid error beyond what vanilla gave. It did raise the motion ratio of the per-frame model at six frames from 0.80 to 1.02 at weight 3, which fits the paper's claim that fractional guidance counters a tendency toward static videos. Vanilla guidance raised it as well, to 0.90, so the toy does not separate the two cleanly.

Temporal guidance did not help here. The blend at weights 0.5 and 0.5 was equal or worse than plain conditioning, and the stronger version roughly matched vanilla guidance at a similar weight. This is the variant where I would least expect the toy to show anything. The paper's motivation is histories that fall outside the training distribution, and every history in this world is in distribution.

### Long rollouts

Rollouts to 36 frames use a window of 8: condition on the last 4 frames and generate 4 more, repeated. Centroid error by frame range:

| model | guidance | 4-11 | 12-19 | 20-27 | 28-35 | ball seen |
|---|---|---:|---:|---:|---:|---:|
| per-frame levels | none | 3.65 | 6.08 | 6.32 | 5.85 | 0.746 |
| per-frame levels | vanilla w=2 | 3.15 | 5.42 | 6.13 | 6.04 | 0.805 |
| per-frame levels | fractional w=2 | 3.72 | 5.74 | 6.23 | 6.05 | 0.801 |
| per-frame levels | temporal | 3.80 | 5.74 | 6.25 | 5.96 | 0.767 |
| binary dropout | none | 2.07 | 5.33 | 6.62 | 6.64 | 0.759 |
| binary dropout | vanilla w=2 | 1.57 | 4.71 | 5.64 | 6.60 | 0.815 |

This part did not work. By the second block of 8 frames every variant is at or near chance, so errors compound and the models do not keep track of the ball across rounds. Guidance helps in the first rounds and not after. The ball does not fade or blow up (its detected intensity in the last block is 0.77 to 0.90 of the true value for the per-frame model), but it does not go where it should. The paper's long-video results rest on much larger models and training. At this size I do not reproduce them, and I have not tried to tell whether the cause is the model, the 4-frame window, or sampling.

## A failed first attempt

The first version used learned positional embeddings, 2x2 patches, and 4,000 steps at batch size 32. Both models trained to a low loss and produced speckled frames, and when I measured denoising error against the number of clean history frames it was flat from zero to six frames, at every noise level. They had learned what the images look like and ignored the history entirely. Switching to fixed sinusoidal positions, 4x4 patches (4x fewer tokens, so more steps for the same time), a larger batch and a higher learning rate fixed it, and history started to matter within 2,000 steps. I did not isolate which of those changes mattered most.

## Choices where the paper is open

Network output is v, with a cosine schedule. History frames that are partly or fully masked use one noise draw per clip, fixed across sampling steps. Generated frames share one noise level at each sampling step, though the model supports different levels per frame. The binary-dropout baseline masks the history with pure noise, where a conventional implementation would use a null token. The temporal weights and the fractional level were picked once and not tuned. Rollouts use a fixed 4-in, 4-out window.

## Limits

One seed, one dataset size, no error bars. A 16x16 world with one object. No latent space, no text or camera conditioning, no causal variant, and no comparison against anything beyond the binary-dropout baseline defined above. Training is short enough that the per-frame model may simply be undertrained relative to the baseline.

## Running it

```
pip install -e ".[dev]"
pytest
python scripts/train.py --mode dfot
python scripts/train.py --mode bd
python scripts/evaluate.py
```

Checkpoints and tables are written to `artifacts/`. The tests cover the simulator and detector, the forward process and its inversions, the level samplers, the model's non-causality, the algebra of each guidance variant against a stub model, and the sampler against an oracle that knows the true video, for which it should recover the generated frames exactly.
