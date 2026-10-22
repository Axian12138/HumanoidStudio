"""Receding-horizon sampling of bounded position targets, with no state resets
inside a rollout and no external forces. This is a SPIDER-inspired baseline,
not a reproduction of its virtual-contact curriculum.
"""

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    horizon: int = 50
    advance: int = 5
    samples: int = 128
    iterations: int = 8
    knot_spacing: int = 5
    sigma: float = 0.12
    temperature: float = 0.05
    seed: int = 0

    def __post_init__(self):
        for name in ("horizon", "advance", "samples", "iterations", "knot_spacing"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.advance > self.horizon or self.samples < 3:
            raise ValueError("advance must not exceed horizon; samples must be >= 3")
        if not np.isfinite([self.sigma, self.temperature]).all() or min(self.sigma, self.temperature) <= 0:
            raise ValueError("sigma and temperature must be finite and positive")


def weighted_update(actions, costs, temperature):
    costs = np.asarray(costs)
    finite = np.isfinite(costs)
    if not finite.any():
        raise RuntimeError("all sampled rollouts are nonfinite")
    weights = np.zeros(len(costs))
    weights[finite] = np.exp(-(costs[finite] - costs[finite].min()) / temperature)
    weights /= weights.sum()
    return np.einsum("n,ntu->tu", weights, actions)


def optimize(backend, reference_controls, config, progress=None):
    """Backend evaluates candidates from its current state and commits a prefix.

    Candidate 0 preserves the incumbent; candidate 1 evaluates the previous
    weighted update. Only an actually evaluated candidate is executed.
    """
    cfg = config
    rng = np.random.default_rng(cfg.seed)
    controls = np.asarray(reference_controls, dtype=np.float64).copy()
    records = []
    began = time.perf_counter()
    for start in range(0, len(controls), cfg.advance):
        end = min(start + cfg.horizon, len(controls))
        mean = controls[start:end].copy()
        best = mean.copy()
        best_cost = np.inf
        h = end - start
        knots = np.arange(0, h + cfg.knot_spacing, cfg.knot_spacing)
        for iteration in range(cfg.iterations):
            noise = rng.normal(size=(cfg.samples, len(knots), controls.shape[1]))
            smooth = np.empty((cfg.samples, h, controls.shape[1]))
            for n in range(cfg.samples):
                for j in range(controls.shape[1]):
                    smooth[n, :, j] = np.interp(np.arange(h), knots, noise[n, :, j])
            scale = cfg.sigma * (0.25 ** (iteration / max(cfg.iterations - 1, 1)))
            candidates = np.clip(
                mean + scale * smooth * getattr(backend, "noise_scale", 1.0), backend.lower, backend.upper
            )
            candidates[0] = best
            candidates[1] = mean
            costs = backend.evaluate(candidates, start)
            winner = int(np.nanargmin(np.where(np.isfinite(costs), costs, np.inf)))
            if costs[winner] < best_cost:
                best, best_cost = candidates[winner].copy(), float(costs[winner])
            mean = weighted_update(candidates, costs, cfg.temperature)
        controls[start:end] = best
        count = min(cfg.advance, len(controls) - start)
        backend.commit(best[:count])
        record = {"frame": start + count, "horizon_cost": best_cost, "wall_s": time.perf_counter() - began}
        records.append(record)
        if progress:
            progress(record)
    return controls, records
