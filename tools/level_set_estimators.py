"""Compare ways of sampling a super-level set from its live points, by dimension.

    python tools/level_set_estimators.py --dimensions 1 2 5 10 20 40 61 --shape ellipsoid
    python tools/level_set_estimators.py --shape banana

The situation of one nested-sampling iteration, with the truth known. The likelihood is a
Gaussian with unequal widths and a random orientation under a flat prior, so its super-level
set is an ellipsoid (``--shape banana`` bends it, pair of axes by pair, with a volume-keeping
shear). ``--live`` points are drawn exactly uniformly on the set: what the survivors of an
iteration are. Each method sees only those points and must produce new uniform points on the
set. Two numbers per method:

- **lost**: the share of the level set the method can never reach (measured on fresh exact
  points). This is the bias: the volume bookkeeping assumes the whole set is sampled.
- **calls**: likelihood evaluations per new point (a candidate costs one call when it has to
  be tested against the threshold; rejections decided without the likelihood are free).

Methods:

- ``dpgmm q``: the current proposal. A DP-GMM is fitted to the points; a draw is kept with
  probability floor / density, discarded where the density is below the floor; the floor is
  the q-quantile of the density over the mixture's own draws.
- ``dpgmm held q``: the same, with the floor at the q-quantile of the density over live points
  held out of the fit, so q is the share of the level set given up, in any dimension.
- ``ellipsoid``: one ellipsoid from the points' mean and covariance, enlarged until bootstrap
  left-out points are inside (dynesty's ``single`` bound). Uniform draws inside it.
- ``friends``: the union of equal balls around every live point in whitened space, the radius
  from bootstrap left-out points (UltraNest's MLFriends). Uniform draws on the union.
- ``slice k``: no region at all. Start at a live point and take k slice-sampling steps along
  random whitened directions inside the set (dynesty's ``rslice``). Nothing is lost by
  construction; instead the new point may not have forgotten its start, which ``moved``
  measures (its distance from the start over the distance of two independent points of the
  set; 1 means forgotten).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

import dpgmm

N_TEST = 4000
N_BOOTSTRAPS = 20
MIN_HITS = 100
CHUNK = 100_000
#: Rounds of candidates after which a method's cost is reported as a lower bound.
MAX_ROUNDS = 25
#: Initial bracket of a slice in whitened units, as a share of the radius of a whitened uniform ball.
SLICE_WIDTH_SHARE = 1.0
MAX_STEPS_OUT = 40


class Shape(StrEnum):
    ELLIPSOID = "ellipsoid"
    BANANA = "banana"


@dataclass(frozen=True)
class LevelSet:
    """A super-level set with exact uniform sampling and an exact membership test."""

    centre: np.ndarray
    axes: np.ndarray          # orthonormal columns
    widths: np.ndarray        # semi-axes
    bend: float

    @classmethod
    def random(cls, d: int, shape: Shape, rng: np.random.Generator) -> "LevelSet":
        axes, _ = np.linalg.qr(rng.normal(size=(d, d)))
        widths = np.geomspace(0.03, 0.25, d) if d > 1 else np.array([0.25])
        return cls(np.full(d, 0.5), axes, rng.permutation(widths), 1.5 if shape is Shape.BANANA else 0.0)

    @property
    def d(self) -> int:
        return len(self.centre)

    def __shear(self, t: np.ndarray, sign: float) -> np.ndarray:
        """Bend each odd principal coordinate by the square of the even one before it."""
        t = t.copy()
        even = np.arange(0, self.d - 1, 2)
        t[:, even + 1] += sign * self.bend * self.widths[even + 1] * ((t[:, even] / self.widths[even]) ** 2 - 0.5)
        return t

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        ball = rng.normal(size=(n, self.d))
        ball *= (rng.random(n) ** (1.0 / self.d) / np.linalg.norm(ball, axis=1))[:, None]
        return self.centre + self.__shear(ball * self.widths, +1.0) @ self.axes.T

    def radius(self, points: np.ndarray) -> np.ndarray:
        """Radial coordinate of each point: below 1 inside the set."""
        t = self.__shear((points - self.centre) @ self.axes, -1.0)
        return np.linalg.norm(t / self.widths, axis=1)

    def contains(self, points: np.ndarray) -> np.ndarray:
        return self.radius(points) <= 1.0


@dataclass(frozen=True)
class Score:
    method: str
    lost: float
    calls: float
    note: str = ""


def calls_per_point(level_set: LevelSet, draw, rng: np.random.Generator) -> float:
    """Likelihood calls per point on the set, for candidates from ``draw(n, rng)``; a lower
    bound (the calls spent) when no candidate landed on the set within ``MAX_ROUNDS``."""
    hits = calls = 0
    for _ in range(MAX_ROUNDS):
        candidates = draw(CHUNK, rng)
        calls += len(candidates)
        hits += int(level_set.contains(candidates).sum())
        if hits >= MIN_HITS:
            break
    return calls / hits if hits else float(calls)   # no hit at all: at least this many


#: Share of the live points set aside to place a held-out floor.
HELD_OUT_SHARE = 0.2


def score_dpgmm(level_set: LevelSet, live: np.ndarray, test: np.ndarray, quantile: float, rng: np.random.Generator,
                *, held_out: bool = False) -> Score:
    """The DP-GMM proposal with its floor at a quantile of the density.

    The quantile is taken over the mixture's own draws (the current rule), or with ``held_out``
    over live points the fit did not see: the mixture is then fitted to the other live points,
    and the quantile is directly the share of the level set given up.
    """
    if held_out:
        order = rng.permutation(len(live))
        n_held = int(HELD_OUT_SHARE * len(live))
        fitted, held = live[order[n_held:]], live[order[:n_held]]
    else:
        fitted, held = live, None
    mixture = dpgmm.fit(fitted, n_sweeps=50, seed=int(rng.integers(1 << 31)))
    own = np.asarray(mixture.log_density(mixture.sample(CHUNK, seed=1)))
    log_floor = np.quantile(np.asarray(mixture.log_density(held)) if held_out else own, quantile)

    def thinned(n: int, rng: np.random.Generator) -> np.ndarray:
        """Mixture draws that survive the floor and the thinning; these cost a likelihood call."""
        draws = mixture.sample(n, seed=int(rng.integers(1 << 31)))
        log_density = np.asarray(mixture.log_density(draws))
        keep = (log_density >= log_floor) & (np.log(rng.random(n)) < log_floor - log_density)
        return draws[keep]

    kept_share = len(thinned(CHUNK, rng)) / CHUNK
    lost = float(np.mean(np.asarray(mixture.log_density(test)) < log_floor))
    name = f"dpgmm held {quantile:g}" if held_out else f"dpgmm {quantile:g}"
    return Score(name, lost, calls_per_point(level_set, thinned, rng),
                 f"{mixture.n_components} comp., thinning keeps {kept_share:.1e}, floor at own-draw quantile {np.mean(own < log_floor):.3f}")


def whitener(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and Cholesky factor of the points' covariance."""
    return points.mean(axis=0), np.linalg.cholesky(np.atleast_2d(np.cov(points.T)))


def whiten(points: np.ndarray, mean: np.ndarray, factor: np.ndarray) -> np.ndarray:
    return np.linalg.solve(factor, (points - mean).T).T


def ball(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(n, d))
    return directions * (rng.random(n) ** (1.0 / d) / np.linalg.norm(directions, axis=1))[:, None]


def score_ellipsoid(level_set: LevelSet, live: np.ndarray, test: np.ndarray, rng: np.random.Generator) -> Score:
    mean, factor = whitener(live)
    radius = np.linalg.norm(whiten(live, mean, factor), axis=1).max()
    enlarge = 1.0
    for _ in range(N_BOOTSTRAPS):
        chosen = np.unique(rng.integers(len(live), size=len(live)))
        left_out = np.setdiff1d(np.arange(len(live)), chosen)
        b_mean, b_factor = whitener(live[chosen])
        inside = np.linalg.norm(whiten(live[chosen], b_mean, b_factor), axis=1).max()
        enlarge = max(enlarge, np.linalg.norm(whiten(live[left_out], b_mean, b_factor), axis=1).max() / inside)
    radius *= enlarge
    lost = float(np.mean(np.linalg.norm(whiten(test, mean, factor), axis=1) > radius))
    draw = lambda n, rng: mean + radius * ball(n, level_set.d, rng) @ factor.T  # noqa: E731
    return Score("ellipsoid", lost, calls_per_point(level_set, draw, rng), f"enlarged x{enlarge:.3f}")


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum((a ** 2).sum(1)[:, None] - 2.0 * a @ b.T + (b ** 2).sum(1)[None, :], 0.0)


def score_friends(level_set: LevelSet, live: np.ndarray, test: np.ndarray, rng: np.random.Generator) -> Score:
    mean, factor = whitener(live)
    white = whiten(live, mean, factor)
    radius2 = 0.0
    for _ in range(N_BOOTSTRAPS):
        chosen = np.unique(rng.integers(len(live), size=len(live)))
        left_out = np.setdiff1d(np.arange(len(live)), chosen)
        radius2 = max(radius2, float(squared_distances(white[left_out], white[chosen]).min(axis=1).max()))
    radius = radius2 ** 0.5
    lost = float(np.mean(squared_distances(whiten(test, mean, factor), white).min(axis=1) > radius2))

    def draw(n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform on the union of balls: a ball at random, thinned by how many balls hold the draw."""
        n = min(n, 20_000)
        candidates = white[rng.integers(len(white), size=n)] + radius * ball(n, level_set.d, rng)
        holders = (squared_distances(candidates, white) <= radius2).sum(axis=1)
        return mean + candidates[rng.random(n) < 1.0 / holders] @ factor.T

    return Score("friends", lost, calls_per_point(level_set, draw, rng), f"radius {radius:.2f} (whitened)")


def score_slice(level_set: LevelSet, live: np.ndarray, n_slices: int, n_chains: int, rng: np.random.Generator) -> Score:
    mean, factor = whitener(live)
    start = live[rng.integers(len(live), size=n_chains)]
    x = start.copy()
    calls = 0
    width = SLICE_WIDTH_SHARE * (level_set.d + 2.0) ** 0.5
    for _ in range(n_slices):
        directions = rng.normal(size=x.shape)
        directions = (directions / np.linalg.norm(directions, axis=1)[:, None]) @ factor.T
        low = -width * rng.random(n_chains)
        high = low + width
        for edge, step in ((low, -width), (high, width)):
            growing = np.ones(n_chains, dtype=bool)
            for _ in range(MAX_STEPS_OUT):
                if not growing.any():
                    break
                calls += int(growing.sum())
                growing[growing] = level_set.contains(x[growing] + edge[growing, None] * directions[growing])
                edge[growing] += step
        pending = np.ones(n_chains, dtype=bool)
        while pending.any():
            where = np.flatnonzero(pending)
            t = low[where] + (high[where] - low[where]) * rng.random(len(where))
            candidates = x[where] + t[:, None] * directions[where]
            inside = level_set.contains(candidates)
            calls += len(where)
            x[where[inside]] = candidates[inside]
            pending[where[inside]] = False
            rejected, t_rejected = where[~inside], t[~inside]
            low[rejected[t_rejected < 0]] = t_rejected[t_rejected < 0]
            high[rejected[t_rejected >= 0]] = t_rejected[t_rejected >= 0]
    # How far a new point is from its start, over how far two independent points of the set are apart.
    apart = np.linalg.norm(whiten(x, mean, factor) - whiten(start, mean, factor), axis=1).mean()
    independent = np.linalg.norm(whiten(x, mean, factor) - whiten(level_set.sample(n_chains, rng), mean, factor), axis=1).mean()
    return Score(f"slice {n_slices}", 0.0, calls / n_chains, f"moved {apart / independent:.2f} of an independent point's distance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40, 61])
    parser.add_argument("--shape", type=Shape, choices=list(Shape), default=Shape.ELLIPSOID)
    parser.add_argument("--live", type=int, default=2000, help="exact uniform points every method is given")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.005, 0.2])
    parser.add_argument("--held-out-quantiles", type=float, nargs="*", default=[0.01, 0.05],
                        help="floors placed on held-out live points: the share of the level set given up")
    parser.add_argument("--slices", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--chains", type=int, default=1000, help="new points drawn by the slice sampler")
    parser.add_argument("--skip", nargs="*", default=[], choices=["dpgmm", "ellipsoid", "friends", "slice"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for d in args.dimensions:
        rng = np.random.default_rng(args.seed + d)
        level_set = LevelSet.random(d, args.shape, rng)
        live, test = level_set.sample(args.live, rng), level_set.sample(N_TEST, rng)
        scores: list[Score] = []
        if "dpgmm" not in args.skip:
            scores += [score_dpgmm(level_set, live, test, q, rng) for q in args.quantiles]
            scores += [score_dpgmm(level_set, live, test, q, rng, held_out=True) for q in args.held_out_quantiles]
        if "ellipsoid" not in args.skip:
            scores.append(score_ellipsoid(level_set, live, test, rng))
        if "friends" not in args.skip:
            scores.append(score_friends(level_set, live, test, rng))
        if "slice" not in args.skip:
            scores += [score_slice(level_set, live, k, args.chains, rng) for k in args.slices]
        print(f"\n{args.shape.value}, d = {d}, {args.live} live points")
        for score in scores:
            print(f"  {score.method:16s} lost {score.lost:6.1%}   calls per point {score.calls:10.1f}   {score.note}", flush=True)


if __name__ == "__main__":
    main()
