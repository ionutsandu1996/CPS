# baseline.py
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


# -----------------------------
# IDM (classical)
# -----------------------------
@dataclass
class IDMParams:
    v0: float = 15.0     # desired speed [m/s] (~54 km/h)
    a: float = 1.2       # max accel [m/s^2]
    b: float = 2.0       # comfortable decel [m/s^2]
    T: float = 1.2       # desired time headway [s]
    s0: float = 2.0      # minimum gap [m]
    delta: float = 4.0   # exponent


def idm_accel(v: float, s: float, dv: float, p: IDMParams) -> float:
    """
    Classical IDM acceleration.

    v  : ego speed [m/s]
    s  : net gap to leader [m] (leader front - ego front - veh_len)
    dv : approaching rate = v - v_leader [m/s]
    """
    s_eff = max(s, 0.1)  # avoid division by zero / negative gaps
    s_star = p.s0 + v * p.T + (v * dv) / (2.0 * math.sqrt(p.a * p.b) + 1e-9)
    return p.a * (1.0 - (v / p.v0) ** p.delta - (s_star / s_eff) ** 2)


# -----------------------------
# Traffic signal (fixed-time)
# -----------------------------
class FixedTimeSignal:
    """
    Fixed-time signal for a single approach:
      GREEN for green_s seconds, then RED for red_s seconds, repeating.
    """

    def __init__(self, green_s: float, red_s: float, start_green: bool = True):
        self.green = float(green_s)
        self.red = float(red_s)
        self.cycle = self.green + self.red
        self.start_green = bool(start_green)

    def is_green(self, t: float) -> bool:
        phase = t % self.cycle
        if self.start_green:
            return phase < self.green
        return phase >= self.red


# -----------------------------
# Simulation objects
# -----------------------------
@dataclass
class Vehicle:
    x: float
    v: float
    spawned_t: float
    exit_t: float | None = None

    # metrics
    stops: int = 0
    moving_prev: bool = False
    done: bool = False
    stopped_time: float = 0.0


@dataclass
class SimConfig:
    # road geometry
    L: float = 400.0         # spawn at x = -L
    stop_x: float = 0.0      # stop line at x = 0
    veh_len: float = 4.5

    # time
    dt: float = 0.2
    T_end: float = 900.0

    # demand
    arrival_rate: float = 0.30   # veh/s
    seed: int = 1

    # measurement / definitions
    queue_zone: float = 150.0    # last X meters upstream where we count a queue
    speed_stop_th: float = 0.5   # below this => "stopped"
    speed_zone: float = 200.0    # last X meters upstream where we compute mean speed


# -----------------------------
# Helpers
# -----------------------------
def poisson_spawn(rng: np.random.Generator, lam: float, dt: float) -> bool:
    """
    Simple Bernoulli approximation for Poisson arrivals.
    For small dt: P(arrival in dt) ~= lam * dt
    """
    return rng.random() < lam * dt


def can_spawn_vehicle(vehicles: list[Vehicle], cfg: SimConfig) -> bool:
    """
    Avoid spawning on top of an existing vehicle near x=-L.
    If the most upstream active vehicle is too close to spawn, we skip.
    """
    active = [v for v in vehicles if not v.done]
    if not active:
        return True
    min_x = min(v.x for v in active)
    return not (min_x < -cfg.L + 10.0)


def compute_queue(vehicles: list[Vehicle], cfg: SimConfig) -> int:
    """
    Queue length = number of active vehicles that are:
      - within [ -queue_zone, stop_x ]
      - and nearly stopped (v < speed_stop_th)
    """
    q = 0
    for veh in vehicles:
        if veh.done:
            continue
        if (-cfg.queue_zone <= veh.x <= cfg.stop_x) and (veh.v < cfg.speed_stop_th):
            q += 1
    return q


def mean_speed_near(vehicles: list[Vehicle], cfg: SimConfig) -> float:
    """
    Mean speed over vehicles in the last 'speed_zone' meters upstream.
    """
    speeds = [
        veh.v
        for veh in vehicles
        if (not veh.done) and (-cfg.speed_zone <= veh.x <= cfg.stop_x)
    ]
    return float(np.mean(speeds)) if speeds else 0.0


# -----------------------------
# Baseline simulation
# -----------------------------
def run_baseline(cfg: SimConfig, idm_p: IDMParams, signal: FixedTimeSignal) -> dict:
    """
    Baseline:
      - Human drivers follow classical IDM (interaction with preceding vehicle).
      - Signal is fixed-time.
      - Physical constraint: cannot cross stop line when RED (hard stop at x=-0.5).

    Returns:
      dict with time series + vehicle list (used by run_experiments.py).
    """
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.dt
    times = np.arange(0.0, cfg.T_end + dt, dt)

    vehicles: list[Vehicle] = []
    queue_ts = np.zeros_like(times)
    ms_ts = np.zeros_like(times)

    for k, t in enumerate(times):
        green = signal.is_green(t)

        # ----------------
        # Spawn upstream
        # ----------------
        if poisson_spawn(rng, cfg.arrival_rate, dt) and can_spawn_vehicle(vehicles, cfg):
            vehicles.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t))

        # ----------------
        # Update vehicles (closest to stop line first)
        # ----------------
        active_idx = [i for i, v in enumerate(vehicles) if not v.done]
        active_idx.sort(key=lambda i: vehicles[i].x, reverse=True)

        for pos, i in enumerate(active_idx):
            ego = vehicles[i]

            # Gap to preceding vehicle
            if pos == 0:
                s_veh = 1e9
                dv = 0.0
            else:
                lead = vehicles[active_idx[pos - 1]]
                s_veh = (lead.x - ego.x) - cfg.veh_len
                dv = ego.v - lead.v

            # IDM accel
            acc = idm_accel(ego.v, s_veh, dv, idm_p)

            # Euler integrate
            v_new = max(0.0, ego.v + acc * dt)
            x_new = ego.x + v_new * dt

            # Stop line constraint on RED
            if (not green) and (x_new > -0.5):
                x_new = -0.5
                v_new = 0.0

            # stopped time in queue zone
            if (-cfg.queue_zone <= x_new <= cfg.stop_x) and (v_new < cfg.speed_stop_th) and (not ego.done):
                ego.stopped_time += dt

            # stop counting (moving -> stopped transition)
            moving_now = v_new > cfg.speed_stop_th
            if ego.moving_prev and (not moving_now):
                ego.stops += 1
            ego.moving_prev = moving_now

            # Exit when crossing stop line on GREEN
            if green and (x_new >= cfg.stop_x):
                ego.done = True
                ego.exit_t = t

            ego.x, ego.v = x_new, v_new

        # ----------------
        # time-series logging
        # ----------------
        queue_ts[k] = compute_queue(vehicles, cfg)
        ms_ts[k] = mean_speed_near(vehicles, cfg)

    return {
        "t": times,
        "queue": queue_ts,
        "mean_speed": ms_ts,
        "vehicles": vehicles,
    }


# Optional quick sanity check when running baseline.py directly
if __name__ == "__main__":
    cfg = SimConfig(arrival_rate=0.60, seed=1, T_end=120.0, queue_zone=150.0)
    idm_p = IDMParams()
    sig = FixedTimeSignal(green_s=20.0, red_s=30.0, start_green=True)

    out = run_baseline(cfg, idm_p, sig)
    throughput = sum(v.exit_t is not None for v in out["vehicles"])
    print("baseline.py OK")
    print("Spawned:", len(out["vehicles"]), "Throughput:", throughput)
