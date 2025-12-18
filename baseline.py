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
    v0: float = 15.0     # desired speed [m/s]
    a: float = 1.2       # max accel [m/s^2]
    b: float = 2.0       # comfortable decel [m/s^2]
    T: float = 1.2       # desired time headway [s]
    s0: float = 2.0      # minimum gap [m]
    delta: float = 4.0   # exponent


def idm_accel(v: float, s: float, dv: float, p: IDMParams) -> float:
    """
    Classical IDM acceleration.
    v  : ego speed
    s  : net gap to leader (meters)
    dv : approaching rate = v - v_leader
    """
    s_eff = max(s, 0.1)  # avoid division by zero
    s_star = p.s0 + v * p.T + (v * dv) / (2.0 * math.sqrt(p.a * p.b) + 1e-9)
    return p.a * (1.0 - (v / p.v0) ** p.delta - (s_star / s_eff) ** 2)


# -----------------------------
# Traffic signal (fixed-time)
# -----------------------------
class FixedTimeSignal:
    def __init__(self, green_s: float, red_s: float, start_green: bool = True):
        self.green = float(green_s)
        self.red = float(red_s)
        self.cycle = self.green + self.red
        self.start_green = start_green

    def is_green(self, t: float) -> bool:
        phase = t % self.cycle
        return phase < self.green if self.start_green else phase >= self.red


# -----------------------------
# Simulation objects
# -----------------------------
@dataclass
class Vehicle:
    x: float
    v: float
    spawned_t: float
    exit_t: float | None = None
    stops: int = 0
    moving_prev: bool = False
    done: bool = False
    stopped_time: float = 0.0


@dataclass
class SimConfig:
    # road
    L: float = 400.0         # spawn at x=-L, stop line at x=0
    stop_x: float = 0.0
    veh_len: float = 4.5

    # time
    dt: float = 0.2
    T_end: float = 300.0

    # demand
    arrival_rate: float = 0.30  # veh/s
    seed: int = 1

    # queue / stop definitions (for later metrics)
    queue_zone: float = 150.0
    speed_stop_th: float = 0.5


# -----------------------------
# Helpers
# -----------------------------
def poisson_spawn(rng: np.random.Generator, lam: float, dt: float) -> bool:
    # For small dt: P(arrival) ≈ lam*dt
    return rng.random() < lam * dt


def compute_queue(vehicles: list[Vehicle], cfg: SimConfig) -> int:
    q = 0
    for veh in vehicles:
        if veh.done:
            continue
        if -cfg.queue_zone <= veh.x <= cfg.stop_x and veh.v < cfg.speed_stop_th:
            q += 1
    return q


def mean_speed_near(vehicles: list[Vehicle]) -> float:
    speeds = [veh.v for veh in vehicles if (not veh.done) and (-200.0 <= veh.x <= 0.0)]
    return float(np.mean(speeds)) if speeds else 0.0


# -----------------------------
# Baseline run
# -----------------------------
def run_baseline(cfg: SimConfig, idm_p: IDMParams, signal: FixedTimeSignal):
    """
    Baseline:
      - Classical IDM: interaction only with the preceding vehicle
      - Fixed-time signal control
      - Hard stop enforced at stop line when red
    Returns time series + vehicle list (for metrics later).
    """
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.dt
    times = np.arange(0.0, cfg.T_end + dt, dt)

    vehicles: list[Vehicle] = []
    queue_ts = np.zeros_like(times)
    ms_ts = np.zeros_like(times)

    for k, t in enumerate(times):
        green = signal.is_green(t)

        # Spawn a new vehicle upstream (simple spacing check)
        if poisson_spawn(rng, cfg.arrival_rate, dt):
            can_spawn = True
            active = [v for v in vehicles if not v.done]
            if active:
                min_x = min(v.x for v in active)
                if min_x < -cfg.L + 10.0:
                    can_spawn = False
            if can_spawn:
                vehicles.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t))

        # Update vehicles: closest to stop line first
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

            acc = idm_accel(ego.v, s_veh, dv, idm_p)

            # Euler integration
            v_new = max(0.0, ego.v + acc * dt)
            x_new = ego.x + v_new * dt

            # Hard stop at stop line on red
            if not green and x_new > -0.5:
                x_new = -0.5
                v_new = 0.0

            # Accumulate stopped time near stop line (proxy)
            if -cfg.queue_zone <= x_new <= cfg.stop_x and v_new < cfg.speed_stop_th and not ego.done:
                ego.stopped_time += dt

            # Count stops (moving -> stopped transition)
            moving_now = v_new > cfg.speed_stop_th
            if ego.moving_prev and not moving_now:
                ego.stops += 1
            ego.moving_prev = moving_now

            # Exit when crossing stop line on green
            if green and x_new >= cfg.stop_x:
                ego.done = True
                ego.exit_t = t

            ego.x, ego.v = x_new, v_new

        queue_ts[k] = compute_queue(vehicles, cfg)
        ms_ts[k] = mean_speed_near(vehicles)

    return {
        "t": times,
        "queue": queue_ts,
        "mean_speed": ms_ts,
        "vehicles": vehicles,
    }
