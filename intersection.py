# intersection.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Literal, Tuple

import numpy as np

Approach = Literal["A", "B"]


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
    s_eff = max(s, 0.1)
    s_star = p.s0 + v * p.T + (v * dv) / (2.0 * math.sqrt(p.a * p.b) + 1e-9)
    return p.a * (1.0 - (v / p.v0) ** p.delta - (s_star / s_eff) ** 2)


# -----------------------------
# Simulation objects
# -----------------------------
@dataclass
class Vehicle:
    x: float
    v: float
    spawned_t: float
    approach: Approach
    exit_t: float | None = None
    stops: int = 0
    moving_prev: bool = False
    done: bool = False
    stopped_time: float = 0.0


@dataclass
class SimConfig:
    # road (each approach modeled as 1D segment x in [-L, 0])
    L: float = 400.0
    stop_x: float = 0.0
    veh_len: float = 4.5

    # time
    dt: float = 0.2
    T_end: float = 300.0

    # demand (veh/s) per approach
    arrival_rate_A: float = 0.30
    arrival_rate_B: float = 0.30
    seed: int = 1

    # queue / stop definitions
    queue_zone: float = 150.0        # meters upstream considered "queue area"
    speed_stop_th: float = 0.5       # below this => considered stopped

    # speed measurement zone (near intersection)
    speed_zone: float = 200.0        # meters upstream for mean approach speed

    # detector zone (presence near stop line)
    detect_zone: float = 25.0        # meters upstream for actuation detector


# -----------------------------
# Helpers
# -----------------------------
def poisson_spawn(rng: np.random.Generator, lam: float, dt: float) -> bool:
    return rng.random() < lam * dt


def compute_queue_for_approach(vehicles: List[Vehicle], cfg: SimConfig, approach: Approach) -> int:
    q = 0
    for veh in vehicles:
        if veh.done or veh.approach != approach:
            continue
        if -cfg.queue_zone <= veh.x <= cfg.stop_x and veh.v < cfg.speed_stop_th:
            q += 1
    return q


def compute_queue_total(vehicles: List[Vehicle], cfg: SimConfig) -> int:
    return compute_queue_for_approach(vehicles, cfg, "A") + compute_queue_for_approach(vehicles, cfg, "B")


def mean_speed_near_for_approach(vehicles: List[Vehicle], cfg: SimConfig, approach: Approach) -> float:
    speeds = [
        veh.v
        for veh in vehicles
        if (not veh.done)
        and veh.approach == approach
        and (-cfg.speed_zone <= veh.x <= cfg.stop_x)
    ]
    return float(np.mean(speeds)) if speeds else 0.0


def mean_speed_near_total(vehicles: List[Vehicle], cfg: SimConfig) -> float:
    # Mean over all active vehicles within the speed zone (both approaches)
    speeds = [
        veh.v
        for veh in vehicles
        if (not veh.done) and (-cfg.speed_zone <= veh.x <= cfg.stop_x)
    ]
    return float(np.mean(speeds)) if speeds else 0.0


def detected_near_stopline(vehicles: List[Vehicle], cfg: SimConfig, approach: Approach) -> bool:
    """
    Presence detector: True if any vehicle of 'approach' is within detect_zone upstream of stop line.
    """
    for veh in vehicles:
        if veh.done or veh.approach != approach:
            continue
        dist_to_stop = cfg.stop_x - veh.x
        if 0.0 <= dist_to_stop <= cfg.detect_zone:
            return True
    return False


def _can_spawn(vehicles: List[Vehicle], cfg: SimConfig, approach: Approach) -> bool:
    """
    Simple spacing check near spawn point so we don't spawn on top of an existing vehicle.
    """
    active = [v for v in vehicles if (not v.done) and v.approach == approach]
    if not active:
        return True
    min_x = min(v.x for v in active)
    return not (min_x < -cfg.L + 10.0)


# -----------------------------
# Core simulator (2 conflicting approaches, 2-phase signal)
# -----------------------------
def simulate_intersection_2phase(
    cfg: SimConfig,
    idm_p: IDMParams,
    green_of: Approach | None,
) -> Dict:
    """
    One time step of simulation is embedded in the main loop. This function runs the whole horizon.

    green_of:
      - "A": approach A has green, B has red
      - "B": approach B has green, A has red
      - None: all-red (optional, not used here)

    IMPORTANT:
    - Same vehicle dynamics for baseline/proposed. Only 'green_of(t)' differs by controller.
    """

    rng = np.random.default_rng(cfg.seed)
    dt = cfg.dt
    times = np.arange(0.0, cfg.T_end + dt, dt)

    vehicles: List[Vehicle] = []

    # time series
    queue_A_ts = np.zeros_like(times)
    queue_B_ts = np.zeros_like(times)
    queue_tot_ts = np.zeros_like(times)

    ms_A_ts = np.zeros_like(times)
    ms_B_ts = np.zeros_like(times)
    ms_tot_ts = np.zeros_like(times)

    passed_A_ts = np.zeros_like(times)  # vehicles that exited at time step (A)
    passed_B_ts = np.zeros_like(times)  # vehicles that exited at time step (B)

    phase_ts = np.zeros_like(times, dtype=int)  # 0 => A green, 1 => B green, 2 => all-red

    for k, t in enumerate(times):
        # ----------------
        # signal state this step
        # ----------------
        if green_of == "A":
            green_A, green_B = True, False
            phase_ts[k] = 0
        elif green_of == "B":
            green_A, green_B = False, True
            phase_ts[k] = 1
        else:
            green_A, green_B = False, False
            phase_ts[k] = 2

        # ----------------
        # spawn vehicles (independent Poisson per approach)
        # ----------------
        if poisson_spawn(rng, cfg.arrival_rate_A, dt) and _can_spawn(vehicles, cfg, "A"):
            vehicles.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t, approach="A"))

        if poisson_spawn(rng, cfg.arrival_rate_B, dt) and _can_spawn(vehicles, cfg, "B"):
            vehicles.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t, approach="B"))

        # ----------------
        # update each approach separately (1 lane per approach)
        # ----------------
        for approach, is_green in (("A", green_A), ("B", green_B)):
            active_idx = [i for i, v in enumerate(vehicles) if (not v.done) and v.approach == approach]
            active_idx.sort(key=lambda i: vehicles[i].x, reverse=True)  # closest to stop line first

            for pos, i in enumerate(active_idx):
                ego = vehicles[i]

                # (1) gap to preceding vehicle (normal IDM interaction)
                if pos == 0:
                    s_veh = 1e9
                    dv_veh = 0.0
                else:
                    lead = vehicles[active_idx[pos - 1]]
                    s_veh = (lead.x - ego.x) - cfg.veh_len
                    dv_veh = ego.v - lead.v

                # (2) gap to stop line if RED (virtual leader constraint)
                if not is_green:
                    s_sig = (cfg.stop_x - ego.x) - cfg.veh_len
                    dv_sig = ego.v
                else:
                    s_sig = 1e9
                    dv_sig = 0.0

                # (3) effective constraint = most restrictive
                if s_sig < s_veh:
                    s_eff, dv_eff = s_sig, dv_sig
                else:
                    s_eff, dv_eff = s_veh, dv_veh

                acc = idm_accel(ego.v, s_eff, dv_eff, idm_p)

                # Euler integration
                v_new = max(0.0, ego.v + acc * dt)
                x_new = ego.x + v_new * dt

                # Hard stop at stop line on red (physical enforcement)
                if not is_green and x_new > -0.5:
                    x_new = -0.5
                    v_new = 0.0

                # stopped time near stop line (proxy)
                if (-cfg.queue_zone <= x_new <= cfg.stop_x) and (v_new < cfg.speed_stop_th) and (not ego.done):
                    ego.stopped_time += dt

                # count stops
                moving_now = v_new > cfg.speed_stop_th
                if ego.moving_prev and not moving_now:
                    ego.stops += 1
                ego.moving_prev = moving_now

                # exit when crossing stop line on green
                if is_green and x_new >= cfg.stop_x:
                    ego.done = True
                    ego.exit_t = t
                    if approach == "A":
                        passed_A_ts[k] += 1
                    else:
                        passed_B_ts[k] += 1

                ego.x, ego.v = x_new, v_new

        # store time series
        queue_A_ts[k] = compute_queue_for_approach(vehicles, cfg, "A")
        queue_B_ts[k] = compute_queue_for_approach(vehicles, cfg, "B")
        queue_tot_ts[k] = queue_A_ts[k] + queue_B_ts[k]

        ms_A_ts[k] = mean_speed_near_for_approach(vehicles, cfg, "A")
        ms_B_ts[k] = mean_speed_near_for_approach(vehicles, cfg, "B")
        ms_tot_ts[k] = mean_speed_near_total(vehicles, cfg)

    return {
        "t": times,
        "phase": phase_ts,               # 0 A green / 1 B green
        "queue_A": queue_A_ts,
        "queue_B": queue_B_ts,
        "queue_total": queue_tot_ts,
        "mean_speed_A": ms_A_ts,
        "mean_speed_B": ms_B_ts,
        "mean_speed_total": ms_tot_ts,
        "passed_A_ts": passed_A_ts,
        "passed_B_ts": passed_B_ts,
        "vehicles": vehicles,
    }


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(cfg: SimConfig, out: Dict, idm_p: IDMParams) -> Dict:
    vehicles = out["vehicles"]
    done = [v for v in vehicles if v.exit_t is not None]

    free_tt = cfg.L / max(idm_p.v0, 0.1)

    travel_times = np.array([v.exit_t - v.spawned_t for v in done]) if done else np.array([])
    delays = travel_times - free_tt if travel_times.size else np.array([])

    stops = np.array([v.stops for v in done]) if done else np.array([])
    stopped_time = np.array([v.stopped_time for v in done]) if done else np.array([])

    throughput_A = int(np.sum(out["passed_A_ts"]))
    throughput_B = int(np.sum(out["passed_B_ts"]))
    throughput_total = throughput_A + throughput_B

    # green share from phase_ts
    phase_ts = out["phase"]
    green_A_share = float(np.mean(phase_ts == 0))
    green_B_share = float(np.mean(phase_ts == 1))

    # served per green second (rough utilization indicator)
    total_time = cfg.T_end
    green_A_time = green_A_share * total_time
    green_B_time = green_B_share * total_time
    served_per_green_A = float(throughput_A / green_A_time) if green_A_time > 1e-9 else np.nan
    served_per_green_B = float(throughput_B / green_B_time) if green_B_time > 1e-9 else np.nan
    served_per_green_total = float(throughput_total / total_time)  # veh/s overall

    return {
        "throughput_A": throughput_A,
        "throughput_B": throughput_B,
        "throughput_total": throughput_total,
        "avg_queue_total": float(np.mean(out["queue_total"])),
        "max_queue_total": int(np.max(out["queue_total"])),
        "avg_queue_A": float(np.mean(out["queue_A"])),
        "avg_queue_B": float(np.mean(out["queue_B"])),
        "avg_travel_time_s": float(np.mean(travel_times)) if travel_times.size else np.nan,
        "avg_delay_s": float(np.mean(delays)) if delays.size else np.nan,
        "p95_delay_s": float(np.percentile(delays, 95)) if delays.size else np.nan,
        "avg_stops_per_vehicle": float(np.mean(stops)) if stops.size else np.nan,
        "avg_stopped_time_s": float(np.mean(stopped_time)) if stopped_time.size else np.nan,
        "N_spawned": len(vehicles),
        "green_A_share": green_A_share,
        "green_B_share": green_B_share,
        "served_per_green_A": served_per_green_A,
        "served_per_green_B": served_per_green_B,
        "served_per_green_total": served_per_green_total,
    }
