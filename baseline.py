# baseline.py
from __future__ import annotations

from dataclasses import dataclass

from intersection import IDMParams, SimConfig, simulate_intersection_2phase, Approach


class FixedTime2PhaseSignal:
    """
    Two-phase fixed-time:
      - Phase A green for green_A seconds
      - Phase B green for green_B seconds
      - repeat
    """

    def __init__(self, green_A: float, green_B: float, start: Approach = "A"):
        self.green_A = float(green_A)
        self.green_B = float(green_B)
        self.cycle = self.green_A + self.green_B
        self.start = start

    def green_of(self, t: float) -> Approach:
        phase = t % self.cycle

        if self.start == "A":
            return "A" if phase < self.green_A else "B"
        else:
            # start with B
            return "B" if phase < self.green_B else "A"


def run_baseline(cfg: SimConfig, idm_p: IDMParams, signal: FixedTime2PhaseSignal):
    """
    Baseline wrapper around the common intersection simulator.
    """
    # We run the sim once, but the simulator needs to know green each step.
    # Easiest: re-run inside a loop would be slow. Instead, we create a tiny closure
    # conceptually. Here we use a trick: we step time and call simulate_intersection_2phase
    # which expects a single constant green_of -> so we must not do that.
    #
    # So: we do a small adaptation: we create a phase array by sampling signal over time,
    # then "replay" by stepping the simulator. BUT our simulator currently takes constant
    # green_of. We'll keep it simple:
    #
    # => We implement fixed-time by running two sims and stitching? Not ok.
    #
    # Solution: For clarity and correctness, we keep the simulator constant-green
    # and instead move the time-varying signal logic into the simulator. That would
    # require edits.
    #
    # BUT we already included phase_ts inside simulator. So we should pass a function.
    #
    # Therefore: use the alternative function below (implemented here).

    return _run_with_time_varying_green(cfg, idm_p, signal.green_of)


# ---- Time-varying green runner (shared pattern) ----
from typing import Callable, Dict
import numpy as np
from intersection import Vehicle, idm_accel, poisson_spawn, compute_queue_for_approach, mean_speed_near_for_approach, mean_speed_near_total, detected_near_stopline  # noqa


def _run_with_time_varying_green(cfg: SimConfig, idm_p: IDMParams, green_fn: Callable[[float], Approach]) -> Dict:
    """
    Same dynamics as intersection.simulate_intersection_2phase, but green is time-varying.
    Implemented as a local copy to avoid circular imports / messy architecture.
    """

    rng = np.random.default_rng(cfg.seed)
    dt = cfg.dt
    times = np.arange(0.0, cfg.T_end + dt, dt)

    vehicles: list[Vehicle] = []

    queue_A_ts = np.zeros_like(times)
    queue_B_ts = np.zeros_like(times)
    queue_tot_ts = np.zeros_like(times)

    ms_A_ts = np.zeros_like(times)
    ms_B_ts = np.zeros_like(times)
    ms_tot_ts = np.zeros_like(times)

    passed_A_ts = np.zeros_like(times)
    passed_B_ts = np.zeros_like(times)

    phase_ts = np.zeros_like(times, dtype=int)  # 0 => A green, 1 => B green

    def can_spawn(approach: Approach) -> bool:
        active = [v for v in vehicles if (not v.done) and v.approach == approach]
        if not active:
            return True
        min_x = min(v.x for v in active)
        return not (min_x < -cfg.L + 10.0)

    for k, t in enumerate(times):
        green_of = green_fn(t)
        green_A = green_of == "A"
        green_B = green_of == "B"
        phase_ts[k] = 0 if green_A else 1

        # spawn
        if poisson_spawn(rng, cfg.arrival_rate_A, dt) and can_spawn("A"):
            vehicles.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t, approach="A"))
        if poisson_spawn(rng, cfg.arrival_rate_B, dt) and can_spawn("B"):
            vehicles.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t, approach="B"))

        # update each approach
        for approach, is_green in (("A", green_A), ("B", green_B)):
            active_idx = [i for i, v in enumerate(vehicles) if (not v.done) and v.approach == approach]
            active_idx.sort(key=lambda i: vehicles[i].x, reverse=True)

            for pos, i in enumerate(active_idx):
                ego = vehicles[i]

                # vehicle leader constraint
                if pos == 0:
                    s_veh = 1e9
                    dv_veh = 0.0
                else:
                    lead = vehicles[active_idx[pos - 1]]
                    s_veh = (lead.x - ego.x) - cfg.veh_len
                    dv_veh = ego.v - lead.v

                # stop line constraint on red
                if not is_green:
                    s_sig = (cfg.stop_x - ego.x) - cfg.veh_len
                    dv_sig = ego.v
                else:
                    s_sig = 1e9
                    dv_sig = 0.0

                if s_sig < s_veh:
                    s_eff, dv_eff = s_sig, dv_sig
                else:
                    s_eff, dv_eff = s_veh, dv_veh

                acc = idm_accel(ego.v, s_eff, dv_eff, idm_p)

                v_new = max(0.0, ego.v + acc * dt)
                x_new = ego.x + v_new * dt

                if not is_green and x_new > -0.5:
                    x_new = -0.5
                    v_new = 0.0

                if (-cfg.queue_zone <= x_new <= cfg.stop_x) and (v_new < cfg.speed_stop_th) and (not ego.done):
                    ego.stopped_time += dt

                moving_now = v_new > cfg.speed_stop_th
                if ego.moving_prev and not moving_now:
                    ego.stops += 1
                ego.moving_prev = moving_now

                if is_green and x_new >= cfg.stop_x:
                    ego.done = True
                    ego.exit_t = t
                    if approach == "A":
                        passed_A_ts[k] += 1
                    else:
                        passed_B_ts[k] += 1

                ego.x, ego.v = x_new, v_new

        queue_A_ts[k] = compute_queue_for_approach(vehicles, cfg, "A")
        queue_B_ts[k] = compute_queue_for_approach(vehicles, cfg, "B")
        queue_tot_ts[k] = queue_A_ts[k] + queue_B_ts[k]

        ms_A_ts[k] = mean_speed_near_for_approach(vehicles, cfg, "A")
        ms_B_ts[k] = mean_speed_near_for_approach(vehicles, cfg, "B")
        ms_tot_ts[k] = mean_speed_near_total(vehicles, cfg)

    return {
        "t": times,
        "phase": phase_ts,
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
