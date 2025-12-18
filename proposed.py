# proposed.py
from __future__ import annotations

import numpy as np

from intersection import (
    IDMParams,
    SimConfig,
    Approach,
    detected_near_stopline,
)

from baseline import _run_with_time_varying_green


class ActuatedGapOut2PhaseSignal:
    """
    2-phase actuated control (A/B) with gap-out termination.

    We keep one phase green at a time. Switching happens when:
      - minimum green satisfied AND
      - (gap-out: no detection for gap_threshold) OR max green reached

    Note:
    - This is a minimal “realistic” actuated controller, not SCOOT/SCATS.
    """

    def __init__(
        self,
        G_min: float = 10.0,
        G_max: float = 45.0,
        gap_threshold: float = 2.0,
        start: Approach = "A",
    ):
        self.G_min = float(G_min)
        self.G_max = float(G_max)
        self.gap = float(gap_threshold)

        self.phase: Approach = start
        self.phase_t: float = 0.0
        self.last_detect_t: float = 0.0

    def step(self, t: float, dt: float, detected_current: bool) -> Approach:
        self.phase_t += dt

        if detected_current:
            self.last_detect_t = t

        # enforce minimum green
        if self.phase_t < self.G_min:
            return self.phase

        # max green
        if self.phase_t >= self.G_max:
            self._switch(t)
            return self.phase

        # gap-out
        if (t - self.last_detect_t) >= self.gap:
            self._switch(t)
            return self.phase

        return self.phase

    def _switch(self, t: float):
        self.phase = "B" if self.phase == "A" else "A"
        self.phase_t = 0.0
        self.last_detect_t = t  # avoid immediate gap-out on new phase


def run_proposed(cfg: SimConfig, idm_p: IDMParams, signal: ActuatedGapOut2PhaseSignal):
    """
    Proposed wrapper (actuated gap-out) using same intersection dynamics as baseline.
    """
    dt = cfg.dt
    times = np.arange(0.0, cfg.T_end + dt, dt)

    # We'll create a green_fn(t) by simulating the controller internally.
    phase_over_time: list[Approach] = []

    # Reset controller state for this run
    # (in case user reuses same instance across seeds)
    signal.phase_t = 0.0
    signal.last_detect_t = 0.0

    # We need a “preview” run to generate green decisions? That would be wrong because
    # detection depends on vehicles. So we do it properly:
    # - We run the simulation step-by-step inside _run_with_time_varying_green? not possible.
    #
    # Practical minimal solution:
    # - We embed the controller into green_fn by keeping a tiny state and querying detection
    #   from the currently-updating sim. But _run_with_time_varying_green doesn't expose vehicles.
    #
    # Therefore: we implement a custom run loop here that mirrors _run_with_time_varying_green,
    # but adds detection->controller->green.
    #
    # This keeps fairness: same physics, only green logic differs.

    from intersection import Vehicle, idm_accel, poisson_spawn, compute_queue_for_approach, mean_speed_near_for_approach, mean_speed_near_total

    rng = np.random.default_rng(cfg.seed)

    vehicles: list[Vehicle] = []

    queue_A_ts = np.zeros_like(times)
    queue_B_ts = np.zeros_like(times)
    queue_tot_ts = np.zeros_like(times)

    ms_A_ts = np.zeros_like(times)
    ms_B_ts = np.zeros_like(times)
    ms_tot_ts = np.zeros_like(times)

    passed_A_ts = np.zeros_like(times)
    passed_B_ts = np.zeros_like(times)

    phase_ts = np.zeros_like(times, dtype=int)  # 0 A green, 1 B green

    def can_spawn(approach: Approach) -> bool:
        active = [v for v in vehicles if (not v.done) and v.approach == approach]
        if not active:
            return True
        min_x = min(v.x for v in active)
        return not (min_x < -cfg.L + 10.0)

    for k, t in enumerate(times):
        # detector for CURRENT phase only
        detected = detected_near_stopline(vehicles, cfg, signal.phase)

        # controller decides who is green now
        green_of = signal.step(t, cfg.dt, detected)
        green_A = green_of == "A"
        green_B = green_of == "B"
        phase_ts[k] = 0 if green_A else 1

        # spawn
        if poisson_spawn(rng, cfg.arrival_rate_A, cfg.dt) and can_spawn("A"):
            vehicles.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t, approach="A"))
        if poisson_spawn(rng, cfg.arrival_rate_B, cfg.dt) and can_spawn("B"):
            vehicles.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t, approach="B"))

        # update each approach
        for approach, is_green in (("A", green_A), ("B", green_B)):
            active_idx = [i for i, v in enumerate(vehicles) if (not v.done) and v.approach == approach]
            active_idx.sort(key=lambda i: vehicles[i].x, reverse=True)

            for pos, i in enumerate(active_idx):
                ego = vehicles[i]

                if pos == 0:
                    s_veh = 1e9
                    dv_veh = 0.0
                else:
                    lead = vehicles[active_idx[pos - 1]]
                    s_veh = (lead.x - ego.x) - cfg.veh_len
                    dv_veh = ego.v - lead.v

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

                v_new = max(0.0, ego.v + acc * cfg.dt)
                x_new = ego.x + v_new * cfg.dt

                if not is_green and x_new > -0.5:
                    x_new = -0.5
                    v_new = 0.0

                if (-cfg.queue_zone <= x_new <= cfg.stop_x) and (v_new < cfg.speed_stop_th) and (not ego.done):
                    ego.stopped_time += cfg.dt

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
