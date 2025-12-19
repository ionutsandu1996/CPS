# proposed.py
from __future__ import annotations

import numpy as np

from baseline import (
    IDMParams,
    Vehicle,
    SimConfig,
    poisson_spawn,
    can_spawn_vehicle,
    compute_queue,
    mean_speed_near,
    idm_accel,
)


# -----------------------------
# CPS: Actuated traffic signal (gap-out)
# -----------------------------
class ActuatedGapOutSignal:
    """
    Actuated control with gap-out termination (single approach demo).

    Parameters:
      G_min         minimum green time [s]
      G_max         maximum green time [s]
      R_min         minimum red time [s]
      gap_threshold if no vehicle is detected for this long during GREEN,
                    the signal ends green (after G_min) [s]

    State variables:
      phase         "GREEN" or "RED"
      phase_t       elapsed time in current phase
      last_detect_t last time a vehicle was detected during GREEN
    """

    def __init__(self, G_min=10.0, G_max=45.0, R_min=10.0, gap_threshold=2.0):
        self.G_min = float(G_min)
        self.G_max = float(G_max)
        self.R_min = float(R_min)
        self.gap = float(gap_threshold)

        self.phase = "GREEN"
        self.phase_t = 0.0
        self.last_detect_t = 0.0

    def step(self, t: float, dt: float, detected: bool) -> bool:
        """
        Advances controller state by dt and returns True if GREEN, False if RED.
        """
        self.phase_t += dt

        if self.phase == "GREEN":
            # update last detection time if detector sees a vehicle
            if detected:
                self.last_detect_t = t

            # enforce minimum green
            if self.phase_t < self.G_min:
                return True

            # prevent over-long green
            if self.phase_t >= self.G_max:
                self.phase = "RED"
                self.phase_t = 0.0
                return False

            # gap-out: no vehicle detected recently => end green
            if (t - self.last_detect_t) >= self.gap:
                self.phase = "RED"
                self.phase_t = 0.0
                return False

            return True

        # RED phase
        if self.phase_t < self.R_min:
            return False

        # after minimum red, go green again (single-approach demo)
        self.phase = "GREEN"
        self.phase_t = 0.0
        # reset detection to "now" to avoid immediate gap-out
        self.last_detect_t = t
        return True


# -----------------------------
# CPS: Detector model (infrastructure sensing)
# -----------------------------
def detected_near_stopline(
    vehicles: list[Vehicle],
    cfg: SimConfig,
    detect_zone: float = 25.0,
) -> bool:
    """
    Returns True if any active vehicle is within `detect_zone` meters upstream
    of the stop line. This mimics a simple loop detector / camera presence zone.

    detect_zone = 25m means "detector covers last 25 meters before x=0".
    """
    for veh in vehicles:
        if veh.done:
            continue

        dist_to_stop = cfg.stop_x - veh.x
        if 0.0 <= dist_to_stop <= detect_zone:
            return True

    return False


# -----------------------------
# Proposed simulation (IDM + CPS traffic signal)
# -----------------------------
def run_proposed(
    cfg: SimConfig,
    idm_p: IDMParams,
    signal: ActuatedGapOutSignal,
    detect_zone: float = 25.0,
):
    """
    Proposed:
      - Vehicles still follow IDM (human drivers).
      - CPS loop: detector -> controller -> signal phase (GREEN/RED).
      - When signal is RED, stop line acts as a "virtual leader" constraint, so IDM
        uses the most restrictive gap:
            s_eff = min( gap_to_leader_vehicle, gap_to_stopline_if_red )

    Returns time series + vehicle list (for metrics later).
    """
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.dt
    times = np.arange(0.0, cfg.T_end + dt, dt)

    vehicles: list[Vehicle] = []
    queue_ts = np.zeros_like(times)
    ms_ts = np.zeros_like(times)

    for k, t in enumerate(times):
        # -----------------------------
        # CPS: sensing -> control -> actuation
        # -----------------------------
        detected = detected_near_stopline(vehicles, cfg, detect_zone)
        green = signal.step(t, dt, detected)

        # -----------------------------
        # Spawn vehicles upstream (same as baseline)
        # -----------------------------
        if poisson_spawn(rng, cfg.arrival_rate, dt) and can_spawn_vehicle(vehicles, cfg):
            vehicles.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t))

        # -----------------------------
        # Update vehicles: closest to stop line first
        # -----------------------------
        active_idx = [i for i, v in enumerate(vehicles) if not v.done]
        active_idx.sort(key=lambda i: vehicles[i].x, reverse=True)

        for pos, i in enumerate(active_idx):
            ego = vehicles[i]

            # (1) constraint from preceding vehicle
            if pos == 0:
                s_veh = 1e9
                dv_veh = 0.0
            else:
                lead = vehicles[active_idx[pos - 1]]
                s_veh = (lead.x - ego.x) - cfg.veh_len
                dv_veh = ego.v - lead.v

            # (2) constraint from stop line if RED (virtual leader)
            if not green:
                s_sig = (cfg.stop_x - ego.x) - cfg.veh_len
                dv_sig = ego.v  # "leader" is stationary at stop line
            else:
                s_sig = 1e9
                dv_sig = 0.0

            # (3) choose most restrictive constraint
            if s_sig < s_veh:
                s_eff, dv_eff = s_sig, dv_sig
            else:
                s_eff, dv_eff = s_veh, dv_veh

            # IDM accel with effective constraint
            acc = idm_accel(ego.v, s_eff, dv_eff, idm_p)

            # Euler integration
            v_new = max(0.0, ego.v + acc * dt)
            x_new = ego.x + v_new * dt

            # Physical enforcement: cannot cross stop line on RED
            if not green and x_new > -0.5:
                x_new = -0.5
                v_new = 0.0

            # stopped time near stop line
            if -cfg.queue_zone <= x_new <= cfg.stop_x and v_new < cfg.speed_stop_th and not ego.done:
                ego.stopped_time += dt

            # stops count
            moving_now = v_new > cfg.speed_stop_th
            if ego.moving_prev and not moving_now:
                ego.stops += 1
            ego.moving_prev = moving_now

            # Exit when crossing stop line on GREEN
            if green and x_new >= cfg.stop_x:
                ego.done = True
                ego.exit_t = t

            ego.x, ego.v = x_new, v_new

        # -----------------------------
        # time-series logging
        # -----------------------------
        queue_ts[k] = compute_queue(vehicles, cfg)
        ms_ts[k] = mean_speed_near(vehicles, cfg)

    return {
        "t": times,
        "queue": queue_ts,
        "mean_speed": ms_ts,
        "vehicles": vehicles,
    }


# Optional sanity check
if __name__ == "__main__":
    from baseline import FixedTimeSignal, run_baseline

    cfg = SimConfig(arrival_rate=0.60, seed=1, T_end=120.0, queue_zone=150.0)
    idm_p = IDMParams()

    # Baseline (for quick side-by-side)
    base_sig = FixedTimeSignal(green_s=20.0, red_s=30.0, start_green=True)
    out_b = run_baseline(cfg, idm_p, base_sig)

    # Proposed
    prop_sig = ActuatedGapOutSignal(G_min=10.0, G_max=45.0, R_min=10.0, gap_threshold=2.0)
    out_p = run_proposed(cfg, idm_p, prop_sig, detect_zone=25.0)

    thr_b = sum(v.exit_t is not None for v in out_b["vehicles"])
    thr_p = sum(v.exit_t is not None for v in out_p["vehicles"])
    print("proposed.py OK")
    print("Throughput baseline:", thr_b, "Throughput proposed:", thr_p)
