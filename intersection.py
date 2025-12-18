from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from baseline import (
    IDMParams,
    Vehicle,
    SimConfig,
    poisson_spawn,
    compute_queue,
    mean_speed_near,
    idm_accel,
)


# -----------------------------
# Two-phase signal controllers
# -----------------------------
class FixedTimeTwoPhaseSignal:
    """
    Two-phase fixed-time controller:
      Phase A: A=GREEN, B=RED
      Phase B: A=RED,   B=GREEN
    """
    def __init__(self, G_A=15.0, G_B=15.0, all_red=2.0):
        self.G_A = float(G_A)
        self.G_B = float(G_B)
        self.all_red = float(all_red)

        self.phase = "A_GREEN"   # "A_GREEN", "ALL_RED_AB", "B_GREEN", "ALL_RED_BA"
        self.phase_t = 0.0

    def step(self, dt: float) -> tuple[bool, bool]:
        """
        Returns (A_green, B_green).
        """
        self.phase_t += dt

        if self.phase == "A_GREEN":
            if self.phase_t >= self.G_A:
                self.phase = "ALL_RED_AB"
                self.phase_t = 0.0

        elif self.phase == "ALL_RED_AB":
            if self.phase_t >= self.all_red:
                self.phase = "B_GREEN"
                self.phase_t = 0.0

        elif self.phase == "B_GREEN":
            if self.phase_t >= self.G_B:
                self.phase = "ALL_RED_BA"
                self.phase_t = 0.0

        elif self.phase == "ALL_RED_BA":
            if self.phase_t >= self.all_red:
                self.phase = "A_GREEN"
                self.phase_t = 0.0

        A_green = self.phase == "A_GREEN"
        B_green = self.phase == "B_GREEN"
        return A_green, B_green


class ActuatedGapOutTwoPhaseSignal:
    """
    Two-phase actuated gap-out controller.
    - Each green has min/max time.
    - Gap-out: after G_min, if no detection for 'gap' seconds -> terminate green.
    - Has all-red (lost time) between phases.
    """
    def __init__(
        self,
        Gmin_A=10.0, Gmax_A=45.0,
        Gmin_B=10.0, Gmax_B=45.0,
        Rmin=5.0,
        gap=2.0,
        all_red=2.0,
    ):
        self.Gmin_A = float(Gmin_A); self.Gmax_A = float(Gmax_A)
        self.Gmin_B = float(Gmin_B); self.Gmax_B = float(Gmax_B)
        self.Rmin = float(Rmin)      # minimum time to keep the opposing approach red
        self.gap = float(gap)
        self.all_red = float(all_red)

        self.phase = "A_GREEN"
        self.phase_t = 0.0
        self.last_det_t = 0.0  # last detection time during current green

    def step(self, t: float, dt: float, det_A: bool, det_B: bool) -> tuple[bool, bool]:
        """
        Returns (A_green, B_green).
        """
        self.phase_t += dt

        # helper: which detector matters now
        if self.phase == "A_GREEN":
            detected = det_A
            Gmin, Gmax = self.Gmin_A, self.Gmax_A
            next_all_red = "ALL_RED_AB"
        elif self.phase == "B_GREEN":
            detected = det_B
            Gmin, Gmax = self.Gmin_B, self.Gmax_B
            next_all_red = "ALL_RED_BA"
        else:
            detected = False
            Gmin, Gmax = 0.0, 0.0
            next_all_red = ""

        # GREEN logic
        if self.phase in ("A_GREEN", "B_GREEN"):
            if detected:
                self.last_det_t = t

            # enforce minimum green
            if self.phase_t < Gmin:
                return (self.phase == "A_GREEN"), (self.phase == "B_GREEN")

            # max green cap
            if self.phase_t >= Gmax:
                self.phase = next_all_red
                self.phase_t = 0.0
                return False, False

            # gap-out
            if (t - self.last_det_t) >= self.gap:
                self.phase = next_all_red
                self.phase_t = 0.0
                return False, False

            return (self.phase == "A_GREEN"), (self.phase == "B_GREEN")

        # ALL-RED logic
        if self.phase in ("ALL_RED_AB", "ALL_RED_BA"):
            if self.phase_t < self.all_red:
                return False, False

            # after all-red -> switch green
            if self.phase == "ALL_RED_AB":
                self.phase = "B_GREEN"
            else:
                self.phase = "A_GREEN"

            self.phase_t = 0.0
            self.last_det_t = t  # prevent instant gap-out
            return (self.phase == "A_GREEN"), (self.phase == "B_GREEN")

        # fallback
        return False, False


# -----------------------------
# Detector (presence in last X meters)
# -----------------------------
def detected_near_stopline(vehicles: list[Vehicle], cfg: SimConfig, detect_zone: float = 25.0) -> bool:
    for veh in vehicles:
        if veh.done:
            continue
        dist_to_stop = cfg.stop_x - veh.x
        if 0.0 <= dist_to_stop <= detect_zone:
            return True
    return False


# -----------------------------
# Vehicle update for one approach
# -----------------------------
def step_approach(
    vehicles: list[Vehicle],
    cfg: SimConfig,
    idm_p: IDMParams,
    dt: float,
    t: float,
    green: bool,
):
    """
    Updates all active vehicles for ONE approach, using:
    - normal IDM car-following
    - virtual stopline when red (like in your proposed.py)
    """
    active_idx = [i for i, v in enumerate(vehicles) if not v.done]
    active_idx.sort(key=lambda i: vehicles[i].x, reverse=True)  # closest to stopline first

    for pos, i in enumerate(active_idx):
        ego = vehicles[i]

        # (1) gap to preceding vehicle
        if pos == 0:
            s_veh = 1e9
            dv_veh = 0.0
        else:
            lead = vehicles[active_idx[pos - 1]]
            s_veh = (lead.x - ego.x) - cfg.veh_len
            dv_veh = ego.v - lead.v

        # (2) stopline virtual leader if red
        if not green:
            s_sig = (cfg.stop_x - ego.x) - cfg.veh_len
            dv_sig = ego.v
        else:
            s_sig = 1e9
            dv_sig = 0.0

        # (3) most restrictive
        if s_sig < s_veh:
            s_eff, dv_eff = s_sig, dv_sig
        else:
            s_eff, dv_eff = s_veh, dv_veh

        acc = idm_accel(ego.v, s_eff, dv_eff, idm_p)
        v_new = max(0.0, ego.v + acc * dt)
        x_new = ego.x + v_new * dt

        # cannot cross stopline on red
        if not green and x_new > -0.5:
            x_new = -0.5
            v_new = 0.0

        # queue proxy
        if -cfg.queue_zone <= x_new <= cfg.stop_x and v_new < cfg.speed_stop_th and not ego.done:
            ego.stopped_time += dt

        moving_now = v_new > cfg.speed_stop_th
        if ego.moving_prev and not moving_now:
            ego.stops += 1
        ego.moving_prev = moving_now

        # exit if crosses on green
        if green and x_new >= cfg.stop_x:
            ego.done = True
            ego.exit_t = t

        ego.x, ego.v = x_new, v_new


# -----------------------------
# Run intersection (two approaches)
# -----------------------------
@dataclass
class IntersectionDemand:
    lam_A: float = 0.6  # veh/s
    lam_B: float = 0.3  # veh/s


def run_intersection(
    cfg: SimConfig,
    idm_p: IDMParams,
    controller,  # FixedTimeTwoPhaseSignal or ActuatedGapOutTwoPhaseSignal
    demand: IntersectionDemand,
    detect_zone: float = 25.0,
):
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.dt
    times = np.arange(0.0, cfg.T_end + dt, dt)

    A: list[Vehicle] = []
    B: list[Vehicle] = []

    qA = np.zeros_like(times)
    qB = np.zeros_like(times)
    msA = np.zeros_like(times)
    msB = np.zeros_like(times)

    for k, t in enumerate(times):
        # detectors
        det_A = detected_near_stopline(A, cfg, detect_zone)
        det_B = detected_near_stopline(B, cfg, detect_zone)

        # controller step
        if isinstance(controller, FixedTimeTwoPhaseSignal):
            A_green, B_green = controller.step(dt)
        else:
            A_green, B_green = controller.step(t, dt, det_A, det_B)

        # spawn on each approach independently
        if poisson_spawn(rng, demand.lam_A, dt):
            A.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t))
        if poisson_spawn(rng, demand.lam_B, dt):
            B.append(Vehicle(x=-cfg.L, v=0.0, spawned_t=t))

        # update both approaches
        step_approach(A, cfg, idm_p, dt, t, A_green)
        step_approach(B, cfg, idm_p, dt, t, B_green)

        # metrics
        qA[k] = compute_queue(A, cfg)
        qB[k] = compute_queue(B, cfg)
        msA[k] = mean_speed_near(A)
        msB[k] = mean_speed_near(B)

    return {
        "t": times,
        "A": {"vehicles": A, "queue": qA, "mean_speed": msA},
        "B": {"vehicles": B, "queue": qB, "mean_speed": msB},
    }
