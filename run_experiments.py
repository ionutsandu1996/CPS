# run_experiments.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import SimConfig, IDMParams, FixedTimeSignal, run_baseline
from proposed import ActuatedGapOutSignal, run_proposed


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(cfg: SimConfig, out: dict, idm_p: IDMParams) -> dict:
    """
    Computes metrics from a simulation output dict.

    Key ideas:
      - "throughput": how many vehicles completed (crossed stop line)
      - "queue": average / maximum queue length
      - "travel_time": spawn -> exit for completed vehicles
      - "delay": travel_time - free_flow_time (how much extra time due to signal/queues)
      - "stops": how many stop events per vehicle (comfort / stop-and-go)
      - "stopped_time": time spent nearly stopped near stop line (proxy)
    """
    vehicles = out["vehicles"]
    done = [v for v in vehicles if v.exit_t is not None]

    # Free-flow travel time: "how fast would it be with no signal & no traffic"
    # Approx = distance / desired speed
    free_tt = cfg.L / max(idm_p.v0, 0.1)

    travel_times = np.array([v.exit_t - v.spawned_t for v in done]) if done else np.array([])
    delays = travel_times - free_tt if travel_times.size else np.array([])

    stops = np.array([v.stops for v in done]) if done else np.array([])
    stopped_time = np.array([v.stopped_time for v in done]) if done else np.array([])

    return {
        "throughput": len(done),
        "N_spawned": len(vehicles),
        "avg_queue": float(np.mean(out["queue"])),
        "max_queue": int(np.max(out["queue"])),
        "avg_travel_time_s": float(np.mean(travel_times)) if travel_times.size else np.nan,
        "avg_delay_s": float(np.mean(delays)) if delays.size else np.nan,
        "p95_delay_s": float(np.percentile(delays, 95)) if delays.size else np.nan,
        "avg_stops_per_vehicle": float(np.mean(stops)) if stops.size else np.nan,
        "avg_stopped_time_s": float(np.mean(stopped_time)) if stopped_time.size else np.nan,
    }


# -----------------------------
# Plots
# -----------------------------
def plot_time_series(base: dict, prop: dict):
    """
    Generates two time-series figures:
      - queue_over_time.png
      - mean_speed.png
    """

    # 1) Queue over time
    plt.figure()
    plt.plot(base["t"], base["queue"], label="Baseline (fixed-time) queue")
    plt.plot(prop["t"], prop["queue"], label="Proposed (actuated) queue")
    plt.xlabel("Time [s]")
    plt.ylabel("Queue length [veh] (last queue_zone m)")
    plt.title("Queue length over time")
    plt.legend()
    plt.savefig("queue_over_time.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Mean speed over time
    plt.figure()
    plt.plot(base["t"], base["mean_speed"], label="Baseline mean speed")
    plt.plot(prop["t"], prop["mean_speed"], label="Proposed mean speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean speed [m/s] (last speed_zone m)")
    plt.title("Mean approach speed near stop line")
    plt.legend()
    plt.savefig("mean_speed.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_bars(summary: pd.DataFrame):
    """
    Quick bar charts for key metrics.
    Produces:
      - metric_bars_delay.png
      - metric_bars_queue.png
      - metric_bars_throughput.png
    """

    # Delay bars
    plt.figure()
    plt.bar(summary["scenario"], summary["avg_delay_s"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Average delay [s]")
    plt.title("Average delay comparison (mean over seeds)")
    plt.savefig("metric_bars_delay.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Queue bars
    plt.figure()
    plt.bar(summary["scenario"], summary["avg_queue"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Average queue [veh]")
    plt.title("Average queue comparison (mean over seeds)")
    plt.savefig("metric_bars_queue.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Throughput bars
    plt.figure()
    plt.bar(summary["scenario"], summary["throughput"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Throughput [veh]")
    plt.title("Throughput comparison (mean over seeds)")
    plt.savefig("metric_bars_throughput.png", dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main experiment runner
# -----------------------------
def main():
    # ------------------
    # 1) Experiment settings
    # ------------------
    seeds = [1, 2, 3, 4, 5]
    T_end = 900.0

    # Demand: vehicles per second.
    # If you want to "see queues" go higher (e.g., 0.8 - 1.2).
    arrival_rate = 0.60

    # Measurement zone: how far upstream we count "queue"
    queue_zone = 250.0

    # Baseline signal timings (fixed-time)
    fixed_green = 25.0
    fixed_red = 40.0

    # Proposed signal parameters (actuated)
    G_min = 10.0
    G_max = 45.0
    R_min = 10.0
    gap_threshold = 2.0
    detect_zone = 25.0

    # ------------------
    # 2) Run experiments
    # ------------------
    idm_p = IDMParams()
    rows = []

    representative_base = None
    representative_prop = None

    for seed in seeds:
        # same config for both baseline and proposed
        cfg = SimConfig(
            arrival_rate=arrival_rate,
            seed=seed,
            T_end=T_end,
            queue_zone=queue_zone,
        )

        # ---- Baseline: fixed-time ----
        fixed = FixedTimeSignal(green_s=fixed_green, red_s=fixed_red, start_green=True)
        out_base = run_baseline(cfg, idm_p, fixed)
        m_base = compute_metrics(cfg, out_base, idm_p)
        rows.append({"seed": seed, "scenario": "Baseline: IDM + fixed-time", **m_base})

        # ---- Proposed: actuated gap-out ----
        sig = ActuatedGapOutSignal(G_min=G_min, G_max=G_max, R_min=R_min, gap_threshold=gap_threshold)
        out_prop = run_proposed(cfg, idm_p, sig, detect_zone=detect_zone)
        m_prop = compute_metrics(cfg, out_prop, idm_p)
        rows.append({"seed": seed, "scenario": "Proposed: CPS-IDM + actuated", **m_prop})

        # keep the first seed run for time-series plots
        if representative_base is None:
            representative_base = out_base
            representative_prop = out_prop

    # ------------------
    # 3) Save tables (CSV)
    # ------------------
    df = pd.DataFrame(rows)
    df.to_csv("runs.csv", index=False)

    summary = df.groupby("scenario").mean(numeric_only=True).reset_index()
    summary.to_csv("summary.csv", index=False)

    # ------------------
    # 4) Save plots (PNG)
    # ------------------
    plot_time_series(representative_base, representative_prop)
    plot_metric_bars(summary)

    print("Done.")
    print("Generated: runs.csv, summary.csv")
    print("Generated: queue_over_time.png, mean_speed.png")
    print("Generated: metric_bars_delay.png, metric_bars_queue.png, metric_bars_throughput.png")


if __name__ == "__main__":
    main()
