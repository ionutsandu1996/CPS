# run_experiments.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import SimConfig, IDMParams, FixedTimeSignal, run_baseline
from proposed import ActuatedGapOutSignal, run_proposed


def compute_metrics(cfg: SimConfig, out: dict, idm_p: IDMParams) -> dict:
    """
    Compute key metrics from one simulation output.

    Metrics:
      - throughput: number of vehicles that passed the stop line (completed)
      - avg/max queue: from queue time-series
      - avg travel time: spawned -> exit, for completed vehicles
      - avg/p95 delay: delay = travel_time - free_flow_time
      - avg stops per vehicle
      - avg stopped time near stop line (proxy for time spent waiting)
    """
    vehicles = out["vehicles"]
    done = [v for v in vehicles if v.exit_t is not None]

    # free flow travel time (approx) over length L at desired speed v0
    free_tt = cfg.L / max(idm_p.v0, 0.1)

    travel_times = np.array([v.exit_t - v.spawned_t for v in done]) if done else np.array([])
    delays = travel_times - free_tt if travel_times.size else np.array([])

    stops = np.array([v.stops for v in done]) if done else np.array([])
    stopped_time = np.array([v.stopped_time for v in done]) if done else np.array([])

    return {
        "throughput": len(done),
        "avg_queue": float(np.mean(out["queue"])),
        "max_queue": int(np.max(out["queue"])),
        "avg_travel_time_s": float(np.mean(travel_times)) if travel_times.size else np.nan,
        "avg_delay_s": float(np.mean(delays)) if delays.size else np.nan,
        "p95_delay_s": float(np.percentile(delays, 95)) if delays.size else np.nan,
        "avg_stops_per_vehicle": float(np.mean(stops)) if stops.size else np.nan,
        "avg_stopped_time_s": float(np.mean(stopped_time)) if stopped_time.size else np.nan,
        "N_spawned": len(vehicles),
    }


def plot_time_series(base: dict, prop: dict):
    """
    Generates:
      - queue_over_time.png
      - mean_speed.png
    """
    # 1) Queue over time
    plt.figure()
    plt.plot(base["t"], base["queue"], label="Baseline: fixed-time")
    plt.plot(prop["t"], prop["queue"], label="Proposed: actuated CPS")
    plt.xlabel("Time [s]")
    plt.ylabel("Queue length [veh] (within queue_zone)")
    plt.title("Queue length over time")
    plt.legend()
    plt.savefig("queue_over_time.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Mean speed over time
    plt.figure()
    plt.plot(base["t"], base["mean_speed"], label="Baseline: fixed-time")
    plt.plot(prop["t"], prop["mean_speed"], label="Proposed: actuated CPS")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean speed [m/s] (within speed_zone)")
    plt.title("Mean approach speed near intersection")
    plt.legend()
    plt.savefig("mean_speed.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_bars(summary: pd.DataFrame):
    """
    Generates:
      - metric_bars_delay.png
      - metric_bars_queue.png
      - metric_bars_throughput.png
    """
    # Delay bars
    plt.figure()
    plt.bar(summary["scenario"], summary["avg_delay_s"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Average delay [s]")
    plt.title("Average delay (mean over seeds)")
    plt.savefig("metric_bars_delay.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Queue bars
    plt.figure()
    plt.bar(summary["scenario"], summary["avg_queue"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Average queue [veh]")
    plt.title("Average queue (mean over seeds)")
    plt.savefig("metric_bars_queue.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Throughput bars
    plt.figure()
    plt.bar(summary["scenario"], summary["throughput"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Throughput [veh] (completed)")
    plt.title("Throughput (mean over seeds)")
    plt.savefig("metric_bars_throughput.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    # ------------------
    # Realistic experiment settings (single-lane, heavy traffic)
    # ------------------
    arrival_rate = 0.40     # veh/s  (~1440 veh/h) => heavy / rush-like for one lane
    seeds = [1, 2, 3, 4, 5]
    T_end = 1800.0          # 30 minutes

    # Baseline signal timings (realistic-ish urban cycle ~75s)
    fixed_green = 30.0
    fixed_red = 45.0

    # Proposed actuated signal settings
    G_min = 15.0
    G_max = 60.0
    R_min = 15.0
    gap_threshold = 2.5
    detect_zone = 30.0

    # Measurement definitions (what counts as "queue" and "near intersection")
    queue_zone = 150.0
    speed_zone = 200.0

    # ------------------
    # Run experiments
    # ------------------
    idm_p = IDMParams()
    rows = []

    representative_base = None
    representative_prop = None

    for seed in seeds:
        cfg = SimConfig(
            arrival_rate=arrival_rate,
            seed=seed,
            T_end=T_end,
            queue_zone=queue_zone,
            speed_zone=speed_zone,
        )

        # --- Baseline run ---
        fixed = FixedTimeSignal(green_s=fixed_green, red_s=fixed_red, start_green=True)
        out_base = run_baseline(cfg, idm_p, fixed)
        m_base = compute_metrics(cfg, out_base, idm_p)
        rows.append({"seed": seed, "scenario": "Baseline: IDM + fixed-time", **m_base})

        # --- Proposed run ---
        sig = ActuatedGapOutSignal(G_min=G_min, G_max=G_max, R_min=R_min, gap_threshold=gap_threshold)
        out_prop = run_proposed(cfg, idm_p, sig, detect_zone=detect_zone)
        m_prop = compute_metrics(cfg, out_prop, idm_p)
        rows.append({"seed": seed, "scenario": "Proposed: CPS-IDM + actuated", **m_prop})

        # keep first run for time-series plots
        if representative_base is None:
            representative_base = out_base
            representative_prop = out_prop

    # ------------------
    # Save tables
    # ------------------
    df = pd.DataFrame(rows)
    df.to_csv("runs.csv", index=False)

    summary = df.groupby("scenario").mean(numeric_only=True).reset_index()
    summary.to_csv("summary.csv", index=False)

    # ------------------
    # Save plots
    # ------------------
    plot_time_series(representative_base, representative_prop)
    plot_metric_bars(summary)

    print("Done.")
    print("Generated: runs.csv, summary.csv")
    print("Generated: queue_over_time.png, mean_speed.png")
    print("Generated: metric_bars_delay.png, metric_bars_queue.png, metric_bars_throughput.png")


if __name__ == "__main__":
    main()
