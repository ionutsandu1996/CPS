# run_intersection_experiment.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from intersection import SimConfig, IDMParams, compute_metrics
from baseline import FixedTime2PhaseSignal, run_baseline
from proposed import ActuatedGapOut2PhaseSignal, run_proposed


def plot_time_series(base: dict, prop: dict):
    # 1) Queue over time (total)
    plt.figure()
    plt.plot(base["t"], base["queue_total"], label="Baseline queue (A+B)")
    plt.plot(prop["t"], prop["queue_total"], label="Proposed queue (A+B)")
    plt.xlabel("Time [s]")
    plt.ylabel("Queue length [veh] (last queue_zone m per approach)")
    plt.title("Queue length over time (intersection)")
    plt.legend()
    plt.savefig("queue_over_time.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Mean speed near intersection (total)
    plt.figure()
    plt.plot(base["t"], base["mean_speed_total"], label="Baseline mean speed")
    plt.plot(prop["t"], prop["mean_speed_total"], label="Proposed mean speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean speed [m/s] (avg over approaches, last speed_zone m)")
    plt.title("Mean approach speed near intersection")
    plt.legend()
    plt.savefig("mean_speed.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_bars(summary: pd.DataFrame):
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
    plt.bar(summary["scenario"], summary["avg_queue_total"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Average queue [veh] (A+B)")
    plt.title("Average queue comparison (mean over seeds)")
    plt.savefig("metric_bars_queue.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Throughput bars
    plt.figure()
    plt.bar(summary["scenario"], summary["throughput_total"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Throughput [veh] (A+B)")
    plt.title("Throughput comparison (mean over seeds)")
    plt.savefig("metric_bars_throughput.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Green share bars (A green share)
    plt.figure()
    plt.bar(summary["scenario"], summary["green_A_share"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Green share for phase A [-]")
    plt.title("Green time share (phase A) (mean over seeds)")
    plt.savefig("metric_bars_green_share.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Served per green (overall)
    plt.figure()
    plt.bar(summary["scenario"], summary["served_per_green_total"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Served rate [veh/s]")
    plt.title("Served rate (throughput / total time) (mean over seeds)")
    plt.savefig("metric_bars_served_per_green.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    # ------------------
    # Experiment settings
    # ------------------
    seeds = [1, 2, 3, 4, 5]
    T_end = 900.0

    # Demand (veh/s) per approach
    arrival_A = 0.60
    arrival_B = 0.60

    # Measurement zones
    queue_zone = 250.0
    speed_zone = 200.0
    detect_zone = 25.0

    # Baseline: fixed-time phase lengths (A then B)
    fixed_green_A = 25.0
    fixed_green_B = 25.0

    # Proposed: actuated gap-out
    G_min = 10.0
    G_max = 45.0
    gap_threshold = 2.0

    # ------------------
    # Run experiments
    # ------------------
    idm_p = IDMParams()
    rows = []

    representative_base = None
    representative_prop = None

    for seed in seeds:
        cfg = SimConfig(
            seed=seed,
            T_end=T_end,
            arrival_rate_A=arrival_A,
            arrival_rate_B=arrival_B,
            queue_zone=queue_zone,
            speed_zone=speed_zone,
            detect_zone=detect_zone,
        )

        # Baseline run (fixed-time 2-phase)
        fixed = FixedTime2PhaseSignal(green_A=fixed_green_A, green_B=fixed_green_B, start="A")
        out_base = run_baseline(cfg, idm_p, fixed)
        m_base = compute_metrics(cfg, out_base, idm_p)
        rows.append({"seed": seed, "scenario": "Baseline: fixed-time (2-phase)", **m_base})

        # Proposed run (actuated gap-out 2-phase)
        sig = ActuatedGapOut2PhaseSignal(G_min=G_min, G_max=G_max, gap_threshold=gap_threshold, start="A")
        out_prop = run_proposed(cfg, idm_p, sig)
        m_prop = compute_metrics(cfg, out_prop, idm_p)
        rows.append({"seed": seed, "scenario": "Proposed: actuated gap-out (2-phase)", **m_prop})

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
    print("Generated: metric_bars_delay.png, metric_bars_queue.png")
    print("Generated: metric_bars_throughput.png, metric_bars_green_share.png, metric_bars_served_per_green.png")


if __name__ == "__main__":
    main()
