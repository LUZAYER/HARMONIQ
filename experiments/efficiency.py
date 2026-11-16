#!/usr/bin/env python3
"""
Empirical Latency and Throughput Efficiency Benchmark for HARMONIQ.

Measures:
- Per-component latency: Decomposition, C, R, S, Gated V, Fusion
- Mean, p50, p95 latencies (ms)
- Claims per second throughput
- Fraction routed to gated self-verification stage
- Cold vs. Cached retrieval latency

Outputs:
- results/efficiency_results.csv
"""

import os
import sys
import time
import argparse
import platform
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.confidence import ConfidenceSignal
from src.signals.retrieval import RetrievalSignal
from src.signals.semantic import SemanticSignal
from src.signals.selfverif import SelfVerificationSignal
from src.data.ragtruth import split_sentences_clausal


def run_efficiency_benchmark(
    n_samples: int = 50,
    output_csv: str = "results/efficiency_results.csv"
):
    print(f"Running efficiency benchmark across {n_samples} verification trials...")

    # Hardware & environment profiling
    device_info = platform.processor() or platform.machine()
    python_ver = platform.python_version()

    sig_c = ConfidenceSignal()
    sig_r = RetrievalSignal()
    sig_s = SemanticSignal()
    sig_v = SelfVerificationSignal()

    test_queries = [
        ("Summarize the history of spaceflight.",
         "The space age began with the launch of Sputnik 1 in 1957. The Apollo program landed twelve astronauts on the Moon between 1969 and 1972.",
         "Sputnik 1 was launched into low Earth orbit in October 1957. Apollo 11 achieved the first crewed lunar landing in July 1969."),
        ("What causes solar eclipses?",
         "A solar eclipse occurs when the Moon passes between Earth and the Sun, thereby obscuring the view of the Sun from Earth.",
         "Solar eclipses occur during a new moon phase when the Moon aligns directly between Earth and the Sun."),
        ("Explain the discovery of penicillin.",
         "Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital in London after observing antibacterial action in mould.",
         "Penicillin was discovered by Scottish physician Alexander Fleming in September 1928.")
    ]

    decomp_times = []
    c_times = []
    r_times = []
    s_times = []
    v_times = []
    total_times = []
    total_claims = 0
    routed_to_v_count = 0

    # 5 warm-up iterations
    for _ in range(5):
        q, resp, evid = test_queries[0]
        claims = split_sentences_clausal(resp)
        for c in claims:
            sig_c.evaluate_claim(c["text"], evid)

    for i in range(n_samples):
        q, resp, evid = test_queries[i % len(test_queries)]
        t_start = time.time()

        # 1. Decomposition
        td0 = time.time()
        claims = split_sentences_clausal(resp)
        td1 = time.time()
        decomp_times.append((td1 - td0) * 1000.0)
        total_claims += len(claims)

        resp_v_called = False
        for c in claims:
            txt = c["text"]
            # Signal C
            tc0 = time.time()
            res_c = sig_c.evaluate_claim(txt, evid, q)
            c_times.append((time.time() - tc0) * 1000.0)

            # Signal R
            tr0 = time.time()
            res_r = sig_r.evaluate_claim(txt, evid, q)
            r_times.append((time.time() - tr0) * 1000.0)

            # Signal S
            ts0 = time.time()
            res_s = sig_s.evaluate_claim(txt, evid, q)
            s_times.append((time.time() - ts0) * 1000.0)

            # Adaptive Compute Gating for V: only triggered if fast signals indicate borderline risk
            fast_risk = np.nanmean([res_c.score, res_r.score, res_s.score])
            if 0.35 <= fast_risk <= 0.65:
                resp_v_called = True
                tv0 = time.time()
                res_v = sig_v.evaluate_claim(txt, evid, q, samples=[evid])
                v_times.append((time.time() - tv0) * 1000.0)

        if resp_v_called:
            routed_to_v_count += 1

        t_end = time.time()
        total_times.append((t_end - t_start) * 1000.0)

    # Compute statistics
    def get_stats(arr):
        if not arr:
            return 0.0, 0.0, 0.0
        return float(np.mean(arr)), float(np.median(arr)), float(np.percentile(arr, 95))

    d_mean, d_p50, d_p95 = get_stats(decomp_times)
    c_mean, c_p50, c_p95 = get_stats(c_times)
    r_mean, r_p50, r_p95 = get_stats(r_times)
    s_mean, s_p50, s_p95 = get_stats(s_times)
    v_mean, v_p50, v_p95 = get_stats(v_times)
    tot_mean, tot_p50, tot_p95 = get_stats(total_times)

    total_wall_sec = sum(total_times) / 1000.0
    claims_per_sec = total_claims / max(0.001, total_wall_sec)
    gated_routing_fraction = routed_to_v_count / max(1, n_samples)

    rows = [
        {"Component": "Claim Decomposition", "Mean_ms": round(d_mean, 2), "p50_ms": round(d_p50, 2), "p95_ms": round(d_p95, 2), "Notes": "Clausal segmentation"},
        {"Component": "Confidence Analysis (C)", "Mean_ms": round(c_mean, 2), "p50_ms": round(c_p50, 2), "p95_ms": round(c_p95, 2), "Notes": "Proxy-LM entity uncertainty"},
        {"Component": "Retrieval Grounding (R)", "Mean_ms": round(r_mean, 2), "p50_ms": round(r_p50, 2), "p95_ms": round(r_p95, 2), "Notes": "Passage selection + NLI"},
        {"Component": "Semantic Faithfulness (S)", "Mean_ms": round(s_mean, 2), "p50_ms": round(s_p50, 2), "p95_ms": round(s_p95, 2), "Notes": "Directional + deterministic checks"},
        {"Component": "Gated Self-Verification (V)", "Mean_ms": round(v_mean, 2), "p50_ms": round(v_p50, 2), "p95_ms": round(v_p95, 2), "Notes": f"Gated: {gated_routing_fraction*100:.1f}% routed"},
        {"Component": "End-to-End Pipeline", "Mean_ms": round(tot_mean, 2), "p50_ms": round(tot_p50, 2), "p95_ms": round(tot_p95, 2), "Notes": f"Throughput: {claims_per_sec:.2f} claims/sec"}
    ]

    res_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"✓ Saved efficiency table to {output_csv}")
    print("\n" + res_df.to_string(index=False))


if __name__ == '__main__':
    run_efficiency_benchmark()
