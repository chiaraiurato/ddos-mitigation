import os, csv
from engineering.costants import MAX_SPIKE_NUMBER

def build_header():
    base = [
        "scenario","ARRIVAL_P","ARRIVAL_L1","ARRIVAL_L2",
        "total_time","total_arrivals",
        
        "web_util_bm","web_rt_mean_bm","web_throughput_bm",
        
        "mit_util_bm","mit_rt_mean_bm","mit_throughput_bm",
       
        "drop_fp_rate","drop_full_rate",
        "spikes_count",
    ]
    
    for i in range(MAX_SPIKE_NUMBER):
        base += [
            f"spike{i}_util_bm",
            f"spike{i}_rt_mean_bm",
            f"spike{i}_throughput_bm",
            f"spike{i}_completions",
        ]
    return base

CSV_HEADER = build_header()

def append_row(csv_path: str, row: dict):
    new_file = not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new_file:
            w.writeheader()
        w.writerow(row)


def append_row_stable(path: str, row: dict, fieldnames: list):
    
    exists = os.path.exists(path)
    safe_row = {k: row.get(k, "") for k in fieldnames}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(safe_row)

def validation_fieldnames():
    return [
        "scenario", "ARRIVAL_P", "ARRIVAL_L1", "ARRIVAL_L2",
        "total_time", "total_arrivals",
        "web_util", "web_rt_mean", "web_throughput",
        "spikes_count", "spike0_util", "spike0_rt_mean", "spike0_throughput",
        "mit_util", "mit_rt_mean", "mit_throughput",
        "drop_fp_rate", "drop_full_rate",
        "web_util_bm_mean", "web_util_bm_ci",
        "web_thr_bm_mean",  "web_thr_bm_ci",
        "web_rt_bm_mean",   "web_rt_bm_ci",
        "spike0_util_bm_mean", "spike0_util_bm_ci",
        "spike0_thr_bm_mean",  "spike0_thr_bm_ci",
        "spike0_rt_bm_mean",   "spike0_rt_bm_ci",
        "mit_util_bm_mean", "mit_util_bm_ci",
        "mit_thr_bm_mean",  "mit_thr_bm_ci",
        "mit_rt_bm_mean",   "mit_rt_bm_ci",
        "analysis_util", "analysis_rt_mean", "analysis_throughput",
        "ana_util_bm_mean", "ana_util_bm_ci",
        "ana_thr_bm_mean",  "ana_thr_bm_ci",
        "ana_rt_bm_mean",   "ana_rt_bm_ci",
        "bm_rt_batch_size", "bm_win_size", "bm_windows_per_batch"
    ]

def transitory_fieldnames(max_spikes: int):
    base = [
        "replica", "time",
        "web_rt_mean", "web_util", "web_throughput",
        "mit_rt_mean", "mit_util", "mit_throughput",
        "analysis_rt_mean", "analysis_util", "analysis_throughput",
        "system_rt_mean",
        "arrivals_so_far", "false_positives_so_far", "mitigation_completions_so_far",
        "spikes_count", "scenario", "mode", "is_final",
        "illegal_share",
        "processed_legal_share", "processed_illegal_share",
        "completed_legal_share", "completed_illegal_share",
        "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
    ]
    for i in range(max_spikes):
        base += [f"spike{i}_rt_mean", f"spike{i}_util", f"spike{i}_throughput"]
    return base

def infinite_fieldnames(max_spikes: int):
    base = [
        "scenario",
        "ARRIVAL_P", "ARRIVAL_L1", "ARRIVAL_L2",
        "total_time", "total_arrivals",
        "burn_in_rt", "batch_size", "n_batches", "confidence",
        "web_rt_mean_bm", "web_rt_ci_hw",
        "web_util_point", "web_throughput_point",
        "mit_rt_mean_bm", "mit_rt_ci_hw",
        "mit_util_point", "mit_throughput_point",
        "system_rt_mean_bm", "system_rt_ci_hw",
        "illegal_share",
        "processed_legal_share", "processed_illegal_share",
        "completed_legal_share", "completed_illegal_share",
        "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
        "spikes_count",
    ]
    for i in range(max_spikes):
        base += [f"spike{i}_rt_mean_bm", f"spike{i}_rt_ci_hw",
                 f"spike{i}_util_point", f"spike{i}_thr_point",
                 f"spike{i}_completions"]
    return base