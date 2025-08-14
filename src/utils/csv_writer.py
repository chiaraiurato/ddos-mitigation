import os, csv
from engineering.costants import MAX_SPIKE_NUMBER

def build_header():
    base = [
        "scenario","ARRIVAL_P","ARRIVAL_L1","ARRIVAL_L2",
        "total_time","total_arrivals",
        "web_util","web_rt_mean","web_throughput",                       
        "mit_util","mit_rt_mean","mit_throughput",
        "drop_fp_rate","drop_full_rate", "spikes_count"
    ]
    # colonne per spike0..spike{MAX_SPIKE_NUMBER-1}
    for i in range(MAX_SPIKE_NUMBER):
        base += [
            f"spike{i}_util",
            f"spike{i}_rt_mean",
            f"spike{i}_throughput",
            f"spike{i}_completions",
        ]
    return base

CSV_HEADER = build_header()

def _check_or_write_header(csv_path: str, writer: csv.DictWriter):
    exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    if not exists:
        writer.writeheader()

def append_row(csv_path: str, row: dict):
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        _check_or_write_header(csv_path, w)
        w.writerow(row)
