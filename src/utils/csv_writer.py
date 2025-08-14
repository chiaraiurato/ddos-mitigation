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
