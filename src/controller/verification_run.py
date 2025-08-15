import simpy
import numpy as np
import os
from library.rngs import random, plantSeeds
from engineering.costants import *
from engineering.distributions import get_interarrival_time
from engineering.statistics import batch_means, window_util_thr
from utils.csv_writer import append_row
from model.job import Job
from model.processor_sharing_server import ProcessorSharingServer
from model.mitigation_manager import MitigationManager
from utils.checkpoint import _rt_mean_upto, _count_upto, _utilization_upto, _throughput_upto
from utils.csv_writer import append_row_stable

class DDoSSystem:
    def __init__(self, env, mode, arrival_p, arrival_l1, arrival_l2, enable_ci=False):
        self.env = env
        self.mode = mode
        self.enable_ci = enable_ci
        self.arrival_p = arrival_p
        self.arrival_l1 = arrival_l1
        self.arrival_l2 = arrival_l2

        self.web_server = ProcessorSharingServer(env, "Web")
        self.spike_servers = [ProcessorSharingServer(env, "Spike-0")]
        
        # Metriche globali
        self.metrics = {
            "mitigation_completions": 0,
            "total_arrivals": 0,
            "false_positives": 0,
            "legal_arrivals": 0,
            "illegal_arrivals": 0,
            "false_positives_legal": 0,
        }

        self.mitigation_manager = MitigationManager(
            env, self.web_server, self.spike_servers, self.metrics, self.mode
        )

        self.env.process(self.arrival_process(mode))

    def snapshot(self, when, replica_id=None):
        """
        Ritorna un dict con le metriche cumulative (RT mean, Util, Thr)
        calcolate FINO al tempo 'when' per Web, Mitigation e Spike i.
        """
        # Assicurati che gli stati 'busy' siano aggiornati al tempo 'when'
        self.web_server.update(when)
        for s in self.spike_servers:
            s.update(when)
        center = self.mitigation_manager.center
        center.update(when)

        row = {
            "replica": int(replica_id) if replica_id is not None else None,
            "time": float(when),

            # Web
            "web_rt_mean": _rt_mean_upto(self.web_server.completed_jobs,
                                         self.web_server.completion_times, when),
            "web_util": _utilization_upto(self.web_server.busy_periods, when),
            "web_throughput": _throughput_upto(self.web_server.completion_times, when),

            # Mitigation
            "mit_rt_mean": _rt_mean_upto(center.completed_jobs,
                                         center.completion_times, when),
            "mit_util": _utilization_upto(center.busy_periods, when),
            "mit_throughput": _throughput_upto(center.completion_times, when),

            # Global counters cumulativi fino a 'when' (per completezza)
            "arrivals_so_far": int(min(self.metrics["total_arrivals"], 
                                       self.metrics["total_arrivals"])),  # già cumulativi
            "false_positives_so_far": int(self.metrics.get("false_positives", 0)),
            "mitigation_completions_so_far": int(self.metrics.get("mitigation_completions", 0)),
            "spikes_count": int(len(self.spike_servers)),
        }

        # Per-spike
        for i in range(len(self.spike_servers)):
            s = self.spike_servers[i]
            row[f"spike{i}_rt_mean"] = _rt_mean_upto(s.completed_jobs, s.completion_times, when)
            row[f"spike{i}_util"]    = _utilization_upto(s.busy_periods, when)
            row[f"spike{i}_throughput"] = _throughput_upto(s.completion_times, when)

        # Se vuoi colonne fisse fino a MAX_SPIKE_NUMBER:
        for i in range(len(self.spike_servers), MAX_SPIKE_NUMBER):
            row[f"spike{i}_rt_mean"] = None
            row[f"spike{i}_util"] = None
            row[f"spike{i}_throughput"] = None

        return row

    def arrival_process(self, mode):

        if mode == "verification":
            arrivals = N_ARRIVALS_VERIFICATION
            interarrival_mode = "verification"
        elif mode == "transitory":
            interarrival_mode = "standard"
        else:  # "standard"
            arrivals = N_ARRIVALS 
            interarrival_mode = "standard"

        if mode == "transitory":
            while True:
                interarrival_time = get_interarrival_time(interarrival_mode, self.arrival_p, self.arrival_l1, self.arrival_l2)
                yield self.env.timeout(interarrival_time)

                self.metrics["total_arrivals"] += 1
                arrival_time = self.env.now
                is_legal = random() < P_LECITO
                if is_legal:
                    self.metrics["legal_arrivals"] += 1
                else:
                    self.metrics["illegal_arrivals"] += 1

                job = Job(self.metrics["total_arrivals"], arrival_time, None, is_legal)
                self.mitigation_manager.handle_job(job)
        else:   
            while self.metrics["total_arrivals"] < arrivals:
                interarrival_time = get_interarrival_time(interarrival_mode, self.arrival_p, self.arrival_l1, self.arrival_l2)
                yield self.env.timeout(interarrival_time)

                self.metrics["total_arrivals"] += 1
                arrival_time = self.env.now
                is_legal = random() < P_LECITO
                if is_legal:
                    self.metrics["legal_arrivals"] += 1
                else:
                    self.metrics["illegal_arrivals"] += 1

                job = Job(self.metrics["total_arrivals"], arrival_time, None, is_legal)
                self.mitigation_manager.handle_job(job)

    def notify_completion(self, server_name, response_time):
        # Usato solo se imposti self.web_server.set_observer(self)
        if self.mode != "verification" or server_name != "Web":
            return
        self.completed_in_batch += 1
        self.sum_rt_in_batch += response_time
        if self.completed_in_batch == BATCH_SIZE:
            batch_mean = self.sum_rt_in_batch / BATCH_SIZE
            self.batch_means.append(batch_mean)
            self.completed_in_batch = 0
            self.sum_rt_in_batch = 0
            self.batch_count += 1

    def make_csv_row(self, scenario: str, arrival_p: float, arrival_l1: float, arrival_l2: float):
        """
        Costruisce la riga CSV.
        """
        now = self.env.now
        total = max(1, self.metrics["total_arrivals"])  

        def bm_mean_or_blank(samples, bsize):
            try:
                if len(samples) < bsize or (len(samples) // bsize) < 2:
                    return ""  # non abbastanza campioni per >=2 batch
                m, _ = batch_means(samples, bsize)
                return float(m)
            except Exception:
                return ""

        def bm_windows_means_or_blank(busy_periods, completion_times):
            util_samples, thr_samples = window_util_thr(
                busy_periods, completion_times, TIME_WINDOW, now
            )
            mu = bm_mean_or_blank(util_samples, TIME_WINDOWS_PER_BATCH)
            mt = bm_mean_or_blank(thr_samples, TIME_WINDOWS_PER_BATCH)
            return mu, mt

        # --- Web (BM: RT, Util e Thr su finestre)
        web_rt_mean_bm = bm_mean_or_blank(self.web_server.completed_jobs, BATCH_SIZE)
        web_util_bm, web_thr_bm = bm_windows_means_or_blank(
            self.web_server.busy_periods, self.web_server.completion_times
        )

        # --- Mitigation (BM)
        center = self.mitigation_manager.center
        mit_rt_mean_bm = bm_mean_or_blank(center.completed_jobs, BATCH_SIZE)
        mit_util_bm, mit_thr_bm = bm_windows_means_or_blank(
            center.busy_periods, center.completion_times
        )

        # --- Drop rates (proporzioni sull'intera run)
        drop_fp_rate = self.metrics.get("false_positives", 0) / total
        drop_full_rate = self.metrics.get("discarded_mitigation", 0) / total

        # Base row
        row = {
            "scenario": scenario,
            "ARRIVAL_P": float(arrival_p),
            "ARRIVAL_L1": float(arrival_l1),
            "ARRIVAL_L2": float(arrival_l2),
            "total_time": float(now),
            "total_arrivals": int(self.metrics["total_arrivals"]),

            "web_util_bm": web_util_bm,
            "web_rt_mean_bm": web_rt_mean_bm,
            "web_throughput_bm": web_thr_bm,

            "mit_util_bm": mit_util_bm,
            "mit_rt_mean_bm": mit_rt_mean_bm,
            "mit_throughput_bm": mit_thr_bm,

            "drop_fp_rate": float(drop_fp_rate),
            "drop_full_rate": float(drop_full_rate),

            "spikes_count": int(len(self.spike_servers)),
        }

        # --- per-spike i (solo BM)
        for i in range(MAX_SPIKE_NUMBER):
            if i < len(self.spike_servers):
                s = self.spike_servers[i]
                s_rt_mean_bm = bm_mean_or_blank(s.completed_jobs, BATCH_SIZE)
                s_util_bm, s_thr_bm = bm_windows_means_or_blank(
                    s.busy_periods, s.completion_times
                )
                row[f"spike{i}_util_bm"]       = s_util_bm
                row[f"spike{i}_rt_mean_bm"]    = s_rt_mean_bm
                row[f"spike{i}_throughput_bm"] = s_thr_bm
                row[f"spike{i}_completions"]   = int(s.total_completions)
            else:
                row[f"spike{i}_util_bm"]       = ""
                row[f"spike{i}_rt_mean_bm"]    = ""
                row[f"spike{i}_throughput_bm"] = ""
                row[f"spike{i}_completions"]   = ""
        return row

    def report(self):
        now = self.env.now

        # flush finali
        self.web_server.update(now)
        for s in self.spike_servers:
            s.update(now)
        center = self.mitigation_manager.center
        center.update(now)

        # chiudi eventuali periodi busy aperti (end=None) per coerenza delle finestre
        def close_open_busy_periods(periods, now_):
            if periods and periods[-1][1] is None:
                periods[-1][1] = now_

        close_open_busy_periods(self.web_server.busy_periods, now)
        for s in self.spike_servers:
            close_open_busy_periods(s.busy_periods, now)
        close_open_busy_periods(center.busy_periods, now)

        print("\n==== SIMULATION COMPLETE ====")
        print(f"Total time: {now:.4f}")
        print(f"Arrivals: {self.metrics['total_arrivals']}")
        print(f"Lecite: {self.metrics['legal_arrivals']}, Illecite: {self.metrics['illegal_arrivals']}")
        print(f"Mitigation completions: {self.metrics['mitigation_completions']}")
        print(f"False positives (dropped): {self.metrics['false_positives']}")
        print(f"  Di cui lecite: {self.metrics['false_positives_legal']}")

        discarded = self.metrics.get("discarded_detail", [])
        print(f"Total discarded jobs: {len(discarded)}")

        def stats(name, server):
            if getattr(server, "completed_jobs", None) and server.completed_jobs:
                avg_rt = np.mean(server.completed_jobs)
                print(f"{name} Completions: {server.total_completions}")
                print(f"  Lecite  : {server.legal_completions}")
                print(f"  Illecite: {server.illegal_completions}")
                print(f"{name} Avg Resp Time: {avg_rt:.6f}")
                print(f"{name} Utilization: {server.busy_time / now:.6f}")
                print(f"{name} Throughput: {server.total_completions / now:.6f}")
            else:
                print(f"{name} Completions: 0")

        def print_ci(label, data, batch_size):
            try:
                if len(data) < batch_size or len(data) // batch_size < 2:
                    raise ValueError(f"Solo {len(data)} campioni → insufficienti per almeno 2 batch.")
                mean, ci = batch_means(data, batch_size)
                print(f"{label}: {mean:.6f} ± {ci:.6f} (95% CI)")
            except Exception as e:
                print(f"{label}: errore - {e}")

        # --- per-server ---
        stats("Web", self.web_server)
        for i, server in enumerate(self.spike_servers):
            stats(f"Spike-{i}", server)

        # --- Mitigation Center ---
        stats("Mitigation", center)

        print(f"Mitigation Discarded : {self.metrics.get('discarded_mitigation', 0)}")

        # === GLOBAL SUMMARY ===
        print("\n==== GLOBAL STATS ====")
        total = self.metrics["total_arrivals"]
        processed_legal = self.metrics.get("processed_legal", 0)
        processed_illegal = self.metrics.get("processed_illegal", 0)
        false_positive = self.metrics.get("false_positives", 0)
        false_positive_legal = self.metrics.get("false_positives_legal", 0)

        def percent(x): return 100.0 * x / total if total > 0 else 0.0

        print(f"Processed legal : {processed_legal} ({percent(processed_legal):.2f}%)")
        print(f"Processed illegal: {processed_illegal} ({percent(processed_illegal):.2f}%)")
        print(f"False positives (dropped by classification): {false_positive}, ({percent(false_positive):.2f}%)")
        print(f"False positives legal (dropped by classification): {false_positive_legal}, ({percent(false_positive_legal):.2f}%)")
        print(f"Mitigation Discarded (queue full): {self.metrics.get('discarded_mitigation', 0)}")

        # === CI  ===
        if self.enable_ci:
            print("\n======== INTERVALLI DI CONFIDENZA ========")

            # --- Web ---
            print("\n-- Web Server --")
            print_ci("Response Time:                           ", self.web_server.completed_jobs, BATCH_SIZE)
            util_samples, thr_samples = window_util_thr(self.web_server.busy_periods,
                                                        self.web_server.completion_times,
                                                        TIME_WINDOW, now)
            print_ci(f"Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
            print_ci(f"Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

            # --- Spike ---
            print("\n-- Spike Server --")
            for i, server in enumerate(self.spike_servers):
                print(f"Spike-{i}:")
                print_ci("Response Time:                           ", server.completed_jobs, BATCH_SIZE)
                util_samples, thr_samples = window_util_thr(server.busy_periods,
                                                            server.completion_times,
                                                            TIME_WINDOW, now)
                print_ci(f"Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
                print_ci(f"Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

            # --- Mitigation ---
            print("\n-- Mitigation Center --")
            print_ci("Response Time:                           ", center.completed_jobs, BATCH_SIZE)
            util_samples, thr_samples = window_util_thr(center.busy_periods,
                                                        center.completion_times,
                                                        TIME_WINDOW, now)
            print_ci(f"Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
            print_ci(f"Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

        print()
        print("==== END OF REPORT ====")


def run_simulation(scenario:str, mode: str, batch_means: bool, arrival_p=None, arrival_l1=None, arrival_l2=None):
    """
    Esegue una singola simulazione in 'mode' ('verification'|'standard')
    e stampa i CI con il metodo Batch Means (idfStudent).
    """
    if mode not in ("verification", "standard"):
        raise ValueError("mode must be 'verification' or 'standard'")
    if arrival_p == None:
        arrival_p = ARRIVAL_P
        arrival_l1 = ARRIVAL_L1
        arrival_l2 = ARRIVAL_L2

    env = simpy.Environment()
    system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, enable_ci=batch_means)
    env.run()
    system.report()

    row = system.make_csv_row(scenario, arrival_p, arrival_l1, arrival_l2)
    append_row("results_standard.csv", row)

def transitory_fieldnames(max_spikes: int):
    base = [
        "replica", "time",
        # Web
        "web_rt_mean", "web_util", "web_throughput",
        # Mitigation
        "mit_rt_mean", "mit_util", "mit_throughput",
        # Contatori globali
        "arrivals_so_far", "false_positives_so_far", "mitigation_completions_so_far",
        # Meta
        "spikes_count", "scenario", "mode", "is_final",
    ]
    for i in range(max_spikes):
        base += [f"spike{i}_rt_mean", f"spike{i}_util", f"spike{i}_throughput"]
    return base


def run_transitory_sim(
        mode: str,
        batch_means: bool,
        scenario: str = "transitory",
        out_csv: str = "plot/results_transitory.csv",
    ):
    # rimuovi il file per evitare header “vecchi”
    if os.path.exists(out_csv):
        os.remove(out_csv)

    arrival_p  = ARRIVAL_P
    arrival_l1 = ARRIVAL_L1_x40
    arrival_l2 = ARRIVAL_L2_x40

    # fieldnames stabili per tutto il run
    fieldnames = transitory_fieldnames(MAX_SPIKE_NUMBER)

    all_logs = []
    for rep in range(REPLICATION_FACTOR):
        seed = SEEDS[rep % len(SEEDS)]
        plantSeeds(seed)

        env = simpy.Environment()
        system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2,
                            enable_ci=batch_means)

        def checkpointer(env, system, rep_id):
            next_cp = CHECKPOINT_TIME
            while next_cp <= STOP_CONDITION:
                yield env.timeout(next_cp - env.now)
                snap = system.snapshot(env.now, replica_id=rep_id)
                # metadati sempre presenti
                snap["scenario"] = scenario
                snap["mode"] = mode
                snap["is_final"] = False

                all_logs.append(snap)
                append_row_stable(out_csv, snap, fieldnames)  # <--- usa lo scrittore stabile
                next_cp += CHECKPOINT_TIME

            # Ultimo snapshot esatto al termine
            if env.now < STOP_CONDITION:
                yield env.timeout(STOP_CONDITION - env.now)
                snap = system.snapshot(env.now, replica_id=rep_id)
                snap["scenario"] = scenario
                snap["mode"] = mode
                snap["is_final"] = True

                all_logs.append(snap)
                append_row_stable(out_csv, snap, fieldnames)

        env.process(checkpointer(env, system, rep))
        env.run(until=STOP_CONDITION)

    return all_logs
