import simpy
import numpy as np
from library.rngs import random
from scipy.stats import t
from engineering.costants import MAX_SPIKE_NUMBER
from engineering.costants import *
from engineering.distributions import get_interarrival_time
from engineering.statistics import batch_means, window_util_thr
from utils.csv_writer import append_row
from model.job import Job
from model.processor_sharing_server import ProcessorSharingServer
from model.mitigation_manager import MitigationManager


class DDoSSystem:
    def __init__(self, env, mode, enable_ci=False):
        self.env = env
        self.mode = mode
        self.enable_ci = enable_ci
        self.web_server = ProcessorSharingServer(env, "Web")
        self.spike_servers = [ProcessorSharingServer(env, "Spike-0")]

        # if self.mode == "verification":
        #     self.batch_count = 0
        #     self.completed_in_batch = 0
        #     self.sum_rt_in_batch = 0.0
        #     self.batch_means = []

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

    def arrival_process(self, mode):
        
        if mode == "verification":
            arrivals = N_ARRIVALS_VERIFICATION
            interarrival_mode = "verification"
        else:  # "standard"
            arrivals = N_ARRIVALS 
            interarrival_mode = "standard"

        while self.metrics["total_arrivals"] < arrivals:
            interarrival_time = get_interarrival_time(interarrival_mode)
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

    @staticmethod
    def direct_ci(data, confidence=0.95):
        n = len(data)
        if n < 2:
            return (np.mean(data) if n == 1 else float('nan')), float('nan')
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        ci = t.ppf((1 + confidence) / 2., n - 1) * std_err
        return mean, ci
    
    def make_csv_row(self, scenario: str, arrival_p: float, arrival_l1: float, arrival_l2: float):
        """
        Costruisce la riga CSV con i dati della simulazione corrente.
        """
        now = self.env.now
        total = max(1, self.metrics["total_arrivals"])  

        # --- Web ---
        web_util = self.web_server.busy_time / now if now > 0 else 0.0
        web_rt_mean = float(np.mean(self.web_server.completed_jobs)) if self.web_server.completed_jobs else ""
        web_throughput = self.web_server.total_completions / now if now > 0 else 0.0

        # --- Mitigation ---
        center = self.mitigation_manager.center
        mit_util = center.busy_time / now if now > 0 else 0.0
        mit_rt_mean = float(np.mean(center.completed_jobs)) if center.completed_jobs else ""
        mit_throughput = center.total_completions / now if now > 0 else 0.0

        # --- Drop rates ---
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
            "web_util": float(web_util),
            "web_rt_mean": web_rt_mean,
            "web_throughput": float(web_throughput),
            "mit_util": float(mit_util),
            "mit_rt_mean": mit_rt_mean,
            "mit_throughput": float(mit_throughput),
            "drop_fp_rate": float(drop_fp_rate),
            "drop_full_rate": float(drop_full_rate),
            "spikes_count": int(len(self.spike_servers)),
        }

        # --- per-spike (fino a MAX_SPIKE_NUMBER) ---
        for i in range(MAX_SPIKE_NUMBER):
            if i < len(self.spike_servers):
                s = self.spike_servers[i]
                util_i = s.busy_time / now if now > 0 else 0.0
                rt_i = float(np.mean(s.completed_jobs)) if s.completed_jobs else ""
                thr_i = s.total_completions / now if now > 0 else 0.0
                comp_i = int(s.total_completions)
                row[f"spike{i}_util"] = float(util_i)
                row[f"spike{i}_rt_mean"] = rt_i
                row[f"spike{i}_throughput"] = float(thr_i)
                row[f"spike{i}_completions"] = comp_i
            else:
                row[f"spike{i}_util"] = ""
                row[f"spike{i}_rt_mean"] = ""
                row[f"spike{i}_throughput"] = ""
                row[f"spike{i}_completions"] = ""
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


# def run_verification():
#     env = simpy.Environment()
#     system = DDoSSystem(env, "verification")
#     env.run()
#     system.report()


# def run_standard():
#     env = simpy.Environment()
#     system = DDoSSystem(env, "standard")
#     env.run()
#     system.report()

def run_simulation(mode: str, batch_means: bool):
    """
    Esegue una singola simulazione in 'mode' ('verification'|'standard')
    e stampa i CI con il metodo Batch Means (idfStudent).
    """
    if mode not in ("verification", "standard"):
        raise ValueError("mode must be 'verification' or 'standard'")
    env = simpy.Environment()
    system = DDoSSystem(env, mode, enable_ci=batch_means)
    env.run()
    system.report()
    row = system.make_csv_row("x1", ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2)
    append_row("results_standard.csv", row)
