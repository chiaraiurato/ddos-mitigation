import simpy
import numpy as np
from library.rngs import random
from scipy.stats import t

from engineering.costants import *
from engineering.distributions import get_interarrival_time
from engineering.statistics import batch_means, window_util_thr
from model.job import Job
from model.processor_sharing_server import ProcessorSharingServer
from model.mitigation_manager import MitigationManager

class DDoSSystem:
    def __init__(self, env, mode):
        self.env = env
        self.mode = mode
        self.web_server = ProcessorSharingServer(env, "Web")
        self.spike_servers = [ProcessorSharingServer(env, "Spike-0")]

        if self.mode == "verification":
            self.batch_count = 0
            self.completed_in_batch = 0
            self.sum_rt_in_batch = 0.0
            self.batch_means = []

        # Metriche globali
        self.metrics = {
            "mitigation_completions": 0,   # aggiornate nel MitigationCenter al completamento reale
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
        
        if(mode == "verification"): 
            arrivals = N_ARRIVALS_VERIFICATION
        else:
            arrivals = N_ARRIVALS

        
        while self.metrics["total_arrivals"] < arrivals:
            interarrival_time = get_interarrival_time(self.mode)
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
        # Batch Means per la sola 'verification' e solo per Web
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
        print()
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

        # === CI (modalità verification) ===
        if self.mode == "verification":
            print("\n==== INTERVALLI DI CONFIDENZA (Batch Means) ====")

            # --- Web ---
            print("\n-- Web Server --")
            print_ci("Response Time (batch completamenti):     ", self.web_server.completed_jobs, BATCH_SIZE)
            util_samples, thr_samples = window_util_thr(self.web_server.busy_periods,
                                                        self.web_server.completion_times,
                                                        TIME_WINDOW, now)
            print_ci(f"Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
            print_ci(f"Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

            # --- Spike ---
            print("\n-- Spike Server --")
            for i, server in enumerate(self.spike_servers):
                print(f"Spike-{i}:")
                print_ci("Response Time (batch completamenti): ", server.completed_jobs, BATCH_SIZE)
                util_samples, thr_samples = window_util_thr(server.busy_periods,
                                                            server.completion_times,
                                                            TIME_WINDOW, now)
                print_ci(f"Utilization:                         ", util_samples, TIME_WINDOWS_PER_BATCH)
                print_ci(f"Throughput:                          ", thr_samples, TIME_WINDOWS_PER_BATCH)

            # --- Mitigation ---
            print("\n-- Mitigation Center --")
            print_ci("Response Time (batch completamenti)      ", center.completed_jobs, BATCH_SIZE)
            util_samples, thr_samples = window_util_thr(center.busy_periods,
                                                        center.completion_times,
                                                        TIME_WINDOW, now)
            print_ci(f"Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
            print_ci(f"Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

        print()
        print("==== END OF REPORT ====")


def choose_mode():
    print("Scegli la modalità:")
    print("1. Verifica (distribuzioni esponenziali)")
    print("2. Simulazione standard (distribuzioni iperesponenziali)")
    choice = input("Inserisci 1 o 2: ").strip()
    if choice == "1":
        return "verification"
    elif choice == "2":
        return "standard"
    else:
        print("Scelta non valida. Default: standard.")
        return "standard"


def run_sim():
    mode = choose_mode()
    print(f"\nModalità selezionata: {mode.upper()}")
    env = simpy.Environment()
    system = DDoSSystem(env, mode)
    env.run()
    system.report()


if __name__ == "__main__":
    run_sim()
