import simpy
import numpy as np
from library.rngs import random

from math import sqrt
from scipy.stats import t

from engineering.costants import *
from engineering.distributions import get_interarrival_time
from engineering.statistics import batch_means
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

        self.env.process(self.arrival_process())

    def arrival_process(self):
        while self.metrics["total_arrivals"] < N_ARRIVALS:
            
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
        if self.mode != "verification":
            return
        if server_name != "Web":
            return

        self.completed_in_batch += 1
        self.sum_rt_in_batch += response_time

        if self.completed_in_batch == BATCH_SIZE:
            batch_mean = self.sum_rt_in_batch / BATCH_SIZE
            self.batch_means.append(batch_mean)
            self.completed_in_batch = 0
            self.sum_rt_in_batch = 0
            self.batch_count += 1

    def report(self):
        now = self.env.now
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
            if server.completed_jobs:
                avg_rt = np.mean(server.completed_jobs)
                print(f"{name} Completions: {server.total_completions}")
                print(f"  Lecite  : {server.legal_completions}")
                print(f"  Illecite: {server.illegal_completions}")
                print(f"{name} Avg Resp Time: {avg_rt:.6f}")
                print(f"{name} Utilization: {server.busy_time / now:.6f}")
                print(f"{name} Throughput: {server.total_completions / now:.6f}")
            else:
                print(f"{name} Completions: 0")

        stats("Web", self.web_server)
        for i, server in enumerate(self.spike_servers):
            stats(f"Spike-{i}", server)
        print(f"Mitigation Discarded : {self.metrics.get('discarded_mitigation', 0)}")

        # === Global percentages summary ===
        print("\n==== GLOBAL STATS ====")
        total = self.metrics["total_arrivals"]
        discarded_total = len(discarded)
        processed_legal = self.metrics.get("processed_legal", 0)
        processed_illegal = self.metrics.get("processed_illegal", 0)

        false_positive = self.metrics.get("false_positives", 0)
        false_positive_legal = self.metrics.get("false_positives_legal", 0)

        discarded_legal = sum(1 for d in discarded if d["is_legal"])
        discarded_illegal = discarded_total - discarded_legal

        def percent(x): return 100.0 * x / total if total > 0 else 0.0

        print(f"Processed legal  : {processed_legal} ({percent(processed_legal):.2f}%)")
        print(f"Processed illegal: {processed_illegal} ({percent(processed_illegal):.2f}%)")
        print(f"False positives (dropped by classification): {false_positive}, ({percent(false_positive):.2f}%)")
        print(f"False positives legal (dropped by classification): {false_positive_legal}, ({percent(false_positive_legal):.2f}%)")
        print(f"Mitigation Discarded (queue full): {self.metrics.get('discarded_mitigation', 0)}")

        # === Confidence Intervals via Batch Means (only in verification mode) ===
        if self.mode == "verification":
            print("\n==== INTERVALLI DI CONFIDENZA (Batch Means) ====")

            def print_ci(label, data, batch_size):
                try:
                    mean, ci = batch_means(data, batch_size)
                    print(f"{label}: {mean:.6f} ± {ci:.6f} (95% CI)")
                except Exception as e:
                    print(f"{label}: errore - {e}")

            print("\n-- Web Server --")
            print_ci("Response Time", self.web_server.completed_jobs, BATCH_SIZE)
            util_samples, thr_samples = self.web_server.get_batch_samples(BATCH_SIZE, now)
            print_ci("Utilization", util_samples, BATCH_SIZE)
            print_ci("Throughput", thr_samples, BATCH_SIZE)

            print("\n-- Spike Server --")
            for i, server in enumerate(self.spike_servers):
                print(f"Spike-{i}:")
                print_ci("Response Time", server.completed_jobs, BATCH_SIZE)
                util_samples, thr_samples = server.get_batch_samples(BATCH_SIZE, now)
                print_ci("Utilization", util_samples, BATCH_SIZE)
                print_ci("Throughput", thr_samples, BATCH_SIZE)


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
