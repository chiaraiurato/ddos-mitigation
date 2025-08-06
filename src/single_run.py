import simpy
import numpy as np
from library.rvgs import Hyperexponential
from library.rngs import random

from engineering.costants import *
from model.job import Job
from model.processor_sharing_server import ProcessorSharingServer
from model.mitigation_manager import MitigationManager

class DDoSSystem:
    def __init__(self, env):
        self.env = env
        self.web_server = ProcessorSharingServer(env, "Web")
        self.spike_servers = [ProcessorSharingServer(env, "Spike-0")]

        
        self.metrics = {
            "mitigation_completions": 0,
            "total_arrivals": 0,
            "false_positives": 0,
            "legal_arrivals": 0,
            "illegal_arrivals": 0,
            "false_positives_legal": 0,
        }

        self.mitigation_manager = MitigationManager(
            env, self.web_server, self.spike_servers, self.metrics
        )

        self.env.process(self.arrival_process())

    def arrival_process(self):
        while self.metrics["total_arrivals"] < N_ARRIVALS:
            interarrival_time = Hyperexponential(ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2)
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
        discarded_total = len(self.metrics.get("discarded_detail", []))
        processed_legal = self.metrics.get("processed_legal", 0)
        processed_illegal = self.metrics.get("processed_illegal", 0)

        false_positive = self.metrics.get("false_positives", 0)
        false_positive_legal = self.metrics.get("false_positives_legal", 0)

        discarded_legal = sum(1 for d in self.metrics.get("discarded_detail", []) if d["is_legal"])
        discarded_illegal = discarded_total - discarded_legal

        def percent(x): return 100.0 * x / total if total > 0 else 0.0

        print(f"Processed legal  : {processed_legal} ({percent(processed_legal):.2f}%)")
        print(f"Processed illegal: {processed_illegal} ({percent(processed_illegal):.2f}%)")

        # Richieste droppate per la coda piena
        #print(f"Discarded legal  : {discarded_legal} ({percent(discarded_legal):.2f}%)")
        #print(f"Discarded illegal: {discarded_illegal} ({percent(discarded_illegal):.2f}%)")

        print(f"False positives (dropped by classification): {self.metrics['false_positives']}, ({percent(false_positive):.2f}%)")
        print(f"False positives legal (dropped by classification): {self.metrics['false_positives_legal']}, ({percent(false_positive_legal):.2f}%)")
        print(f"Mitigation Discarded (queue full): {self.metrics.get('discarded_mitigation', 0)}")

        print("==== END OF REPORT ====")

def run_sim():
    print("Starting DDoS Mitigation Simulation...")
    env = simpy.Environment()
    system = DDoSSystem(env)
    env.run()
    system.report()

if __name__ == "__main__":
    run_sim()
