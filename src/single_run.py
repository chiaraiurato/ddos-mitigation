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

        def stats(name, server):
            if server.completed_jobs:
                avg_rt = np.mean(server.completed_jobs)
                print(f"{name} Completions: {server.total_completions}")
                print(f"  Lecite  : {server.legal_completions}")
                print(f"  Illecite: {server.illegal_completions}")
                print(f"{name} Avg Resp Time: {avg_rt:.4f}")
                print(f"{name} Utilization: {server.busy_time / now:.4f}")
                print(f"{name} Throughput: {server.total_completions / now:.4f}")
            else:
                print(f"{name} Completions: 0")

        stats("Web", self.web_server)
        for i, server in enumerate(self.spike_servers):
            stats(f"Spike-{i}", server)
        print(f"Mitigation Discarded : {self.metrics.get('discarded_mitigation', 0)}")
        print("==== END OF REPORT ====")

def run_sim():
    print("Starting DDoS Mitigation Simulation...")
    env = simpy.Environment()
    system = DDoSSystem(env)
    env.run()
    system.report()

if __name__ == "__main__":
    run_sim()
