import simpy
import numpy as np
from library.rvgs import Hyperexponential, Exponential
from library.rngs import random

from engineering.costants import *
from model.job import Job
from model.processor_sharing_server import ProcessorSharingServer

class DDoSSystem:
    def __init__(self, env):
        self.env = env
        self.web_server = ProcessorSharingServer(env, "Web")
        self.spike_servers = [ProcessorSharingServer(env, "Spike-0")]
        self.mitigation_completions = 0
        self.total_arrivals = 0
        self.false_positives = 0
        self.legal_arrivals = 0
        self.illegal_arrivals = 0
        self.false_positives_legal = 0
        self.env.process(self.arrival_process())

    def arrival_process(self):
        while self.total_arrivals < N_ARRIVALS:
            interarrival_time = Hyperexponential(ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2)
            yield self.env.timeout(interarrival_time)

            self.total_arrivals += 1
            arrival_time = self.env.now
            is_legal = random() < P_LECITO
            if is_legal:
                self.legal_arrivals += 1
            else:
                self.illegal_arrivals += 1
            job = Job(self.total_arrivals, arrival_time, None, is_legal)
            self.env.process(self.mitigation_process(job))

    def mitigation_process(self, job):
        try:
            yield self.env.timeout(Exponential(MITIGATION_MEAN))
        except simpy.Interrupt:
            return

        now = self.env.now
        self.mitigation_completions += 1

        if random() < P_FALSE_POSITIVE:
            self.false_positives += 1
            if job.is_legal:
                self.false_positives_legal += 1
            return

        if random() < P_FEEDBACK:
            job.arrival = now
            self.env.process(self.mitigation_process(job))
        else:
            service_time = Hyperexponential(SERVICE_P, SERVICE_L1, SERVICE_L2)
            job.remaining = service_time
            job.original_service = service_time
            job.last_updated = now

            if len(self.web_server.jobs) < MAX_WEB_CAPACITY:
                self.web_server.arrival(job)
            else:
                # Find spike server with capacity
                assigned = False
                for server in self.spike_servers:
                    if len(server.jobs) < MAX_SPIKE_CAPACITY:
                        server.arrival(job)
                        assigned = True
                        break
                if not assigned:
                    new_id = len(self.spike_servers)
                    new_server = ProcessorSharingServer(self.env, f"Spike-{new_id}")
                    self.spike_servers.append(new_server)
                    new_server.arrival(job)

    def report(self):
        now = self.env.now
        print("\n==== SIMULATION COMPLETE ====")
        print(f"Total time: {now:.4f}")
        print(f"Arrivals: {self.total_arrivals}")
        print(f"Lecite: {self.legal_arrivals}, Illecite: {self.illegal_arrivals}")
        print(f"Mitigation completions: {self.mitigation_completions}")
        print(f"False positives (dropped): {self.false_positives}")
        print(f"  Di cui lecite: {self.false_positives_legal}")

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

def run_sim():
    env = simpy.Environment()
    system = DDoSSystem(env)
    env.run()
    system.report()

if __name__ == "__main__":
    run_sim()
