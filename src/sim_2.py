import simpy
import numpy as np
from library.rvgs import Hyperexponential

# Parameters
ARRIVAL_P = 0.03033
ARRIVAL_L1 = 0.4044
ARRIVAL_L2 = 12.9289
SERVICE_P = 0.03033
SERVICE_L1 = 0.3791
SERVICE_L2 = 12.1208
MITIGATION_MEAN = 0.001
P_FEEDBACK = 0.02
P_FALSE_POSITIVE = 0.01  # 1% false positive rate
MAX_WEB_CAPACITY = 20
MAX_SPIKE_CAPACITY = 20
SCALE_THRESHOLD = 20
N_ARRIVALS = 360000
P_LECITO = 0.1  # 10% delle richieste sono lecite

class Job:
    def __init__(self, job_id, arrival_time, service_time, is_legal):
        self.id = job_id
        self.arrival = arrival_time
        self.remaining = service_time
        self.original_service = service_time
        self.last_updated = arrival_time
        self.is_legal = is_legal

class ProcessorSharingServer:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.jobs = []
        self.proc = None
        self.completed_jobs = []
        self.total_completions = 0
        self.legal_completions = 0
        self.illegal_completions = 0
        self.last_time = 0.0
        self.area = 0.0
        self.busy_time = 0.0

    def update(self, now):
        dt = now - self.last_time
        if dt <= 0:
            return

        n = len(self.jobs)
        self.area += n * dt
        if n > 0:
            self.busy_time += dt
            service_per_job = dt / n
            for job in self.jobs:
                job.remaining = max(job.remaining - service_per_job, 0.0)
                job.last_updated = now
        self.last_time = now

    def arrival(self, job):
        now = self.env.now
        self.update(now)
        self.jobs.append(job)
        self.schedule_completion()

    def schedule_completion(self):
        if not self.jobs:
            return
        next_job = min(self.jobs, key=lambda j: j.remaining)
        n = len(self.jobs)
        delay = next_job.remaining * n

        if self.proc and self.proc.is_alive and self.proc != self.env.active_process:
            self.proc.interrupt()

        if self.env.active_process != self.proc:
            self.proc = self.env.process(self.completion_event(next_job, delay))

    def completion_event(self, job, delay):
        try:
            yield self.env.timeout(delay)
        except simpy.Interrupt:
            return

        now = self.env.now
        self.update(now)

        if job in self.jobs:
            self.jobs.remove(job)
            response_time = now - job.arrival
            self.completed_jobs.append(response_time)
            self.total_completions += 1
            if job.is_legal:
                self.legal_completions += 1
            else:
                self.illegal_completions += 1
            if self.jobs:
                self.env.process(self.schedule_next_completion())

    def schedule_next_completion(self):
        yield self.env.timeout(0)
        self.schedule_completion()

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
            is_legal = np.random.rand() < P_LECITO
            if is_legal:
                self.legal_arrivals += 1
            else:
                self.illegal_arrivals += 1
            job = Job(self.total_arrivals, arrival_time, None, is_legal)
            self.env.process(self.mitigation_process(job))

    def mitigation_process(self, job):
        try:
            yield self.env.timeout(np.random.exponential(MITIGATION_MEAN))
        except simpy.Interrupt:
            return

        now = self.env.now
        self.mitigation_completions += 1

        if np.random.rand() < P_FALSE_POSITIVE:
            self.false_positives += 1
            if job.is_legal:
                self.false_positives_legal += 1
            return

        if np.random.rand() < P_FEEDBACK:
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
