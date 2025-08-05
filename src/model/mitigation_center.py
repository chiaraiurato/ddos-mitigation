import simpy
from library.rvgs import Exponential
from engineering.costants import MITIGATION_MEAN

class MitigationCenter:
    def __init__(self, env, name, capacity):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.queue = simpy.Store(env, capacity=capacity)
        self.busy = False
        self.completed_jobs = []
        self.total_completions = 0
        self.legal_completions = 0
        self.illegal_completions = 0
        self.busy_time = 0.0
        self.last_start = None

        self.env.process(self.server_process())

    def has_capacity(self):
        return len(self.queue.items) < self.capacity

    def arrival(self, job):
        self.queue.put(job)

    def server_process(self):
        while True:
            job = yield self.queue.get() 
            self.busy = True
            self.last_start = self.env.now

            yield self.env.timeout(Exponential(MITIGATION_MEAN))

            now = self.env.now
            self.busy_time += now - self.last_start
            self.completed_jobs.append(now - job.arrival)
            self.total_completions += 1
            if job.is_legal:
                self.legal_completions += 1
            else:
                self.illegal_completions += 1

            self.busy = False
