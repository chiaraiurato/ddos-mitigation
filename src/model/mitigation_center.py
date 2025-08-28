import simpy
from library.rvgs import Exponential
from engineering.costants import MITIGATION_MEAN
from library.rngs import selectStream
from engineering.costants import RNG_STREAM_MITIGATION_SERVICE


class MitigationCenter:
    def __init__(self, env, name, capacity, metrics=None):
        self.env = env
        self.name = name
        self.capacity = capacity

        self.queue = simpy.Store(env, capacity=capacity)

        self.busy = False
        self.last_start = None

        self.busy_time = 0.0
        self.busy_periods = []         

        self.completed_jobs = []       
        self.completion_times = []     

        self.total_completions = 0
        self.legal_completions = 0
        self.illegal_completions = 0

        self.metrics = metrics

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

            selectStream(RNG_STREAM_MITIGATION_SERVICE)
            yield self.env.timeout(Exponential(MITIGATION_MEAN))

            now = self.env.now
            self.busy_time += now - self.last_start
            self.busy_periods.append((self.last_start, now))

            self.completed_jobs.append(now - job.arrival)
            self.completion_times.append(now)

            self.total_completions += 1
            if job.is_legal:
                self.legal_completions += 1
            else:
                self.illegal_completions += 1

            if self.metrics is not None:
                self.metrics["mitigation_completions"] = self.metrics.get("mitigation_completions", 0) + 1

            self.busy = False
            self.last_start = None

    def update(self, now):
        if self.busy and self.last_start is not None:
            self.busy_time += now - self.last_start
            self.busy_periods.append((self.last_start, now))
            self.last_start = now 