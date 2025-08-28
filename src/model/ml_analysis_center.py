import simpy
from library.rvgs import Exponential
from library.rngs import selectStream
from engineering.costants import (
    ANALYSIS_CORES, ANALYSIS_ES_CORE, RNG_STREAM_ANALYSIS_SERVICE
)

class AnalysisCenter:
    def __init__(self, env, name, metrics, on_complete=None):
        self.env = env
        self.name = name
        self.metrics = metrics
        self.on_complete = on_complete

        self.cores = ANALYSIS_CORES
        self.busy = 0

        self.core_busy_periods = [[] for _ in range(self.cores)]
        self.completed_jobs = []
        self.completion_times = []
        self.total_completions = 0
        self.legal_completions = 0
        self.illegal_completions = 0

    def has_capacity(self):
        return self.busy < self.cores

    def arrival(self, job):
        if not self.has_capacity():
            return False

        core_id = self.busy
        self.busy += 1
        start = self.env.now
        self.core_busy_periods[core_id].append([start, None])

        selectStream(RNG_STREAM_ANALYSIS_SERVICE)
        svc = Exponential(ANALYSIS_ES_CORE)

        def _serve():
            yield self.env.timeout(svc)
            now = self.env.now

            periods = self.core_busy_periods[core_id]
            if periods and periods[-1][1] is None:
                periods[-1][1] = now

            self.busy -= 1

            self.completed_jobs.append(now - job.arrival)  # RT locale al centro
            self.completion_times.append(now)
            self.total_completions += 1
            if job.is_legal:
                self.legal_completions += 1
            else:
                self.illegal_completions += 1

            if self.on_complete:
                self.on_complete(job, now)

        self.env.process(_serve())
        return True

    def update(self, now):
        for core_id in range(self.cores):
            periods = self.core_busy_periods[core_id]
            if periods and periods[-1][1] is None:
                periods[-1][1] = now
                periods.append([now, None])
