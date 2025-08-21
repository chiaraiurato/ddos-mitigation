import simpy
from library.rvgs import Exponential
from library.rngs import selectStream
from engineering.costants import (
    ANALYSIS_CORES, ANALYSIS_ES_CORE, RNG_STREAM_ANALYSIS_SERVICE
)

class AnalysisCenter:
    """
    Centro di Analisi ML: M/M/c/c (NO coda)
    - c = ANALYSIS_CORES
    - servizio per-core ~ Exp(ANALYSIS_ES_CORE)
    - on_complete(job, now) viene richiamato al termine
    """
    def __init__(self, env, name, metrics, on_complete=None):
        self.env = env
        self.name = name
        self.metrics = metrics
        self.on_complete = on_complete

        self.cores = ANALYSIS_CORES
        self.busy = 0

        # statistiche
        self.core_busy_periods = [[] for _ in range(self.cores)]
        self.completed_jobs = []
        self.completion_times = []
        self.total_completions = 0
        self.legal_completions = 0
        self.illegal_completions = 0

    def has_capacity(self):
        return self.busy < self.cores

    def arrival(self, job):
        """Ritorna False se pieno (drop per capacità)."""
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

            # chiudi periodo busy del core
            periods = self.core_busy_periods[core_id]
            if periods and periods[-1][1] is None:
                periods[-1][1] = now

            self.busy -= 1

            # metriche
            self.completed_jobs.append(now - job.arrival)  # RT locale al centro
            self.completion_times.append(now)
            self.total_completions += 1
            if getattr(job, "is_legal", False):
                self.legal_completions += 1
            else:
                self.illegal_completions += 1

            # callback al manager (classificazione ML + routing)
            if self.on_complete:
                self.on_complete(job, now)

        self.env.process(_serve())
        return True

    def update(self, now):
        """Chiude periodi busy 'aperti' fino a now (per util/throughput a finestra)."""
        for core_id in range(self.cores):
            periods = self.core_busy_periods[core_id]
            if periods and periods[-1][1] is None:
                periods[-1][1] = now
                # riapri per continuità, se serve
                periods.append([now, None])
