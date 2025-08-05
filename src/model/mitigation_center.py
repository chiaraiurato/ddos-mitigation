import simpy

class MitigationCenter:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.queue = simpy.Store(env)
        self.busy = False
        self.completed_jobs = []
        self.total_completions = 0
        self.legal_completions = 0
        self.illegal_completions = 0
        self.busy_time = 0.0
        self.last_start = None
        env.process(self.server_process())

    def arrival(self, job):
        return self.queue.put(job)

    def server_process(self):
        while True:
            job = yield self.queue.get()
            self.busy = True
            self.last_start = self.env.now

            # Simula il tempo di elaborazione della mitigazione (fisso o esponenziale)
            from library.rvgs import Exponential
            from engineering.costants import MITIGATION_MEAN
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
