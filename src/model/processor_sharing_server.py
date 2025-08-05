import simpy

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