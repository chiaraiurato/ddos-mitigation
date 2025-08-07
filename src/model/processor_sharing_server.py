import simpy
import numpy as np

class ProcessorSharingServer:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.jobs = []
        self.proc = None
        self.completed_jobs = []  # response times
        self.total_completions = 0
        self.legal_completions = 0
        self.illegal_completions = 0
        self.last_time = 0.0
        self.area = 0.0
        self.busy_time = 0.0
        self.busy_periods = []  # list of (start, end) tuples
        self.completion_times = []  # absolute times
        self._busy_start = None

    def set_observer(self, observer):
        self.observer = observer

    def update(self, now):
        dt = now - self.last_time
        if dt <= 0:
            return

        n = len(self.jobs)
        self.area += n * dt

        if n > 0:
            self.busy_time += dt
            if self._busy_start is None:
                self._busy_start = self.last_time
            service_per_job = dt / n
            for job in self.jobs:
                job.remaining = max(job.remaining - service_per_job, 0.0)
                job.last_updated = now
        else:
            if self._busy_start is not None:
                self.busy_periods.append((self._busy_start, now))
                self._busy_start = None

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
            self.completion_times.append(now)
            self.total_completions += 1
            
            if hasattr(self, 'observer') and self.observer:
                self.observer.notify_completion(self.name, response_time)

            if job.is_legal:
                self.legal_completions += 1
            else:
                self.illegal_completions += 1
            if self.jobs:
                self.env.process(self.schedule_next_completion())

    def schedule_next_completion(self):
        yield self.env.timeout(0)
        self.schedule_completion()

    def get_batch_samples(self, batch_size, simulation_time):
        utilization_samples = []
        throughput_samples = []

        time_points = np.arange(0, simulation_time, batch_size)
        completions = np.array(self.completion_times)
        if len(completions) == 0:
            return utilization_samples, throughput_samples

        for start_time in time_points:
            end_time = start_time + batch_size
            in_batch = (completions >= start_time) & (completions < end_time)
            num_jobs = np.sum(in_batch)

            # Utilization: tempo occupato nel batch
            busy_in_batch = sum(min(end_time, b_end) - max(start_time, b_start)
                                for b_start, b_end in self.busy_periods
                                if b_end > start_time and b_start < end_time)
            utilization = busy_in_batch / batch_size if batch_size > 0 else 0.0

            utilization_samples.append(utilization)
            throughput_samples.append(num_jobs / batch_size if batch_size > 0 else 0.0)

        return utilization_samples, throughput_samples
