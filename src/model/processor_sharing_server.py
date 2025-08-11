import simpy
import numpy as np
import bisect

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

        # ✅ timeline cumulativa del tempo busy
        self._busy_cum_times = [0.0]
        self._busy_cum_values = [0.0]

        self.completion_times = []  # absolute times

    def set_observer(self, observer):
        self.observer = observer

    def _record_busy_point(self, now):
        # registra sempre il punto attuale (step function)
        self._busy_cum_times.append(now)
        self._busy_cum_values.append(self.busy_time)

    def _busy_cum_at(self, t):
        # step function: restituisce il valore cumulativo al tempo t
        # usa ricerca binaria sugli istanti registrati
        idx = bisect.bisect_right(self._busy_cum_times, t) - 1
        if idx < 0:
            return 0.0
        return self._busy_cum_values[idx]

    def update(self, now):
        dt = now - self.last_time
        if dt <= 0:
            return

        n = len(self.jobs)
        # area sotto N(t) per eventuali analisi (Little)
        self.area += n * dt

        # ✅ tempo busy: incrementa SOLO se n > 0
        if n > 0:
            self.busy_time += dt

        self.last_time = now
        # ✅ registra il punto cumulativo
        self._record_busy_point(now)

        # progresso PS: servizio equo
        if n > 0:
            service_per_job = dt / n
            for job in self.jobs:
                job.remaining = max(job.remaining - service_per_job, 0.0)
                job.last_updated = now

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
        delay = max(next_job.remaining * n, 0.0)

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
        self.update(now)  # flush e avanzamento PS

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

    def get_batch_samples(self, time_window, simulation_time):
        """
        Utilization per finestra = (busy_cum(end) - busy_cum(start)) / (end - start)
        Throughput per finestra = completamenti / (end - start)
        """
        utilization_samples = []
        throughput_samples = []

        # finestre (ultima potenzialmente parziale)
        t = 0.0
        windows = []
        while t < simulation_time - 1e-12:
            end = min(t + time_window, simulation_time)
            windows.append((t, end))
            t += time_window

        completions = np.array(self.completion_times)

        for start_time, end_time in windows:
            win_len = end_time - start_time
            if win_len <= 0:
                continue

            # completamenti nella finestra
            in_batch = (completions >= start_time) & (completions < end_time)
            num_jobs = int(np.sum(in_batch))

            # ✅ busy via cumulativo: niente busy_periods, niente doppi conteggi
            busy_in_batch = self._busy_cum_at(end_time) - self._busy_cum_at(start_time)

            utilization = busy_in_batch / win_len
            throughput = num_jobs / win_len

            utilization_samples.append(utilization)
            throughput_samples.append(throughput)

        return utilization_samples, throughput_samples
