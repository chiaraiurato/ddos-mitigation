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

        # timeline cumulativa del tempo busy
        self._busy_cum_times = [0.0]
        self._busy_cum_values = [0.0]

        self.completed_jobs = []           # response times "locali" (nel centro)
        self.global_completed_jobs = []    # NEW: response times "globali" (incl. mitigazione)

        self.completion_times = []  # absolute times

        # NEW: periodi busy per window stats unificati
        self.busy_periods = []

    def set_observer(self, observer):
        self.observer = observer

    def _record_busy_point(self, now):
        self._busy_cum_times.append(now)
        self._busy_cum_values.append(self.busy_time)

    def _busy_cum_at(self, t):
        idx = bisect.bisect_right(self._busy_cum_times, t) - 1
        if idx < 0:
            return 0.0
        return self._busy_cum_values[idx]

    def update(self, now):
        dt = now - self.last_time
        if dt <= 0:
            return

        n = len(self.jobs)
        self.area += n * dt

        if n > 0:
            self.busy_time += dt

        self.last_time = now
        self._record_busy_point(now)

        if n > 0:
            service_per_job = dt / n
            for job in self.jobs:
                job.remaining = max(job.remaining - service_per_job, 0.0)
                job.last_updated = now

    def arrival(self, job):
        now = self.env.now
        self.update(now)

        # se si passa da 0 a 1 job, inizia un periodo busy
        if len(self.jobs) == 0:
            # apre intervallo busy
            self.busy_periods.append([now, None])

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
        self.update(now)

        if job in self.jobs:
            self.jobs.remove(job)
            response_time = now - job.arrival      # RT locale centro
            # NEW: RT globale = dal primo ingresso nel sistema
            global_rt = now - getattr(job, "sys_arrival", job.arrival)

            self.completed_jobs.append(response_time)
            self.global_completed_jobs.append(global_rt)   # NEW
            self.completion_times.append(now)
            self.total_completions += 1

            if hasattr(self, 'observer') and self.observer:
                self.observer.notify_completion(self.name, response_time)

            if job.is_legal:
                self.legal_completions += 1
            else:
                self.illegal_completions += 1

            # se ora non ci sono più job, chiudi il periodo busy corrente
            if len(self.jobs) == 0:
                # trova l'ultimo intervallo aperto e chiudilo
                if self.busy_periods and self.busy_periods[-1][1] is None:
                    self.busy_periods[-1][1] = now

            if self.jobs:
                self.env.process(self.schedule_next_completion())

    def schedule_next_completion(self):
        yield self.env.timeout(0)
        self.schedule_completion()

    # (facoltativo: puoi tenere questo metodo, ma non lo useremo più dal report)
    def get_batch_samples(self, time_window, simulation_time):
        utilization_samples = []
        throughput_samples = []

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

            in_batch = (completions >= start_time) & (completions < end_time)
            num_jobs = int(np.sum(in_batch))

            busy_in_batch = self._busy_cum_at(end_time) - self._busy_cum_at(start_time)

            utilization = busy_in_batch / win_len
            throughput = num_jobs / win_len

            utilization_samples.append(utilization)
            throughput_samples.append(throughput)

        return utilization_samples, throughput_samples