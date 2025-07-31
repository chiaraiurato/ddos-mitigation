import simpy
import random
import numpy as np

# Parameters
SI_MAX = 20
ARRIVAL_P = 0.03033
ARRIVAL_L1 = 0.4044
ARRIVAL_L2 = 12.9289
SERVICE_P = 0.03033
SERVICE_L1 = 0.3791
SERVICE_L2 = 12.1208
SIM_TIME = 10000
CONF_LEVEL = 2.576
EPSILON = 1e-5


def hyperexp(p, l1, l2):
    return np.random.exponential(1 / l1) if random.random() < p else np.random.exponential(1 / l2)


class Job:
    def __init__(self, arrival_time, service_time, is_web):
        self.arrival_time = arrival_time
        self.remaining = service_time
        self.is_web = is_web


class ProcessorSharingSim:
    def __init__(self, env):
        self.env = env
        self.jobs = []
        self.tokens = SI_MAX

        self.total_arrivals = 0
        self.web_completions = 0
        self.spike_completions = 0
        self.web_stats = []
        self.spike_stats = []
        self.area_web = 0.0
        self.area_spike = 0.0
        self.busy_web = 0.0
        self.busy_spike = 0.0
        self.last_time = 0.0

        self.next_arrival = hyperexp(ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2)
        self.next_completion = float('inf')

        self.env.process(self.simulate())

    def simulate(self):
        while True:
            now = self.env.now
            dt = min(self.next_arrival, self.next_completion) - now
            self.update_jobs(dt)
            yield self.env.timeout(dt)

            if abs(self.env.now - self.next_arrival) < EPSILON:
                self.handle_arrival()
                self.next_arrival = self.env.now + hyperexp(ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2)
            else:
                self.handle_completion()

            if self.env.now >= SIM_TIME:
                break

    def handle_arrival(self):
        self.total_arrivals += 1
        service_time = hyperexp(SERVICE_P, SERVICE_L1, SERVICE_L2)
        is_web = self.tokens > 0

        if is_web:
            self.tokens -= 1
        #     print(f"[ARRIVAL] t={self.env.now:.4f} | WEB   | service={service_time:.4f} | tokens={self.tokens}")
        # else:
        #     print(f"[ARRIVAL] t={self.env.now:.4f} | SPIKE | service={service_time:.4f}")

        n = len(self.jobs)
        if n > 0:
            for job in self.jobs:
                job.remaining *= (n + 1) / n

        job = Job(self.env.now, service_time, is_web)
        self.jobs.append(job)
        self.update_next_completion()

    def handle_completion(self):
        if not self.jobs:
            self.next_completion = float('inf')
            return

        min_job = min(self.jobs, key=lambda j: j.remaining)
        response_time = self.env.now - min_job.arrival_time

        if min_job.is_web:
            self.tokens = min(self.tokens + 1, SI_MAX)
            self.web_completions += 1
            self.web_stats.append(response_time)
            print(f"[COMPL]   t={self.env.now:.4f} | WEB   | resp_time={response_time:.4f} | tokens={self.tokens}")
        else:
            self.spike_completions += 1
            self.spike_stats.append(response_time)
            print(f"[COMPL]   t={self.env.now:.4f} | SPIKE | resp_time={response_time:.4f}")

        self.jobs.remove(min_job)
        n = len(self.jobs)
        if n > 0:
            for job in self.jobs:
                job.remaining *= (n - 1) / n

        self.update_next_completion()

    def update_jobs(self, dt):
        if len(self.jobs) == 0:
            self.last_time = self.env.now
            return

        n = len(self.jobs)
        delta = dt / n
        for job in self.jobs:
            job.remaining = max(0.0, job.remaining - delta)

        self.area_web += sum(1 for j in self.jobs if j.is_web) * dt
        self.area_spike += sum(1 for j in self.jobs if not j.is_web) * dt
        if any(j.is_web for j in self.jobs):
            self.busy_web += dt
        if any(not j.is_web for j in self.jobs):
            self.busy_spike += dt
        self.last_time = self.env.now

    def update_next_completion(self):
        if not self.jobs:
            self.next_completion = float('inf')
        else:
            min_remaining = min(job.remaining for job in self.jobs)
            self.next_completion = self.env.now + min_remaining * len(self.jobs)

    def report(self):
        print("\n[REPORT FINAL]")

        def ci(data):
            if len(data) < 2:
                return 0, 0, 0
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            margin = CONF_LEVEL * std / np.sqrt(len(data))
            return mean, mean - margin, mean + margin

        web_mean, web_low, web_high = ci(self.web_stats)
        spike_mean, spike_low, spike_high = ci(self.spike_stats)

        total_time = self.env.now
        avg_num_web = self.area_web / total_time
        avg_num_spike = self.area_spike / total_time
        util_web = self.busy_web / total_time
        util_spike = self.busy_spike / total_time
        tput_web = self.web_completions / total_time
        tput_spike = self.spike_completions / total_time
        sys_tput = (self.web_completions + self.spike_completions) / total_time

        print(f"Total Arrivals         : {self.total_arrivals}")
        print(f"Web Completions        : {self.web_completions}")
        print(f"Spike Completions      : {self.spike_completions}")
        print(f"Avg Web Response Time  : {web_mean:.4f} ± {(web_mean - web_low):.4f} (99% CI [{web_low:.4f}, {web_high:.4f}])")
        print(f"Avg Spike Response Time: {spike_mean:.4f} ± {(spike_mean - spike_low):.4f} (99% CI [{spike_low:.4f}, {spike_high:.4f}])")
        print(f"Avg Number in Web      : {avg_num_web:.4f}")
        print(f"Avg Number in Spike    : {avg_num_spike:.4f}")
        print(f"Web Utilization        : {util_web:.4f}")
        print(f"Spike Utilization      : {util_spike:.4f}")
        print(f"Web Throughput         : {tput_web:.4f}")
        print(f"Spike Throughput       : {tput_spike:.4f}")
        print(f"System Throughput      : {sys_tput:.4f}")


def run_sim():
    env = simpy.Environment()
    sim = ProcessorSharingSim(env)
    env.run(until=SIM_TIME)
    sim.report()


if __name__ == '__main__':
    run_sim()
