import simpy
import numpy as np
#import random

from library.rngs import random

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
N_ARRIVALS = 68561

def hyperexp(p, l1, l2):
    return np.random.exponential(1/l1) if random() < p else np.random.exponential(1/l2)

class Job:
    def __init__(self, arrival_time, service_time):
        self.arrival = arrival_time
        self.remaining = service_time
        self.last_updated = arrival_time

class DualPoolPS:
    def __init__(self, env):
        self.env = env

        self.web_jobs = []
        self.spike_jobs = []

        self.web_proc = None
        self.spike_proc = None

        self.web_stats = []
        self.spike_stats = []
        self.web_completions = 0
        self.spike_completions = 0
        self.total_arrivals = 0

        self.area_web = 0.0
        self.area_spike = 0.0
        self.busy_web = 0.0
        self.busy_spike = 0.0

        self.last_time = 0.0

        #env.process(self.arrival_process_time()) # <- Time driven
        env.process(self.arrival_process_arrival_fixed()) # <- Arrival fixed

    def arrival_process_time(self):
        while True:
            yield self.env.timeout(hyperexp(ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2))
            now = self.env.now
            self.update_areas(now)
            self.update_pool(self.web_jobs, now)
            self.update_pool(self.spike_jobs, now)
            self.last_time = now

            self.total_arrivals += 1
            service_time = hyperexp(SERVICE_P, SERVICE_L1, SERVICE_L2)
            job = Job(now, service_time)

            # Decisione istantanea basata sulla lunghezza attuale
            if len(self.web_jobs) < SI_MAX:
                self.web_jobs.append(job)
                self.schedule_completion(self.web_jobs, 'web')
            else:
                self.spike_jobs.append(job)
                self.schedule_completion(self.spike_jobs, 'spike')

    def arrival_process_arrival_fixed(self):
        while self.total_arrivals < N_ARRIVALS:
            yield self.env.timeout(hyperexp(ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2))
            now = self.env.now
            self.update_areas(now)
            self.update_pool(self.web_jobs, now)
            self.update_pool(self.spike_jobs, now)
            self.last_time = now

            self.total_arrivals += 1
            service_time = hyperexp(SERVICE_P, SERVICE_L1, SERVICE_L2)
            job = Job(now, service_time)

            if len(self.web_jobs) < SI_MAX:
                self.web_jobs.append(job)
                self.schedule_completion(self.web_jobs, 'web')
            else:
                self.spike_jobs.append(job)
                self.schedule_completion(self.spike_jobs, 'spike')
    
    def update_pool(self, job_list, now):
        dt = now - self.last_time
        n = len(job_list)
        if n == 0:
            return
        for job in job_list:
            served = dt / n
            job.remaining = max(job.remaining - served, 0.0)
            job.last_updated = now

    def update_areas(self, now):
        dt = now - self.last_time
        self.area_web += len(self.web_jobs) * dt
        self.area_spike += len(self.spike_jobs) * dt
        if self.web_jobs:
            self.busy_web += dt
        if self.spike_jobs:
            self.busy_spike += dt

    def schedule_completion(self, job_list, pool_type):
        if not job_list:
            return
        n = len(job_list)
        next_job = min(job_list, key=lambda j: j.remaining)
        delay = max(next_job.remaining * n, 0.0)

        proc_name = f"{pool_type}_proc"
        prev_proc = getattr(self, proc_name)
        if prev_proc and prev_proc.is_alive and prev_proc != self.env.active_process:
            prev_proc.interrupt()

        proc = self.env.process(self.completion_event(job_list, next_job, pool_type, delay))
        setattr(self, proc_name, proc)

    def completion_event(self, job_list, job, pool_type, delay):
        try:
            yield self.env.timeout(delay)
        except simpy.Interrupt:
            return

        now = self.env.now
        self.update_areas(now)
        self.update_pool(self.web_jobs, now)
        self.update_pool(self.spike_jobs, now)
        self.last_time = now

        if job in job_list:
            job_list.remove(job)
            response_time = now - job.arrival
            if pool_type == 'web':
                self.web_stats.append(response_time)
                self.web_completions += 1
            else:
                self.spike_stats.append(response_time)
                self.spike_completions += 1

        self.schedule_completion(job_list, pool_type)

    def report(self):
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


def batch_means(num_batches=512, sim_time=SIM_TIME):
    try:
        from tqdm import trange
        iterator = trange(num_batches, desc="Running batches")
    except ImportError:
        iterator = range(num_batches)

    web_resp = []
    spike_resp = []
    web_tput = []
    spike_tput = []
    web_util = []
    spike_util = []

    for _ in iterator:
        env = simpy.Environment()
        sim = DualPoolPS(env)
        env.run(until=sim_time)

        # Raccogli metriche batch
        web_resp.append(np.mean(sim.web_stats) if sim.web_stats else 0.0)
        spike_resp.append(np.mean(sim.spike_stats) if sim.spike_stats else 0.0)
        web_tput.append(sim.web_completions / sim_time)
        spike_tput.append(sim.spike_completions / sim_time)
        web_util.append(sim.busy_web / sim_time)
        spike_util.append(sim.busy_spike / sim_time)

    def ci(data):
        if len(data) < 2:
            return 0.0, 0.0, 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        margin = CONF_LEVEL * std / np.sqrt(len(data))
        return mean, mean - margin, mean + margin

    # Calcolo CI per tutte le metriche
    w_mean, w_low, w_high = ci(web_resp)
    s_mean, s_low, s_high = ci(spike_resp)
    tput_w, tput_w_l, tput_w_h = ci(web_tput)
    tput_s, tput_s_l, tput_s_h = ci(spike_tput)
    util_w, util_w_l, util_w_h = ci(web_util)
    util_s, util_s_l, util_s_h = ci(spike_util)

    # Stampa risultati
    print(f"\n[Batch Means over {num_batches} batches of {sim_time} sec each]")
    print(f"Avg Web Resp Time     : {w_mean:.4f} ± {(w_mean - w_low):.4f} (99% CI [{w_low:.4f}, {w_high:.4f}])")
    print(f"Avg Spike Resp Time   : {s_mean:.4f} ± {(s_mean - s_low):.4f} (99% CI [{s_low:.4f}, {s_high:.4f}])")
    print(f"Avg Web Throughput    : {tput_w:.4f} ± {(tput_w - tput_w_l):.4f} (99% CI [{tput_w_l:.4f}, {tput_w_h:.4f}])")
    print(f"Avg Spike Throughput  : {tput_s:.4f} ± {(tput_s - tput_s_l):.4f} (99% CI [{tput_s_l:.4f}, {tput_s_h:.4f}])")
    print(f"Avg Web Utilization   : {util_w:.4f} ± {(util_w - util_w_l):.4f} (99% CI [{util_w_l:.4f}, {util_w_h:.4f}])")
    print(f"Avg Spike Utilization : {util_s:.4f} ± {(util_s - util_s_l):.4f} (99% CI [{util_s_l:.4f}, {util_s_h:.4f}])")



def run_sim():
    env = simpy.Environment()
    sim = DualPoolPS(env)
    # env.run(until=SIM_TIME) # <- time version
    env.run() # <- number of arrival fixed
    sim.report()

if __name__ == "__main__":
    run_sim()
    batch_means(num_batches=512)  