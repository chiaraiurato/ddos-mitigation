import simpy
import random
import numpy as np

# Parameters
SI_MAX = 4
ARRIVAL_P = 0.03033
ARRIVAL_L1 = 0.4044
ARRIVAL_L2 = 12.9289
SERVICE_P = 0.03033
SERVICE_L1 = 0.3791
SERVICE_L2 = 12.1208
SIM_TIME = 5.0
STEP = 0.01
CONF_LEVEL = 2.576

# Helper function for hyperexponential distribution
def hyperexp(p, l1, l2):
    return np.random.exponential(1/l1) if random.random() < p else np.random.exponential(1/l2)

class ProcessorSharingSim:
    def __init__(self, env):
        self.env = env
        self.web_jobs = []
        self.spike_jobs = []
        self.tokens = SI_MAX
        self.web_stats = []
        self.spike_stats = []
        self.total_arrivals = 0
        self.web_completions = 0
        self.spike_completions = 0
        self.area_web = 0.0
        self.area_spike = 0.0
        self.busy_web = 0.0
        self.busy_spike = 0.0
        self.last_time = 0.0
        self.job_counter = 0
        self.event_log = []


    def arrival_process(self):
        while True:
            yield self.env.timeout(hyperexp(ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2))
            self.update_areas()
            self.total_arrivals += 1
            is_web = self.tokens > 0
            job_id = self.total_arrivals

            
            service_time = hyperexp(SERVICE_P, SERVICE_L1, SERVICE_L2)
            self.event_log.append(f"{self.env.now:.4f} ARRIVAL  {job_id:04d} {'WEB' if is_web else 'SPIKE'} {service_time:.4f}")

            if self.tokens > 0:
                self.tokens -= 1
                self.env.process(self.processor_sharing_job(service_time, self.web_jobs, self.web_stats, True))
            else:
                self.env.process(self.processor_sharing_job(service_time, self.spike_jobs, self.spike_stats, False))

    def processor_sharing_job(self, service_time, job_list, stats, is_web):
        arrival = self.env.now
        job = {'remaining': service_time}
        job_list.append(job)

        while job['remaining'] > 0:
            yield self.env.timeout(STEP)
            if len(job_list) > 0:
                job['remaining'] -= STEP / len(job_list)

        job_list.remove(job)
        self.update_areas()
        response_time = self.env.now - arrival
        stats.append(response_time)
        self.event_log.append(f"{self.env.now:.4f} COMPLETE {'WEB' if is_web else 'SPIKE'} {response_time:.4f}")

        if is_web:
            self.tokens = min(self.tokens + 1, SI_MAX)
            self.web_completions += 1
        else:
            self.spike_completions += 1

    def update_areas(self):
        now = self.env.now
        dt = now - self.last_time
        self.area_web += len(self.web_jobs) * dt
        self.area_spike += len(self.spike_jobs) * dt
        if self.web_jobs:
            self.busy_web += dt
        if self.spike_jobs:
            self.busy_spike += dt
        self.last_time = now

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

        print("\n[EVENT LOG]")
        for entry in self.event_log:
            print(entry)

def run_sim_and_collect(batch_time):
    env = simpy.Environment()
    sim = ProcessorSharingSim(env)
    env.process(sim.arrival_process())
    env.run(until=batch_time)
    
    return {
        "web_response_times": sim.web_stats,
        "spike_response_times": sim.spike_stats,
        "web_completions": sim.web_completions,
        "spike_completions": sim.spike_completions,
        "total_time": env.now
    }

def batch_means_analysis(n_batches=10, batch_time=1000):
    web_means = []
    spike_means = []

    for _ in range(n_batches):
        results = run_sim_and_collect(batch_time)
        if results["web_response_times"]:
            web_means.append(np.mean(results["web_response_times"]))
        else:
            web_means.append(0)

        if results["spike_response_times"]:
            spike_means.append(np.mean(results["spike_response_times"]))
        else:
            spike_means.append(0)

    def ci(data):
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        margin = CONF_LEVEL * std / np.sqrt(len(data))
        return mean, mean - margin, mean + margin

    web_mean, web_low, web_high = ci(web_means)
    spike_mean, spike_low, spike_high = ci(spike_means)

    print("Batch Means Analysis (99% CI)")
    print(f"Web  : {web_mean:.4f} ± {(web_mean - web_low):.4f} [{web_low:.4f}, {web_high:.4f}]")
    print(f"Spike: {spike_mean:.4f} ± {(spike_mean - spike_low):.4f} [{spike_low:.4f}, {spike_high:.4f}]")



def run_sim():
    env = simpy.Environment()
    sim = ProcessorSharingSim(env)
    env.process(sim.arrival_process())
    env.run(until=SIM_TIME)
    sim.report()
    


if __name__ == '__main__':
    run_sim()
    #batch_means_analysis(n_batches=64, batch_time=1024)
