import numpy as np
from src.library.rngs import random 
from src.library.rvgs import Exponential

# Parametri realistici (Cloudflare-like)
INTERARRIVAL_MEAN = 0.00002   # 50 000 rps
SERVICE_MEAN = 0.00002  # 20 μs → μ = 50 000 req/s
P_FEEDBACK = 0.02             # 2% probabilità di incertezza (feedback)

# Parametri batch means
BATCH_SIZE = 1.0              # tempo di simulazione per batch (in secondi)
N_BATCHES = 300
CONF_LEVEL = 1.96             # 95% confidence level

def exponential(mean):
    return Exponential(1.0 / mean)

class SumStats:
    def __init__(self):
        self.delay = 0.0
        self.wait = 0.0
        self.service = 0.0
        self.completed_jobs = 0

    def update(self, delay, wait, service):
        self.delay += delay
        self.wait += wait
        self.service += service
        self.completed_jobs += 1

    def average_wait(self):
        if self.completed_jobs == 0:
            return 0.0
        return self.wait / self.completed_jobs

def simulate_batch(batch_time):
    arrival = 0.0
    departure = 0.0
    stats = SumStats()
    jobs = []

    while arrival < batch_time:
        interarrival = exponential(INTERARRIVAL_MEAN)
        service = exponential(SERVICE_MEAN)
        arrival += interarrival
        jobs.append((arrival, service))

    i = 0
    while i < len(jobs):
        arrival, service = jobs[i]
        if arrival < departure:
            delay = departure - arrival
        else:
            delay = 0.0
        wait = delay + service
        departure = arrival + wait
        stats.update(delay, wait, service)

        # Feedback: reinserimento con nuova arrival se incertezza
        if random.random() < P_FEEDBACK and departure < batch_time:
            new_interarrival = exponential(INTERARRIVAL_MEAN)
            new_service = exponential(SERVICE_MEAN)
            new_arrival = departure + new_interarrival
            jobs.append((new_arrival, new_service))

        i += 1

    return stats.average_wait()

def batch_means():
    wait_times = []
    for _ in range(N_BATCHES):
        avg_wait = simulate_batch(BATCH_SIZE)
        wait_times.append(avg_wait)

    mean = np.mean(wait_times)
    std = np.std(wait_times, ddof=1)
    margin = CONF_LEVEL * std / np.sqrt(N_BATCHES)
    low = mean - margin
    high = mean + margin

    print(f"\n[Batch Means] {N_BATCHES} batch da {BATCH_SIZE} sec con feedback probabilistico")
    print(f"Arrivi: ~{1 / INTERARRIVAL_MEAN:.0f} req/s | Servizio medio: {SERVICE_MEAN*1000:.1f} ms | Feedback: {P_FEEDBACK*100:.1f}%")
    print(f"Average Response Time : {mean*1000:.4f} ms")
    print(f"95% Confidence Interval: [{low*1000:.4f}, {high*1000:.4f}] ± {margin*1000:.4f} ms")

if __name__ == "__main__":
    batch_means()
