import numpy as np
import simpy
from library.rngs import random
from src.costants import *
from library.rngs import random
from library.rvgs import Exponential, Hyperexponential
from src.job import Job

class MitigationCenter:
    def __init__(self, arrival_time, service_time):
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.start_time = None
        self.completion_time = None

class SystemSimulation:
    """Mitigation Center + Dual Pool PS"""
    
    def __init__(self, env):
        self.env = env
        
        # Mitigation Center
        self.single_queue = []
        self.single_server_busy = False
        self.single_completion_event = None
        self.single_stats = []
        self.single_completions = 0
        self.overflow_jobs = 0
        
        # Sistema dual-pool
        self.web_jobs = []
        self.spike_jobs = []
        self.web_proc = None
        self.spike_proc = None
        self.web_stats = []
        self.spike_stats = []
        self.web_completions = 0
        self.spike_completions = 0
        
        # Statistiche generali
        self.total_arrivals = 0
        self.area_single = 0.0
        self.area_web = 0.0
        self.area_spike = 0.0
        self.busy_single = 0.0
        self.busy_web = 0.0
        self.busy_spike = 0.0
        self.last_time = 0.0
        
        # Avvia il processo di arrivo
        env.process(self.arrival_process())

    def update_areas_optimized(self, now):
        dt = now - self.last_time
        single_queue_size = len(self.single_queue) + (1 if self.single_server_busy else 0)
        self.area_single += single_queue_size * dt
        if self.single_server_busy:
            self.busy_single += dt
        self.area_web += len(self.web_jobs) * dt
        self.area_spike += len(self.spike_jobs) * dt
        if self.web_jobs:
            self.busy_web += dt
        if self.spike_jobs:
            self.busy_spike += dt

    def update_dual_pool_optimized(self, now):
        dt = now - self.last_time
        n_web = len(self.web_jobs)
        if n_web > 0:
            for job in self.web_jobs:
                served = dt / n_web
                job.remaining = max(job.remaining - served, 0.0)
                job.last_updated = now
        n_spike = len(self.spike_jobs)
        if n_spike > 0:
            for job in self.spike_jobs:
                served = dt / n_spike
                job.remaining = max(job.remaining - served, 0.0)
                job.last_updated = now
    
    def arrival_process(self):
        """Processo di arrivo delle richieste al sistema"""
        while self.total_arrivals < N_ARRIVALS:
            # Genera inter-arrival time
            yield self.env.timeout(Exponential(INTERARRIVAL_MEAN))
            
            now = self.env.now
            self.update_areas_optimized(now)
            self.update_dual_pool_optimized(now)
            self.last_time = now
            
            self.total_arrivals += 1
            # QUA SI DEVE CAMBIARE
            self.handle_mitigation_center_arrival(now)
    
    def handle_mitigation_center_arrival(self, now):
        """Gestisce l'arrivo al centro singolo"""
        service_time = Exponential(SERVICE_MEAN_SINGLE)
        job = MitigationCenter(now, service_time)
        
        if not self.single_server_busy:
            # Server libero, inizia subito il servizio
            self.single_server_busy = True
            job.start_time = now
            self.env.process(self.handle_mitigation_center_service(job))
        else:
            # Server occupato, aggiungi alla coda
            self.single_queue.append(job)
    
    def handle_mitigation_center_service(self, job):
        """Processo di servizio nel centro singolo"""
        try:
            yield self.env.timeout(job.service_time)
            
            now = self.env.now
            self.update_areas_optimized(now)
            self.update_dual_pool_optimized(now)
            self.last_time = now
            
            job.completion_time = now
            response_time = now - job.arrival_time
            self.single_stats.append(response_time)
            self.single_completions += 1
            
            # Feedback probabilistico
            if random() < P_FEEDBACK:
                # Reinserisce il job nel sistema
                new_service_time = Exponential(SERVICE_MEAN_SINGLE)
                feedback_job = Job(now, new_service_time)
                
                self.env.process(self.handle_mitigation_center_service(feedback_job))

            else:
                # Job completato, controlla se c'è coda
                if self.single_queue:
                    next_job = self.single_queue.pop(0)
                    next_job.start_time = now
                    self.env.process(self.handle_route_to_servers(now, next_job))
                else:
                    self.single_server_busy = False
                    
        except simpy.Interrupt:
            # Gestione interruzioni se necessario
            pass
    
    def handle_handle_route_to_servers(self, now, job):
        """Gestisce l'overflow al sistema dual-pool"""
        self.overflow_jobs += 1
        service_time = Hyperexponential(SERVICE_P, SERVICE_L1, SERVICE_L2)
        job = Job(now, service_time, source='overflow')
        
        # Logica di routing nel dual-pool
        if len(self.web_jobs) < SI_MAX:
            self.web_jobs.append(job)
            self.schedule_completion(self.web_jobs, 'web')
        else:
            self.spike_jobs.append(job)
            self.schedule_completion(self.spike_jobs, 'spike')
    
    def update_dual_pool(self, now):
        """Aggiorna il pool dual (web e spike)"""
        dt = now - self.last_time
        
        # Aggiorna web pool
        n_web = len(self.web_jobs)
        if n_web > 0:
            for job in self.web_jobs:
                served = dt / n_web
                job.remaining = max(job.remaining - served, 0.0)
                job.last_updated = now
        
        # Aggiorna spike pool
        n_spike = len(self.spike_jobs)
        if n_spike > 0:
            for job in self.spike_jobs:
                served = dt / n_spike
                job.remaining = max(job.remaining - served, 0.0)
                job.last_updated = now
    
    def update_areas(self, now):
        """Aggiorna le aree per calcolare metriche temporali"""
        dt = now - self.last_time
        
        # Area centro singolo
        single_queue_size = len(self.single_queue) + (1 if self.single_server_busy else 0)
        self.area_single += single_queue_size * dt
        if self.single_server_busy:
            self.busy_single += dt
        
        # Aree dual-pool
        self.area_web += len(self.web_jobs) * dt
        self.area_spike += len(self.spike_jobs) * dt
        if self.web_jobs:
            self.busy_web += dt
        if self.spike_jobs:
            self.busy_spike += dt
    
    def schedule_completion(self, job_list, pool_type):
        """Schedula il completamento per il dual-pool"""
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
        """Evento di completamento per il dual-pool"""
        try:
            yield self.env.timeout(delay)
        except simpy.Interrupt:
            return
        
        now = self.env.now
        # Sostituisci le chiamate con versioni ottimizzate
        self.update_areas_optimized(now)
        self.update_dual_pool_optimized(now)
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
        """Report delle statistiche del sistema integrato"""
        def ci(data):
            if len(data) < 2:
                return 0, 0, 0
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            margin = CONF_LEVEL * std / np.sqrt(len(data))
            return mean, mean - margin, mean + margin
        
        total_time = self.env.now
        
        # Statistiche centro singolo
        if self.single_stats:
            single_mean, single_low, single_high = ci(self.single_stats)
        else:
            single_mean = single_low = single_high = 0
        
        # Statistiche dual-pool
        web_mean, web_low, web_high = ci(self.web_stats)
        spike_mean, spike_low, spike_high = ci(self.spike_stats)
        
        # Metriche di utilizzo e throughput
        avg_num_single = self.area_single / total_time
        avg_num_web = self.area_web / total_time
        avg_num_spike = self.area_spike / total_time
        
        util_single = self.busy_single / total_time
        util_web = self.busy_web / total_time
        util_spike = self.busy_spike / total_time
        
        tput_single = self.single_completions / total_time
        tput_web = self.web_completions / total_time
        tput_spike = self.spike_completions / total_time
        total_tput = (self.single_completions + self.web_completions + self.spike_completions) / total_time
        
        print("=" * 60)
        print("INTEGRATED SYSTEM SIMULATION REPORT")
        print("=" * 60)
        print(f"Total Arrivals             : {self.total_arrivals}")
        print(f"Single Center Completions  : {self.single_completions}")
        print(f"Overflow Jobs              : {self.overflow_jobs}")
        print(f"Web Pool Completions       : {self.web_completions}")
        print(f"Spike Pool Completions     : {self.spike_completions}")
        print()
        print("RESPONSE TIMES:")
        print(f"Single Center Avg Response : {single_mean:.4f} ± {(single_mean - single_low):.4f} (99% CI [{single_low:.4f}, {single_high:.4f}])")
        print(f"Web Pool Avg Response      : {web_mean:.4f} ± {(web_mean - web_low):.4f} (99% CI [{web_low:.4f}, {web_high:.4f}])")
        print(f"Spike Pool Avg Response    : {spike_mean:.4f} ± {(spike_mean - spike_low):.4f} (99% CI [{spike_low:.4f}, {spike_high:.4f}])")
        print()
        print("QUEUE LENGTHS:")
        print(f"Avg Number in Single Center: {avg_num_single:.4f}")
        print(f"Avg Number in Web Pool     : {avg_num_web:.4f}")
        print(f"Avg Number in Spike Pool   : {avg_num_spike:.4f}")
        print()
        print("UTILIZATIONS:")
        print(f"Single Center Utilization  : {util_single:.4f}")
        print(f"Web Pool Utilization       : {util_web:.4f}")
        print(f"Spike Pool Utilization     : {util_spike:.4f}")
        print()
        print("THROUGHPUTS:")
        print(f"Single Center Throughput   : {tput_single:.4f}")
        print(f"Web Pool Throughput        : {tput_web:.4f}")
        print(f"Spike Pool Throughput      : {tput_spike:.4f}")
        print(f"Total System Throughput    : {total_tput:.4f}")
        print("=" * 60)


def single_run_sim():
    """Esegue la simulazione del sistema integrato"""
    env = simpy.Environment()
    system = SystemSimulation(env)
    env.run()
    system.report()

    return system

if __name__ == "__main__":
    sim = single_run_sim()