import simpy
import numpy as np
from library.rvgs import Hyperexponential

# Parameters for verification
ARRIVAL_P = 0.03033
ARRIVAL_L1 = 0.4044
ARRIVAL_L2 = 12.9289
SERVICE_P = 0.03033
SERVICE_L1 = 0.3791
SERVICE_L2 = 12.1208
N_ARRIVALS = 10  # Fixed number of arrivals for testing
VERBOSE = True   # Enable detailed logging

class Job:
    def __init__(self, job_id, arrival_time, service_time):
        self.id = job_id
        self.arrival = arrival_time
        self.remaining = service_time
        self.original_service = service_time
        self.last_updated = arrival_time

class SingleServerPS:
    def __init__(self, env):
        self.env = env
        self.jobs = []  # Active jobs in the system
        self.proc = None  # Current completion process
        
        # Statistics
        self.completed_jobs = []
        self.total_arrivals = 0
        self.completions = 0
        self.last_time = 0.0
        
        # For area calculations
        self.area = 0.0
        self.busy_time = 0.0
        
        # Start arrival process
        env.process(self.arrival_process())
        
    def log(self, message):
        """Print log message if verbose mode is enabled"""
        if VERBOSE:
            print(f"t={self.env.now:.4f} {message}")
    
    def get_next_completion_time(self):
        """Calculate the time when the next job will complete"""
        if not self.jobs:
            return None
            
        # Find job with minimum remaining time
        next_job = min(self.jobs, key=lambda j: j.remaining)
        n = len(self.jobs)
        
        # Time until completion = remaining time * number of jobs
        time_to_completion = next_job.remaining * n
        completion_time = self.env.now + time_to_completion
        
        return completion_time, next_job
    
    def print_completion_times(self, event_type):
        """Print completion times for all jobs currently in system"""
        if not self.jobs:
            self.log(f"{event_type} - No jobs in system")
            return
            
        self.log(f"{event_type} - Completion times:")
        
        # Sort jobs by remaining time to show completion order
        sorted_jobs = sorted(self.jobs, key=lambda j: j.remaining)
        
        for i, job in enumerate(sorted_jobs):
            n = len(self.jobs)  # Current number of jobs
            # Time until this job completes = remaining time * current number of jobs
            time_to_completion = job.remaining * n
            completion_time = self.env.now + time_to_completion
            
            self.log(f"  J{job.id:04d}: remaining={job.remaining:.4f}, "
                    f"completion_time={completion_time:.4f} "
                    f"(in {time_to_completion:.4f} time units)")
    
    def arrival_process(self):
        """Generate arrivals until N_ARRIVALS is reached"""
        while self.total_arrivals < N_ARRIVALS:
            # Wait for next arrival
            interarrival_time = Hyperexponential(ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2)
            yield self.env.timeout(interarrival_time)
            
            now = self.env.now
            self.update_system(now)
            
            # Create new job
            self.total_arrivals += 1
            service_time = Hyperexponential(SERVICE_P, SERVICE_L1, SERVICE_L2)
            job = Job(self.total_arrivals, now, service_time)
            
            self.log(f"ARRIVAL {job.id:04d} SERVICE={service_time:.4f}")
            
            # Add job to system
            self.jobs.append(job)
            self.log(f"Jobs in system: {len(self.jobs)} -> {[f'J{j.id}({j.remaining:.4f})' for j in self.jobs]}")
            
            # Print completion times after arrival
            self.print_completion_times("AFTER_ARRIVAL")
            
            # Schedule next completion
            self.schedule_completion()
    
    def update_system(self, now):
        """Update remaining times for all jobs using processor sharing"""
        dt = now - self.last_time
        if dt <= 0:
            return
            
        # Update area statistics
        n = len(self.jobs)
        self.area += n * dt
        if n > 0:
            self.busy_time += dt
        
        # Update remaining times
        if n > 0:
            service_per_job = dt / n
            self.log(f"UPDATE: dt={dt:.4f}, n={n}, service_per_job={service_per_job:.4f}")
            
            for job in self.jobs:
                old_remaining = job.remaining
                job.remaining = max(job.remaining - service_per_job, 0.0)
                job.last_updated = now
                self.log(f"  J{job.id}: {old_remaining:.4f} -> {job.remaining:.4f}")
        
        self.last_time = now
    
    def schedule_next_completion(self):
        """Helper method to schedule next completion without self-interrupt"""
        yield self.env.timeout(0)  # Small delay to avoid self-interrupt
        self.schedule_completion()
    
    def schedule_completion(self):
        """Schedule the next completion event"""
        if not self.jobs:
            return
            
        # Find job with minimum remaining time
        next_job = min(self.jobs, key=lambda j: j.remaining)
        n = len(self.jobs)
        
        # Calculate delay until completion
        delay = next_job.remaining * n
        
        self.log(f"SCHEDULE: J{next_job.id} will complete in {delay:.4f} (remaining={next_job.remaining:.4f}, n={n})")
        
        # Cancel previous completion if exists and it's not the current process
        if self.proc and self.proc.is_alive and self.proc != self.env.active_process:
            self.proc.interrupt()
        
        # Schedule new completion only if we're not already in a completion process
        if self.env.active_process != self.proc:
            self.proc = self.env.process(self.completion_event(next_job, delay))
    
    def completion_event(self, job, delay):
        """Handle job completion"""
        try:
            yield self.env.timeout(delay)
        except simpy.Interrupt:
            self.log("COMPLETION interrupted")
            return
        
        now = self.env.now
        self.update_system(now)
        
        # Complete the job if it's still in the system
        if job in self.jobs:
            self.jobs.remove(job)
            response_time = now - job.arrival
            self.completed_jobs.append({
                'id': job.id,
                'arrival': job.arrival,
                'completion': now,
                'response_time': response_time,
                'service_time': job.original_service
            })
            self.completions += 1
            
            self.log(f"COMPLETE J{job.id} RESPONSE_TIME={response_time:.4f}")
            self.log(f"Jobs remaining: {len(self.jobs)} -> {[f'J{j.id}({j.remaining:.4f})' for j in self.jobs]}")
            
            # Print completion times after completion
            self.print_completion_times("AFTER_COMPLETION")
            
            # Schedule next completion - but avoid self-interrupt
            if self.jobs:  # Only if there are still jobs
                self.env.process(self.schedule_next_completion())
    
    def report(self):
        """Print simulation results"""
        total_time = self.env.now
        
        print(f"\n{'='*60}")
        print(f"SIMULATION RESULTS")
        print(f"{'='*60}")
        print(f"Total simulation time  : {total_time:.4f}")
        print(f"Total arrivals         : {self.total_arrivals}")
        print(f"Total completions      : {self.completions}")
        print(f"Jobs still in system   : {len(self.jobs)}")
        
        if self.completed_jobs:
            response_times = [job['response_time'] for job in self.completed_jobs]
            print(f"Avg response time      : {np.mean(response_times):.4f}")
            print(f"Avg number in system   : {self.area / total_time:.4f}")
            print(f"Utilization            : {self.busy_time / total_time:.4f}")
            print(f"Throughput             : {self.completions / total_time:.4f}")
        
        print(f"\nCOMPLETED JOBS:")
        print(f"{'ID':>4} {'Arrival':>8} {'Completion':>10} {'Response':>8} {'Service':>8}")
        print(f"{'-'*44}")
        for job in self.completed_jobs:
            print(f"{job['id']:>4} {job['arrival']:>8.4f} {job['completion']:>10.4f} "
                  f"{job['response_time']:>8.4f} {job['service_time']:>8.4f}")
        
        if self.jobs:
            print(f"\nJOBS STILL IN SYSTEM:")
            print(f"{'ID':>4} {'Arrival':>8} {'Remaining':>10} {'Original':>8}")
            print(f"{'-'*34}")
            for job in self.jobs:
                print(f"{job.id:>4} {job.arrival:>8.4f} {job.remaining:>10.4f} {job.original_service:>8.4f}")

def run_simulation():
    """Run a single simulation"""
    # Set random seed for reproducibility    
    print("Starting Processor Sharing Simulation")
    print(f"Number of arrivals: {N_ARRIVALS}")
    print(f"Verbose logging: {VERBOSE}")
    print("-" * 60)
    
    env = simpy.Environment()
    sim = SingleServerPS(env)
    
    # Run until all arrivals are generated and processed
    # We'll run until no more events are scheduled
    env.run()
    
    sim.report()
    
    return sim

if __name__ == "__main__":
    sim = run_simulation()