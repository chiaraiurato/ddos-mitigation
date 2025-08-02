import simpy
import numpy as np
from library.rngs import random

# Parametri del sistema
MITIGATION_CAPACITY = 10       # Capacità centro di mitigazione
INTERARRIVAL_MEAN = 0.001      # Tempo tra arrivi
SERVICE_MEAN_MITIGATION = 0.0012  # Tempo servizio mitigazione
P_FEEDBACK = 0.02              # Probabilità feedback

# Parametri routing post-mitigazione
WEB_SERVER_CAPACITY = 20       # Capacità web server
# Spike server ha capacità infinita (processor sharing)

# Parametri distribuzione iperexponenziale per i server finali
SERVICE_P = 0.03033
SERVICE_L1 = 0.3791
SERVICE_L2 = 12.1208

# Parametri simulazione
N_ARRIVALS = 1000
CONF_LEVEL = 2.576

def exponential(mean):
    """Distribuzione esponenziale"""
    return np.random.exponential(mean)

def hyperexp(p, l1, l2):
    """Distribuzione iperexponenziale"""
    return np.random.exponential(1/l1) if random() < p else np.random.exponential(1/l2)

class Request:
    """Richiesta nel sistema"""
    def __init__(self, arrival_time, request_id):
        self.arrival_time = arrival_time
        self.request_id = request_id
        self.mitigation_start = None
        self.mitigation_end = None
        self.final_destination = None  # 'web' o 'spike'
        self.completion_time = None

class MitigatedRequest:
    """Richiesta dopo mitigazione, pronta per i server finali"""
    def __init__(self, original_request, service_time, destination):
        self.original_request = original_request
        self.arrival_time = original_request.mitigation_end
        self.service_time = service_time
        self.remaining_time = service_time
        self.destination = destination  # 'web' o 'spike'
        self.last_updated = self.arrival_time

class CorrectSystem:
    """Sistema con architettura corretta: Mitigazione → Web/Spike"""
    
    def __init__(self, env):
        self.env = env
        
        # Centro di Mitigazione (obbligatorio per tutti)
        self.mitigation_queue = []
        self.mitigation_busy = False
        self.mitigation_stats = []
        self.mitigation_completions = 0
        
        # Web Server (capacità limitata, FIFO)
        self.web_queue = []
        self.web_busy = False
        self.web_stats = []
        self.web_completions = 0
        
        # Spike Server (capacità infinita, processor sharing)
        self.spike_jobs = []
        self.spike_stats = []
        self.spike_completions = 0
        self.spike_next_completion = None
        
        # Statistiche generali
        self.total_arrivals = 0
        self.requests_to_web = 0
        self.requests_to_spike = 0
        self.area_mitigation = 0.0
        self.area_web = 0.0
        self.area_spike = 0.0
        self.busy_mitigation = 0.0
        self.busy_web = 0.0
        self.busy_spike = 0.0
        self.last_time = 0.0
        
        # Avvia processo di arrivo
        env.process(self.arrival_process())
    
    def update_metrics(self, now):
        """Aggiorna tutte le metriche del sistema"""
        dt = now - self.last_time
        if dt <= 0:
            return
        
        # Centro mitigazione
        mitigation_size = len(self.mitigation_queue) + (1 if self.mitigation_busy else 0)
        self.area_mitigation += mitigation_size * dt
        if self.mitigation_busy:
            self.busy_mitigation += dt
        
        # Web server
        web_size = len(self.web_queue) + (1 if self.web_busy else 0)
        self.area_web += web_size * dt
        if self.web_busy:
            self.busy_web += dt
        
        # Spike server (processor sharing)
        n_spike = len(self.spike_jobs)
        if n_spike > 0:
            self.area_spike += n_spike * dt
            self.busy_spike += dt
            
            # Aggiorna remaining time per tutti i job nello spike server
            service_per_job = dt / n_spike
            for job in self.spike_jobs:
                job.remaining_time = max(0.0, job.remaining_time - service_per_job)
                job.last_updated = now
        
        self.last_time = now
    
    def arrival_process(self):
        """Processo di arrivo - TUTTE le richieste vanno alla mitigazione"""
        while self.total_arrivals < N_ARRIVALS:
            yield self.env.timeout(exponential(INTERARRIVAL_MEAN))
            
            now = self.env.now
            self.update_metrics(now)
            self.total_arrivals += 1
            
            # Crea nuova richiesta
            request = Request(now, self.total_arrivals)
            
            # TUTTE le richieste vanno al centro di mitigazione
            self.handle_mitigation_arrival(request, now)
    
    def handle_mitigation_arrival(self, request, now):
        """Gestisce arrivo al centro di mitigazione"""
        if not self.mitigation_busy:
            # Centro mitigazione libero
            self.mitigation_busy = True
            request.mitigation_start = now
            self.env.process(self.mitigation_service(request))
        else:
            # Centro mitigazione occupato, metti in coda
            self.mitigation_queue.append(request)
    
    def mitigation_service(self, request):
        """Processo di servizio del centro mitigazione"""
        try:
            # Simula il tempo di mitigazione
            mitigation_time = exponential(SERVICE_MEAN_MITIGATION)
            yield self.env.timeout(mitigation_time)
            
            now = self.env.now
            self.update_metrics(now)
            
            request.mitigation_end = now
            mitigation_response_time = now - request.arrival_time
            self.mitigation_stats.append(mitigation_response_time)
            self.mitigation_completions += 1
            
            # Feedback probabilistico
            if random() < P_FEEDBACK:
                # Richiesta richiede ulteriore mitigazione
                feedback_request = Request(now, f"{request.request_id}_feedback")
                feedback_request.mitigation_start = now
                self.env.process(self.mitigation_service(feedback_request))
            else:
                # Mitigazione completata, instrada verso web o spike
                self.route_after_mitigation(request, now)
            
            # Processa prossima richiesta in coda mitigazione
            if self.mitigation_queue:
                next_request = self.mitigation_queue.pop(0)
                next_request.mitigation_start = now
                self.env.process(self.mitigation_service(next_request))
            else:
                self.mitigation_busy = False
                
        except simpy.Interrupt:
            pass
    
    def route_after_mitigation(self, request, now):
        """Instrada la richiesta mitigata verso web o spike server"""
        # Genera tempo di servizio per il server finale
        final_service_time = hyperexp(SERVICE_P, SERVICE_L1, SERVICE_L2)
        
        # Logica di routing: prova prima web server
        if len(self.web_queue) < WEB_SERVER_CAPACITY and not self.web_busy:
            # Vai al web server
            request.final_destination = 'web'
            self.requests_to_web += 1
            mitigated_request = MitigatedRequest(request, final_service_time, 'web')
            self.handle_web_arrival(mitigated_request, now)
        else:
            # Web server pieno o occupato, vai allo spike server
            request.final_destination = 'spike'
            self.requests_to_spike += 1
            mitigated_request = MitigatedRequest(request, final_service_time, 'spike')
            self.handle_spike_arrival(mitigated_request, now)
    
    def handle_web_arrival(self, mitigated_request, now):
        """Gestisce arrivo al web server"""
        if not self.web_busy:
            # Web server libero
            self.web_busy = True
            self.env.process(self.web_service(mitigated_request))
        else:
            # Web server occupato, metti in coda
            self.web_queue.append(mitigated_request)
    
    def web_service(self, mitigated_request):
        """Processo di servizio del web server"""
        try:
            yield self.env.timeout(mitigated_request.service_time)
            
            now = self.env.now
            self.update_metrics(now)
            
            # Calcola response time totale (dall'arrivo iniziale)
            total_response_time = now - mitigated_request.original_request.arrival_time
            self.web_stats.append(total_response_time)
            self.web_completions += 1
            
            # Processa prossima richiesta in coda web
            if self.web_queue:
                next_request = self.web_queue.pop(0)
                self.env.process(self.web_service(next_request))
            else:
                self.web_busy = False
                
        except simpy.Interrupt:
            pass
    
    def handle_spike_arrival(self, mitigated_request, now):
        """Gestisce arrivo allo spike server (processor sharing)"""
        self.spike_jobs.append(mitigated_request)
        self.schedule_spike_completion()
    
    def schedule_spike_completion(self):
        """Schedula prossimo completamento nello spike server"""
        if not self.spike_jobs:
            if self.spike_next_completion:
                self.spike_next_completion = None
            return
        
        # Trova job che finirà prima
        min_job = min(self.spike_jobs, key=lambda j: j.remaining_time)
        n_jobs = len(self.spike_jobs)
        delay = max(0.001, min_job.remaining_time * n_jobs)  # Evita delay zero
        
        # Cancella evento precedente SOLO se non è il processo corrente
        if (self.spike_next_completion and 
            self.spike_next_completion.is_alive and 
            self.spike_next_completion != self.env.active_process):
            try:
                self.spike_next_completion.interrupt()
            except RuntimeError:
                pass  # Ignora errori di auto-interruzione
        
        # Schedula nuovo evento
        self.spike_next_completion = self.env.process(self.spike_completion_event(delay))
    
    def spike_completion_event(self, delay):
        """Evento di completamento per spike server"""
        try:
            yield self.env.timeout(delay)
        except simpy.Interrupt:
            return
        
        now = self.env.now
        self.update_metrics(now)
        
        # Trova e rimuovi job completato
        if self.spike_jobs:
            completed_jobs = [job for job in self.spike_jobs if job.remaining_time <= 0.001]
            
            for completed_job in completed_jobs:
                self.spike_jobs.remove(completed_job)
                
                # Calcola response time totale
                total_response_time = now - completed_job.original_request.arrival_time
                self.spike_stats.append(total_response_time)
                self.spike_completions += 1
        
        # Schedula prossimo completamento SOLO se ci sono ancora job
        if self.spike_jobs:
            self.env.process(self.schedule_next_spike_completion())
    
    def schedule_next_spike_completion(self):
        """Helper per schedulare il prossimo completamento senza auto-interruzione"""
        yield self.env.timeout(0)  # Piccolo delay per evitare auto-interruzione
        self.schedule_spike_completion()
    
    def report(self):
        """Report delle statistiche del sistema"""
        def ci(data):
            if len(data) < 2:
                return 0, 0, 0
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            margin = CONF_LEVEL * std / np.sqrt(len(data))
            return mean, mean - margin, mean + margin
        
        total_time = self.env.now
        
        print("=" * 60)
        print("CORRECT SYSTEM ARCHITECTURE REPORT")
        print("=" * 60)
        print("SYSTEM FLOW: All Requests → Mitigation → Web/Spike")
        print("=" * 60)
        
        print(f"Total Arrivals               : {self.total_arrivals:,}")
        print(f"Mitigation Completions       : {self.mitigation_completions:,}")
        print(f"Requests to Web Server       : {self.requests_to_web:,}")
        print(f"Requests to Spike Server     : {self.requests_to_spike:,}")
        print(f"Web Server Completions       : {self.web_completions:,}")
        print(f"Spike Server Completions     : {self.spike_completions:,}")
        print()
        
        # Response times
        mitigation_mean, mitigation_low, mitigation_high = ci(self.mitigation_stats) if self.mitigation_stats else (0, 0, 0)
        web_mean, web_low, web_high = ci(self.web_stats) if self.web_stats else (0, 0, 0)
        spike_mean, spike_low, spike_high = ci(self.spike_stats) if self.spike_stats else (0, 0, 0)
        
        print("RESPONSE TIMES (Total: Arrival → Completion):")
        print(f"Mitigation Only              : {mitigation_mean:.4f} ± {(mitigation_mean-mitigation_low):.4f}")
        if self.web_stats:
            print(f"Total via Web Server         : {web_mean:.4f} ± {(web_mean-web_low):.4f}")
        if self.spike_stats:
            print(f"Total via Spike Server       : {spike_mean:.4f} ± {(spike_mean-spike_low):.4f}")
        print()
        
        # Utilizations
        mitigation_util = self.busy_mitigation / total_time
        web_util = self.busy_web / total_time
        spike_util = self.busy_spike / total_time
        
        print("UTILIZATIONS:")
        print(f"Mitigation Center            : {mitigation_util:.4f}")
        print(f"Web Server                   : {web_util:.4f}")
        print(f"Spike Server                 : {spike_util:.4f}")
        print()
        
        # Throughputs
        mitigation_tput = self.mitigation_completions / total_time
        web_tput = self.web_completions / total_time
        spike_tput = self.spike_completions / total_time
        
        print("THROUGHPUTS:")
        print(f"Mitigation Center            : {mitigation_tput:.4f}")
        print(f"Web Server                   : {web_tput:.4f}")
        print(f"Spike Server                 : {spike_tput:.4f}")
        print()
        
        # Routing statistics
        total_routed = self.requests_to_web + self.requests_to_spike
        if total_routed > 0:
            web_percentage = (self.requests_to_web / total_routed) * 100
            spike_percentage = (self.requests_to_spike / total_routed) * 100
            print("ROUTING DISTRIBUTION:")
            print(f"To Web Server                : {web_percentage:.1f}%")
            print(f"To Spike Server              : {spike_percentage:.1f}%")
        
        print("=" * 60)

def run_correct_simulation():
    """Esegue la simulazione con architettura corretta"""
    env = simpy.Environment()
    system = CorrectSystem(env)
    env.run()
    system.report()
    return system

if __name__ == "__main__":
    print("CORRECT SYSTEM ARCHITECTURE")
    print("=" * 50)
    print("Flow: All Requests → Mitigation → Web/Spike")
    print(f"Mitigation Capacity: {MITIGATION_CAPACITY}")
    print(f"Web Server Capacity: {WEB_SERVER_CAPACITY}")
    print(f"Spike Server: Infinite capacity (processor sharing)")
    print(f"Arrivals per simulation: {N_ARRIVALS:,}")
    print("=" * 50)
    
    system = run_correct_simulation()
    
    print(f"\nKEY INSIGHTS:")
    print("=" * 50)
    print(f"• ALL {system.total_arrivals} requests passed through mitigation")
    print(f"• {system.requests_to_web} went to Web Server")
    print(f"• {system.requests_to_spike} went to Spike Server")
    print(f"• Feedback rate: {(system.mitigation_completions - system.total_arrivals)/system.total_arrivals*100:.1f}%")
    print("=" * 50)