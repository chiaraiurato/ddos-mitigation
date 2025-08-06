from library.rvgs import Exponential, Hyperexponential
from library.rngs import random
from model.mitigation_center import MitigationCenter
from model.processor_sharing_server import ProcessorSharingServer
from engineering.costants import *

class MitigationManager:
    def __init__(self, env, web_server, spike_servers, metrics):
        self.env = env
        self.center = MitigationCenter(env, "MitigationCenter", capacity=MITIGATION_CAPACITY_VERIFICATION)

        self.web_server = web_server
        self.spike_servers = spike_servers
        self.metrics = metrics 

    def handle_job(self, job):
        if not self.center.has_capacity():
            # Inizializza lista di dettagli scarti se non esiste
            if "discarded_detail" not in self.metrics:
                self.metrics["discarded_detail"] = []

            # Aggiungi info sul job scartato
            self.metrics["discarded_detail"].append({
                "time": self.env.now,
                "job_id": job.id,
                "is_legal": job.is_legal
            })

            # Incrementa contatore aggregato
            self.metrics["discarded_mitigation"] = self.metrics.get("discarded_mitigation", 0) + 1
            return

        if job.is_legal:
            self.metrics["processed_legal"] = self.metrics.get("processed_legal", 0) + 1
        else:
            self.metrics["processed_illegal"] = self.metrics.get("processed_illegal", 0) + 1

        self._mitigation_process(job)


    def _mitigation_process(self, job):
        try:
            self.center.arrival(job)
        except Exception:
            return

        now = self.env.now
        self.metrics["mitigation_completions"] += 1

        if random() < P_FALSE_POSITIVE:
            self.metrics["false_positives"] += 1
            if job.is_legal:
                self.metrics["false_positives_legal"] += 1
            return

        if random() < P_FEEDBACK_VERIFICATION:
            job.arrival = now
            self.handle_job(job)  # retry
        else:
            service_time = Hyperexponential(SERVICE_P, SERVICE_L1, SERVICE_L2)
            job.remaining = service_time
            job.original_service = service_time
            job.last_updated = now        
            # Route the job to the appropriate server (eg. E(N_s) < Threshold)
            if len(self.web_server.jobs) < MAX_WEB_CAPACITY:
                self.web_server.arrival(job)
            else:
                assigned = False
                for server in self.spike_servers:
                    if len(server.jobs) < MAX_SPIKE_CAPACITY:
                        server.arrival(job)
                        assigned = True
                        break
                if not assigned:
                    # Add new spike server if all are busy
                    new_id = len(self.spike_servers)
                    new_server = ProcessorSharingServer(self.env, f"Spike-{new_id}")
                    self.spike_servers.append(new_server)
                    new_server.arrival(job)
