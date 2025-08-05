from library.rvgs import Exponential, Hyperexponential
from library.rngs import random
from model.mitigation_center import MitigationCenter
from model.processor_sharing_server import ProcessorSharingServer
from engineering.costants import *

class MitigationManager:
    def __init__(self, env, web_server, spike_servers, metrics):
        self.env = env
        self.center = MitigationCenter(env, "MitigationCenter")
        self.web_server = web_server
        self.spike_servers = spike_servers
        self.metrics = metrics  # dizionario condiviso per le metriche

    def handle_job(self, job):
        self.env.process(self._mitigation_process(job))

    def _mitigation_process(self, job):
        try:
            yield self.env.timeout(Exponential(MITIGATION_MEAN))
        except Exception:
            return

        now = self.env.now
        self.metrics["mitigation_completions"] += 1

        if random() < P_FALSE_POSITIVE:
            self.metrics["false_positives"] += 1
            if job.is_legal:
                self.metrics["false_positives_legal"] += 1
            return

        if random() < P_FEEDBACK:
            job.arrival = now
            self.handle_job(job)  # retry
        else:
            service_time = Hyperexponential(SERVICE_P, SERVICE_L1, SERVICE_L2)
            job.remaining = service_time
            job.original_service = service_time
            job.last_updated = now

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
                    new_id = len(self.spike_servers)
                    new_server = ProcessorSharingServer(self.env, f"Spike-{new_id}")
                    self.spike_servers.append(new_server)
                    new_server.arrival(job)