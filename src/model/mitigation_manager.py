from library.rngs import random

from model.mitigation_center import MitigationCenter
from model.processor_sharing_server import ProcessorSharingServer
from engineering.costants import *
from engineering.distributions import get_service_time


class MitigationManager:
    """
    Responsabile di:
    - Accodare i job al MitigationCenter (se c'è capacità) o scartarli (e loggare).
    - Applicare la logica di classificazione (false positive) e feedback.
    - Fare routing verso Web / Spike una volta superata la mitigazione.
    """
    def __init__(self, env, web_server, spike_servers, metrics, mode):
        self.env = env
        self.mode = mode
        self.metrics = metrics

        # MitigationCenter aggiornata: accetta metrics per aggiornare "mitigation_completions" al VERO completamento
        self.center = MitigationCenter(
            env,
            "MitigationCenter",
            capacity=MITIGATION_CAPACITY_VERIFICATION,
            metrics=self.metrics
        )

        self.web_server = web_server
        self.spike_servers = spike_servers

    def handle_job(self, job):
        """
        Primo touchpoint: decide se scartare (coda piena) o mandare in mitigazione.
        Tiene anche traccia di processed_legal/illegal.
        """
        if not self.center.has_capacity():
            # inizializza lista scarti se non esiste
            if "discarded_detail" not in self.metrics:
                self.metrics["discarded_detail"] = []
            self.metrics["discarded_detail"].append({
                "time": self.env.now,
                "job_id": job.id,
                "is_legal": job.is_legal
            })
            self.metrics["discarded_mitigation"] = self.metrics.get("discarded_mitigation", 0) + 1
            return

        # conteggio processati in ingresso alla mitigazione
        if job.is_legal:
            self.metrics["processed_legal"] = self.metrics.get("processed_legal", 0) + 1
        else:
            self.metrics["processed_illegal"] = self.metrics.get("processed_illegal", 0) + 1

        self._mitigation_process(job)

    def _mitigation_process(self, job):
        """
        Inserisce il job nel centro di mitigazione.
        NOTA: NON incrementa più mitigation_completions qui!
        L'incremento avviene nel MitigationCenter al completamento reale.
        """
        try:
            self.center.arrival(job)
        except Exception:
            return

        now = self.env.now

        # Classificazione: false positive => job droppato
        if random() < P_FALSE_POSITIVE:
            self.metrics["false_positives"] = self.metrics.get("false_positives", 0) + 1
            if job.is_legal:
                self.metrics["false_positives_legal"] = self.metrics.get("false_positives_legal", 0) + 1
            return

        # Feedback (retry) verso la mitigazione
        if random() < P_FEEDBACK_VERIFICATION:
            job.arrival = now
            self.handle_job(job)  # retry
            return

        # Superata la mitigazione: assegna tempo di servizio e route verso Web/Spike
        service_time = get_service_time(self.mode)
        job.remaining = service_time
        job.original_service = service_time
        job.last_updated = now

        # Routing verso Web se c'è spazio, altrimenti Spike esistenti, altrimenti creane uno nuovo
        if len(self.web_server.jobs) < MAX_WEB_CAPACITY:
            self.web_server.arrival(job)
            return

        for server in self.spike_servers:
            if len(server.jobs) < MAX_SPIKE_CAPACITY:
                server.arrival(job)
                return

        # Se arrivo qui, tutti gli Spike sono saturi: aggiungo un nuovo Spike
        new_id = len(self.spike_servers)
        new_server = ProcessorSharingServer(self.env, f"Spike-{new_id}")
        self.spike_servers.append(new_server)
        new_server.arrival(job)
