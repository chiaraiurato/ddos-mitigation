from library.rngs import random, selectStream
from engineering.costants import *
from engineering.distributions import get_service_time

from model.mitigation_center import MitigationCenter
from model.processor_sharing_server import ProcessorSharingServer

# Centro di Analisi (opzionale nella variante migliorativa)

from model.ml_analysis_center import AnalysisCenter

class MitigationManager:
    """
    Pipeline:
      Arrivi -> Mitigation (cap, FP, feedback)
              -> (se passa) AnalysisCenter ML (senza coda, multiserver)
                 -> decisione ML (TPR/TNR) drop/pass
                 -> routing Web/Spike (+ autoscaling)

    Nella variante 'baseline' si salta l'AnalysisCenter e si va diretti a Web/Spike.
    """

    def __init__(self, env, web_server, spike_servers, metrics, mode, variant):
        self.env = env
        self.mode = mode
        self.metrics = metrics
        self.variant = variant

        # Mitigation invariata
        cap = MITIGATION_CAPACITY_VERIFICATION if self.mode == "verification" else MITIGATION_CAPACITY
        self.center = MitigationCenter(
            env,
            "MitigationCenter",
            capacity=cap,
            metrics=self.metrics
        )


        self.web_server = web_server
        self.spike_servers = spike_servers

        # Analysis center solo se variante migliorativa
        self.analysis_center = None
        if (self.variant == "ml_analysis") and (AnalysisCenter is not None):
            self.analysis_center = AnalysisCenter(
                env=self.env,
                name="AnalysisCenter",
                metrics=self.metrics,
                on_complete=self._after_analysis
            )

        # contatori ML aggiuntivi
        self.metrics.setdefault("discarded_analysis_capacity", 0)
        self.metrics.setdefault("ml_drop_illicit", 0)  # TN
        self.metrics.setdefault("ml_drop_legal", 0)    # FN
        self.metrics.setdefault("ml_pass_illicit", 0)  # FP
        self.metrics.setdefault("ml_pass_legal", 0)    # TP

    # ------- Ingresso dal processo di arrival -------
    def handle_job(self, job):
        # Capienza Mitigation
        if not self.center.has_capacity():
            if "discarded_detail" not in self.metrics:
                self.metrics["discarded_detail"] = []
            self.metrics["discarded_detail"].append({
                "time": self.env.now,
                "job_id": job.id,
                "is_legal": job.is_legal
            })
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

        # Decisione Mitigation: false positive => drop
        selectStream(RNG_STREAM_FALSE_POSITIVE)
        if random() < P_FALSE_POSITIVE:
            self.metrics["false_positives"] = self.metrics.get("false_positives", 0) + 1
            if job.is_legal:
                self.metrics["false_positives_legal"] = self.metrics.get("false_positives_legal", 0) + 1
            return

        # Feedback (retry) verso Mitigation
        selectStream(RNG_STREAM_FEEDBACK)
        p_feedback = P_FEEDBACK_VERIFICATION if self.mode == "verification" else P_FEEDBACK
        if random() < p_feedback:
            job.arrival = now
            self.handle_job(job)
            return


        # --- Variante MIGLIORATIVA: manda al Centro di Analisi ML ---
        if self.analysis_center is not None:
            job.arrival = now  # arrivo locale al centro di analisi
            accepted = self.analysis_center.arrival(job)
            if not accepted:
                # drop per capacità del centro di analisi (no-queue)
                self.metrics["discarded_analysis_capacity"] += 1
            return  # proseguirà su _after_analysis al termine del servizio

        # --- BASELINE: instrada direttamente verso Web/Spike ---
        self._forward_to_service_tier(job, now)

    # ------- Callback dopo il servizio di Analisi -------
    def _after_analysis(self, job, completion_time):
        """
        Chiamata dall'AnalysisCenter quando termina il servizio.
        Decide (TPR/TNR) se scartare o inoltrare.
        """
        selectStream(RNG_STREAM_ML_DECISION)

        if job.is_legal:
            # LECITO → passa con probabilità TPR, altrimenti FN (drop)
            if random() < P_TPR_ML:
                self.metrics["ml_pass_legal"] += 1
                self._forward_to_service_tier(job, completion_time)
            else:
                self.metrics["ml_drop_legal"] += 1
            return
        else:
            # ILLECITO → scarta con probabilità TNR (TN), altrimenti FP (passa)
            if random() < P_TNR_ML:
                self.metrics["ml_drop_illicit"] += 1
            else:
                self.metrics["ml_pass_illicit"] += 1
                self._forward_to_service_tier(job, completion_time)
            return

    def _forward_to_service_tier(self, job, now):
        selectStream(RNG_STREAM_SERVICE_TIMES)
        service_time = get_service_time(self.mode)
        job.remaining = service_time
        job.original_service = service_time
        job.last_updated = now

        job.arrival = now

        if len(self.web_server.jobs) < MAX_WEB_CAPACITY:
            self.web_server.arrival(job)
            return

        for server in self.spike_servers:
            if len(server.jobs) < MAX_SPIKE_CAPACITY:
                server.arrival(job)
                return

        new_id = len(self.spike_servers)
        new_server = ProcessorSharingServer(self.env, f"Spike-{new_id}")
        self.spike_servers.append(new_server)
        new_server.arrival(job)
