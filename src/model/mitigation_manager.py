from library.rngs import random, selectStream
from engineering.costants import *
from engineering.distributions import get_service_time

from model.mitigation_center import MitigationCenter
from model.processor_sharing_server import ProcessorSharingServer

from model.ml_analysis_center import AnalysisCenter

class MitigationManager:
    def __init__(self, env, web_server, spike_servers, metrics, mode, variant):
        self.env = env
        self.mode = mode
        self.metrics = metrics
        self.variant = variant

        
        cap = MITIGATION_CAPACITY_VERIFICATION if self.mode == "verification" else MITIGATION_CAPACITY
        self.center = MitigationCenter(
            env,
            "MitigationCenter",
            capacity=cap,
            metrics=self.metrics
        )


        self.web_server = web_server
        self.spike_servers = spike_servers

        self.analysis_center = None
        if (self.variant == "ml_analysis") and (AnalysisCenter is not None):
            self.analysis_center = AnalysisCenter(
                env=self.env,
                name="AnalysisCenter",
                metrics=self.metrics,
                on_complete=self._after_analysis
            )

        self.metrics.setdefault("discarded_analysis_capacity", 0)
        self.metrics.setdefault("ml_drop_illegal", 0)  # TN
        self.metrics.setdefault("ml_drop_legal", 0)    # FN
        self.metrics.setdefault("ml_pass_illegal", 0)  # FP
        self.metrics.setdefault("ml_pass_legal", 0)    # TP

    def handle_job(self, job):
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

        selectStream(RNG_STREAM_FALSE_POSITIVE)
        if random() < P_FALSE_POSITIVE:
            self.metrics["false_positives"] = self.metrics.get("false_positives", 0) + 1
            if job.is_legal:
                self.metrics["false_positives_legal"] = self.metrics.get("false_positives_legal", 0) + 1
            return

        selectStream(RNG_STREAM_FEEDBACK)
        p_feedback = P_FEEDBACK_VERIFICATION if self.mode == "verification" else P_FEEDBACK
        if random() < p_feedback:
            job.arrival = now
            self.handle_job(job)
            return

        if self.analysis_center is not None:
            job.arrival = now  
            accepted = self.analysis_center.arrival(job)
            if not accepted:
                self.metrics["discarded_analysis_capacity"] += 1
            return  

        self._forward_to_service_tier(job, now)

    def _after_analysis(self, job, completion_time):
        selectStream(RNG_STREAM_ML_DECISION)
        u = random()

        if u < P_DROP_ML:
            if job.is_legal:
                self.metrics["ml_drop_legal"] += 1
            else:
                self.metrics["ml_drop_illegal"] += 1
            return
        else:
            if job.is_legal:
                self.metrics["ml_pass_legal"] += 1
                self._forward_to_service_tier(job, completion_time)
                return
            else:
                self.metrics["ml_pass_illegal"] += 1
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