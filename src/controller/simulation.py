import simpy
import numpy as np
import os

from library.rngs import random, plantSeeds, getSeed, selectStream
from engineering.costants import *
from engineering.distributions import get_interarrival_time
from utils.csv_writer import append_row, append_row_stable
from model.job import Job
from model.processor_sharing_server import ProcessorSharingServer
from model.mitigation_manager import MitigationManager
from utils.checkpoint import _rt_mean_upto, _utilization_upto, _throughput_upto
from engineering.statistics import batch_means, export_bm_series_to_wide_csv, print_autocorrelation, util_thr_per_batch, window_util_thr


try:
    STOP_TRANSITORY = STOP_CONDITION_TRANSITORY
except NameError:
    try:
        STOP_TRANSITORY = STOP_CONDITION_TRANSITORY
    except NameError:
        STOP_TRANSITORY = STOP_CONDITION_FINITE_SIMULATION

try:
    CHECKPOINT_TRANSITORY = CHECKPOINT_TIME_TRANSITORY
except NameError:
    try:
        CHECKPOINT_TRANSITORY = CHECKPOINT_TIME_TRANSITORY
    except NameError:
        CHECKPOINT_TRANSITORY = CHECKPOINT_TIME_FINITE_SIMULATION


class DDoSSystem:
    def __init__(self, env, mode, arrival_p, arrival_l1, arrival_l2, variant):
        self.env = env
        self.mode = mode
        self.variant = variant

        self.arrival_p = arrival_p
        self.arrival_l1 = arrival_l1
        self.arrival_l2 = arrival_l2

        self.web_server = ProcessorSharingServer(env, "Web")
        self.spike_servers = [ProcessorSharingServer(env, "Spike-0")]

        # Metriche globali
        self.metrics = {
            "mitigation_completions": 0,
            "total_arrivals": 0,
            "false_positives": 0,
            "legal_arrivals": 0,
            "illegal_arrivals": 0,
            "false_positives_legal": 0,
        }

        self.mitigation_manager = MitigationManager(
            env, self.web_server, self.spike_servers, self.metrics, self.mode, variant=self.variant
        )

        self.env.process(self.arrival_process(mode))

    # ---------------------- SNAPSHOT (orizzonte finito) ----------------------
    def snapshot(self, when, replica_id=None):
        # flush stati al tempo 'when'
        self.web_server.update(when)
        for s in self.spike_servers:
            s.update(when)
        center = self.mitigation_manager.center
        center.update(when)

        # (se presente) chiudi periods Analysis
        if self.mitigation_manager.analysis_center is not None:
            self.mitigation_manager.analysis_center.update(when)

        row = {
            "replica": int(replica_id) if replica_id is not None else None,
            "time": float(when),

            # Web
            "web_rt_mean": _rt_mean_upto(self.web_server.completed_jobs,
                                         self.web_server.completion_times, when),
            "web_util": _utilization_upto(self.web_server.busy_periods, when),
            "web_throughput": _throughput_upto(self.web_server.completion_times, when),

            # Mitigation
            "mit_rt_mean": _rt_mean_upto(center.completed_jobs, center.completion_times, when),
            "mit_util": _utilization_upto(center.busy_periods, when),
            "mit_throughput": _throughput_upto(center.completion_times, when),
        }

        # (facoltativo) Analysis nel CSV se necessario
        if self.mitigation_manager.analysis_center is not None:
            ac = self.mitigation_manager.analysis_center
            row["analysis_rt_mean"] = _rt_mean_upto(ac.completed_jobs, ac.completion_times, when)
            row["analysis_util"] = _utilization_upto(ac.busy_periods, when)
            row["analysis_throughput"] = _throughput_upto(ac.completion_times, when)

        # Contatori globali
        row.update({
            "arrivals_so_far": int(self.metrics["total_arrivals"]),
            "false_positives_so_far": int(self.metrics.get("false_positives", 0)),
            "mitigation_completions_so_far": int(self.metrics.get("mitigation_completions", 0)),
            "spikes_count": int(len(self.spike_servers)),
        })

        # Per-spike
        for i in range(len(self.spike_servers)):
            s = self.spike_servers[i]
            row[f"spike{i}_rt_mean"] = _rt_mean_upto(s.completed_jobs, s.completion_times, when)
            row[f"spike{i}_util"] = _utilization_upto(s.busy_periods, when)
            row[f"spike{i}_throughput"] = _throughput_upto(s.completion_times, when)

        # Padding colonne spike fino a MAX_SPIKE_NUMBER
        for i in range(len(self.spike_servers), MAX_SPIKE_NUMBER):
            row[f"spike{i}_rt_mean"] = None
            row[f"spike{i}_util"] = None
            row[f"spike{i}_throughput"] = None

        # Mix/quote fino a 'when'
        tot_arr = max(1, self.metrics["total_arrivals"])

        proc_leg = int(self.metrics.get("processed_legal", 0))
        proc_illeg = int(self.metrics.get("processed_illegal", 0))

        web_leg = int(self.web_server.legal_completions)
        web_illeg = int(self.web_server.illegal_completions)
        spike_leg = sum(int(s.legal_completions) for s in self.spike_servers)
        spike_illeg = sum(int(s.illegal_completions) for s in self.spike_servers)

        comp_leg = web_leg + spike_leg
        comp_illeg = web_illeg + spike_illeg
        comp_tot = max(1, comp_leg + comp_illeg)

        row["illegal_share"] = self.metrics["illegal_arrivals"] / tot_arr
        row["processed_legal_share"] = proc_leg / tot_arr
        row["processed_illegal_share"] = proc_illeg / tot_arr
        row["completed_legal_share"] = comp_leg / tot_arr
        row["completed_illegal_share"] = comp_illeg / tot_arr
        row["completed_legal_of_completed_share"] = comp_leg / comp_tot
        row["completed_illegal_of_completed_share"] = comp_illeg / comp_tot

        # RT medio globale (Web+Spike) fino a 'when'
        sys_rts = list(self.web_server.completed_jobs)
        sys_times = list(self.web_server.completion_times)
        for s in self.spike_servers:
            sys_rts.extend(s.completed_jobs)
            sys_times.extend(s.completion_times)
        row["system_rt_mean"] = _rt_mean_upto(sys_rts, sys_times, when)

        return row

    # ---------------------- ARRIVALS ----------------------
    def arrival_process(self, mode):
        if mode == "verification":
            arrivals = N_ARRIVALS_VERIFICATION
            interarrival_mode = "verification"
            stop_on_arrivals = True
        elif mode in ("transitory", "finite simulation"):
            interarrival_mode = "standard"
            stop_on_arrivals = False
        elif mode == "infinite simulazion":
            arrivals = N_ARRIVALS_BATCH_MEANS
            interarrival_mode = "standard"
            stop_on_arrivals = True
        else:  # "standard"
            arrivals = N_ARRIVALS
            interarrival_mode = "standard"
            stop_on_arrivals = True

        if not stop_on_arrivals:
            while True:
                interarrival_time = get_interarrival_time(interarrival_mode, self.arrival_p, self.arrival_l1, self.arrival_l2)
                yield self.env.timeout(interarrival_time)
                self.metrics["total_arrivals"] += 1
                arrival_time = self.env.now
                is_legal = (random() < P_LECITO)
                if is_legal: self.metrics["legal_arrivals"] += 1
                else:        self.metrics["illegal_arrivals"] += 1
                job = Job(self.metrics["total_arrivals"], arrival_time, None, is_legal)
                self.mitigation_manager.handle_job(job)
        else:
            while self.metrics["total_arrivals"] < arrivals:
                interarrival_time = get_interarrival_time(interarrival_mode, self.arrival_p, self.arrival_l1, self.arrival_l2)
                yield self.env.timeout(interarrival_time)
                self.metrics["total_arrivals"] += 1
                arrival_time = self.env.now
                is_legal = (random() < P_LECITO)
                if is_legal: self.metrics["legal_arrivals"] += 1
                else:        self.metrics["illegal_arrivals"] += 1
                job = Job(self.metrics["total_arrivals"], arrival_time, None, is_legal)
                self.mitigation_manager.handle_job(job)

    def report_windowing(self):
        now = self.env.now
        self.web_server.update(now)
        for s in self.spike_servers: s.update(now)
        center = self.mitigation_manager.center
        center.update(now)

        def close_open_busy_periods(periods, now_):
            if periods and periods[-1][1] is None:
                periods[-1][1] = now_
        close_open_busy_periods(self.web_server.busy_periods, now)
        for s in self.spike_servers:
            close_open_busy_periods(s.busy_periods, now)
        close_open_busy_periods(center.busy_periods, now)

        print("\n==== SIMULATION COMPLETE ====")
        print(f"Total time: {now:.4f}")
        print(f"Arrivals: {self.metrics['total_arrivals']}")
        print(f"Lecite: {self.metrics['legal_arrivals']}, Illecite: {self.metrics['illegal_arrivals']}")
        print(f"Mitigation completions: {self.metrics['mitigation_completions']}")
        print(f"False positives (dropped): {self.metrics['false_positives']}")
        print(f"  Di cui lecite: {self.metrics['false_positives_legal']}")

        discarded = self.metrics.get("discarded_detail", [])
        print(f"Total discarded jobs: {len(discarded)}")

        def stats(name, server):
            if getattr(server, "completed_jobs", None) and server.completed_jobs:
                avg_rt = np.mean(server.completed_jobs)
                print(f"{name} Completions: {server.total_completions}")
                print(f"  Lecite  : {server.legal_completions}")
                print(f"  Illecite: {server.illegal_completions}")
                print(f"{name} Avg Resp Time: {avg_rt:.6f}")
                print(f"{name} Utilization: {server.busy_time / now:.6f}")
                print(f"{name} Throughput: {server.total_completions / now:.6f}")
            else:
                print(f"{name} Completions: 0")

        def print_ci(label, data, batch_size):
            try:
                if len(data) < batch_size or len(data) // batch_size < 2:
                    raise ValueError(f"Solo {len(data)} campioni → insufficienti per almeno 2 batch.")
                mean, ci = batch_means(data, batch_size)
                print(f"{label}: {mean:.6f} ± {ci:.6f} (95% CI)")
            except Exception as e:
                print(f"{label}: errore - {e}")

        stats("Web", self.web_server)
        for i, server in enumerate(self.spike_servers):
            stats(f"Spike-{i}", server)
        stats("Mitigation", center)
        print(f"Mitigation Discarded : {self.metrics.get('discarded_mitigation', 0)}")

        # CI con finestre per finito/se richiesto
        print("\n======== INTERVALLI DI CONFIDENZA ========")
        print("\n-- Web Server --")
        print_ci("Response Time:                           ", self.web_server.completed_jobs, BATCH_SIZE)
        util_samples, thr_samples = window_util_thr(self.web_server.busy_periods, self.web_server.completion_times, TIME_WINDOW, now)
        print_ci("Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
        print_ci("Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

        print("\n-- Spike Server --")
        for i, server in enumerate(self.spike_servers):
            print(f"Spike-{i}:")
            print_ci("Response Time:                           ", server.completed_jobs, BATCH_SIZE)
            util_samples, thr_samples = window_util_thr(server.busy_periods, server.completion_times, TIME_WINDOW, now)
            print_ci("Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
            print_ci("Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

        print("\n-- Mitigation Center --")
        center = self.mitigation_manager.center
        print_ci("Response Time:                           ", center.completed_jobs, BATCH_SIZE)
        util_samples, thr_samples = window_util_thr(center.busy_periods, center.completion_times, TIME_WINDOW, now)
        print_ci("Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
        print_ci("Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

        print("\n==== END OF REPORT ====")

    def report_bm(self,
              B: int = None,
              K: int = None,
              confidence: float = CONFIDENCE_LEVEL,
              burn_in_rt: int = 0,
              include_system: bool = False):
        """
        Stampa:
        - CI (t) sui Response Time per centro con Batch Means classico (batch di B completamenti)
        - CI (t) su Utilization e Throughput per centro usando la serie per-batch
        Parametri:
        B:           batch size (default = BATCH_SIZE)
        K:           max num batch (default = N_BATCH) per i RT
        confidence:  livello di confidenza (es. 0.95)
        burn_in_rt:  # di completamenti da scartare prima di batchizzare gli RT
        include_system: se True stampa anche 'System RT' (Web+Spike)
        """
        # --- setup e chiusura periodi aperti
        B = B or BATCH_SIZE
        K = K or N_BATCH
        now = self.env.now

        def _close_open_busy_periods(periods, now_):
            if periods and periods[-1][1] is None:
                periods[-1][1] = now_
        self.web_server.update(now)
        for s in self.spike_servers: s.update(now)
        center = self.mitigation_manager.center
        center.update(now)
        _close_open_busy_periods(self.web_server.busy_periods, now)
        for s in self.spike_servers:
            _close_open_busy_periods(s.busy_periods, now)
        _close_open_busy_periods(center.busy_periods, now)

        # --- helper CI ---
        def _print_ci(label, samples, *, batch_size, n_batches, conf, burn_in=0):
            try:
                mean, hw = batch_means(
                    samples,
                    batch_size=batch_size,
                    n_batches=n_batches,
                    confidence=conf,
                    burn_in=burn_in
                )
                print(f"{label}: {mean:.6f} ± {hw:.6f} ({int(conf*100)}% CI)")
            except Exception as e:
                print(f"{label}: errore - {e}")

        print("\n==== SIMULATION COMPLETE ====")
        print(f"Total time: {now:.4f}")
        print(f"Arrivals: {self.metrics['total_arrivals']}")
        print(f"Lecite: {self.metrics['legal_arrivals']}, Illecite: {self.metrics['illegal_arrivals']}")
        print(f"Mitigation completions: {self.metrics['mitigation_completions']}")
        print(f"False positives (dropped): {self.metrics['false_positives']}")
        print(f"  Di cui lecite: {self.metrics['false_positives_legal']}")
        print(f"Total discarded jobs: {len(self.metrics.get('discarded_detail', []))}")

        # ====== 1) CI sui Response Time  ======
        print("\n======== INTERVALLI DI CONFIDENZA (Batch Means sui RT) ========")

        # Web
        _print_ci("Web RT",
                self.web_server.completed_jobs,
                batch_size=B, n_batches=K, conf=confidence, burn_in=burn_in_rt)

        # Spike-i
        for i, srv in enumerate(self.spike_servers):
            _print_ci(f"Spike-{i} RT",
                    srv.completed_jobs,
                    batch_size=B, n_batches=K, conf=confidence, burn_in=burn_in_rt)

        # Mitigation
        _print_ci("Mitigation RT",
                center.completed_jobs,
                batch_size=B, n_batches=K, conf=confidence, burn_in=burn_in_rt)

        # (Opzionale) System RT = Web + Spike
        # if include_system:
        #     sys_rts = list(self.web_server.completed_jobs)
        #     for s in self.spike_servers:
        #         sys_rts.extend(s.completed_jobs)
        #     _print_ci("System RT",
        #             sys_rts,
        #             batch_size=B, n_batches=K, conf=confidence, burn_in=burn_in_rt)

        # ====== 2) CI su Utilization e Throughput (serie per-batch) ======
        print("\n======== INTERVALLI DI CONFIDENZA (Utilization / Throughput per batch) ========")

        def _print_util_thr_ci_for(label_prefix, busy_periods, completion_times):
            # Costruisco la serie per-batch usando lo stesso B dei RT.
            util_series, thr_series = util_thr_per_batch(
                busy_periods,
                completion_times,
                B=B,
                burn_in=burn_in_rt,  
                k_max=None,
                tmax=now
            )
            # t-CI direttamente sulla serie per-batch
            _print_ci(f"{label_prefix} Utilization",
                    util_series, batch_size=1, n_batches=None, conf=confidence)
            _print_ci(f"{label_prefix} Throughput",
                    thr_series, batch_size=1, n_batches=None, conf=confidence)

        # Web
        _print_util_thr_ci_for("Web",
                            self.web_server.busy_periods,
                            self.web_server.completion_times)

        # Spike-i
        for i, srv in enumerate(self.spike_servers):
            _print_util_thr_ci_for(f"Spike-{i}",
                                srv.busy_periods,
                                srv.completion_times)

        # Mitigation
        _print_util_thr_ci_for("Mitigation",
                            center.busy_periods,
                            center.completion_times)

        print("\n==== END OF REPORT ====")

    # ---------------------- REPORT  ----------------------
    def report_single_run(self):
        now = self.env.now
        self.web_server.update(now)
        for s in self.spike_servers: s.update(now)
        center = self.mitigation_manager.center
        center.update(now)

        ac = self.mitigation_manager.analysis_center
        if ac is not None:
            ac.update(now)

        # fun per chiudere intervalli busy (Web/Spike/Mitigation hanno già busy_periods)
        def total_busy_time_of(obj):
            # AnalysisCenter: somma per-core
            if hasattr(obj, "core_busy_periods"):
                total = 0.0
                for periods in obj.core_busy_periods:
                    for (a, b) in periods:
                        if b is None:
                            b = now
                        if b > a:
                            total += (b - a)
                return total
            # Altri centri
            return getattr(obj, "busy_time", 0.0)

        print("\n==== SIMULATION COMPLETE ====")
        print(f"Total time: {now:.4f}")
        print(f"Arrivals: {self.metrics['total_arrivals']}")
        print(f"Lecite: {self.metrics['legal_arrivals']}, Illecite: {self.metrics['illegal_arrivals']}")
        print(f"Mitigation completions: {self.metrics['mitigation_completions']}")
        print(f"False positives (dropped): {self.metrics['false_positives']}")
        print(f"  Di cui lecite: {self.metrics['false_positives_legal']}")

        discarded = self.metrics.get("discarded_detail", [])
        print(f"Total discarded jobs: {len(discarded)}")

        def stats(name, server):
            # response time / throughput standard
            if getattr(server, "completed_jobs", None) and server.completed_jobs:
                avg_rt = np.mean(server.completed_jobs)
                busy_time = total_busy_time_of(server)
                print(f"{name} Completions: {server.total_completions}")
                if hasattr(server, "legal_completions"):
                    print(f"  Lecite  : {server.legal_completions}")
                    print(f"  Illecite: {server.illegal_completions}")
                print(f"{name} Avg Resp Time: {avg_rt:.6f}")
                print(f"{name} Utilization: {busy_time / now:.6f}")
                print(f"{name} Throughput: {server.total_completions / now:.6f}")
            else:
                print(f"{name} Completions: 0")

        stats("Web", self.web_server)
        for i, server in enumerate(self.spike_servers):
            stats(f"Spike-{i}", server)
        stats("Mitigation", center)

        if ac is not None:
            stats("Analysis", ac)
            print(f"Analysis capacity drops : {self.metrics.get('discarded_analysis_capacity', 0)}")
            print(f"ML  drop illicit (TN)   : {self.metrics.get('ml_drop_illicit', 0)}")
            print(f"ML  drop legal   (FN)   : {self.metrics.get('ml_drop_legal', 0)}")
            print(f"ML  pass illicit (FP)   : {self.metrics.get('ml_pass_illicit', 0)}")
            print(f"ML  pass legal   (TP)   : {self.metrics.get('ml_pass_legal', 0)}")

        print(f"Mitigation Discarded : {self.metrics.get('discarded_mitigation', 0)}")

    # (le altre funzioni report_windowing / report_bm le puoi aggiornare in seguito)
    # -------------------------------------------------------------------------


def run_simulation(scenario: str, mode: str, model: str, enable_windowing: bool, 
                   arrival_p=None, arrival_l1=None, arrival_l2=None):
    
    print("Model " + model)

    if mode not in ("verification", "standard"):
        raise ValueError("mode must be 'verification' or 'standard'")
    if arrival_p is None:
        arrival_p = ARRIVAL_P
        arrival_l1 = ARRIVAL_L1
        arrival_l2 = ARRIVAL_L2

    env = simpy.Environment()
    system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, variant=model)
    env.run()
    system.report_single_run()
    if(enable_windowing == True):
        system.report_windowing()
    # CSV: quando vorrai aggiungere le colonne del centro di analisi, estendiamo i fieldnames e l'append_row_stable

def transitory_fieldnames(max_spikes: int):
    base = [
        "replica", "time",
        "web_rt_mean", "web_util", "web_throughput",
        "mit_rt_mean", "mit_util", "mit_throughput",
        # NEW: colonne opzionali per Analysis Center (se attivo)
        "ana_rt_mean", "ana_util", "ana_throughput",
        "system_rt_mean",
        "arrivals_so_far", "false_positives_so_far", "mitigation_completions_so_far",
        "spikes_count", "scenario", "mode", "is_final",
        "illegal_share",
        "processed_legal_share", "processed_illegal_share",
        "completed_legal_share", "completed_illegal_share",
        "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
    ]
    for i in range(MAX_SPIKE_NUMBER):
        base += [f"spike{i}_rt_mean", f"spike{i}_util", f"spike{i}_throughput"]
    return base

def infinite_fieldnames(max_spikes: int):
    base = [
        "scenario",
        "ARRIVAL_P", "ARRIVAL_L1", "ARRIVAL_L2",
        "total_time", "total_arrivals",
        "burn_in_rt", "batch_size", "n_batches", "confidence",
        "web_rt_mean_bm", "web_rt_ci_hw",
        "web_util_point", "web_throughput_point",
        "mit_rt_mean_bm", "mit_rt_ci_hw",
        "mit_util_point", "mit_throughput_point",
        # TODO: aggiungere anche analysis_* quando estendiamo il BM per quel centro
        "system_rt_mean_bm", "system_rt_ci_hw",
        "illegal_share",
        "processed_legal_share", "processed_illegal_share",
        "completed_legal_share", "completed_illegal_share",
        "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
        "spikes_count",
    ]
    for i in range(MAX_SPIKE_NUMBER):
        base += [f"spike{i}_rt_mean_bm", f"spike{i}_rt_ci_hw",
                 f"spike{i}_util_point", f"spike{i}_thr_point",
                 f"spike{i}_completions"]
    return base

def run_finite_horizon(mode: str, scenario: str, out_csv: str):
    if os.path.exists(out_csv):
        os.remove(out_csv)

    arrival_p  = ARRIVAL_P
    arrival_l1 = ARRIVAL_L1_x40
    arrival_l2 = ARRIVAL_L2_x40

    fieldnames = transitory_fieldnames(MAX_SPIKE_NUMBER)
    all_logs = []

    if mode == "transitory":
        for rep in range(REPLICATION_FACTOR_TRANSITORY):
            seed = SEEDS_TRANSITORY[rep % len(SEEDS_TRANSITORY)]
            plantSeeds(seed)

            env = simpy.Environment()
            system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2)

            def checkpointer_optimized(env, system, rep_id):
                checkpoint_times = []
                t = CHECKPOINT_TIME_TRANSITORY
                while t <= STOP_CONDITION_TRANSITORY:
                    checkpoint_times.append(t)
                    t += CHECKPOINT_TIME_TRANSITORY

                for cp_time in checkpoint_times:
                    yield env.timeout(cp_time - env.now)
                    snap = system.snapshot(env.now, replica_id=rep_id)
                    snap["scenario"] = scenario
                    snap["mode"] = mode
                    snap["is_final"] = (cp_time == checkpoint_times[-1])
                    all_logs.append(snap)
                    append_row_stable(out_csv, snap, fieldnames)
                    progress = (cp_time / STOP_CONDITION_TRANSITORY) * 100
                    print(f"  Replica {rep+1}: {progress:.1f}% completato")

            env.process(checkpointer_optimized(env, system, rep))
            env.run(until=STOP_CONDITION_TRANSITORY)
            print(f"Replica {rep+1} completata!")

    elif mode == "finite simulation":
        seeds = [1234566789]
        for rep in range(REPLICATION_FACTORY_FINITE_SIMULATION):
            plantSeeds(seeds[rep])
            env = simpy.Environment()
            system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2)

            def checkpointer(env, system, rep_id):
                next_cp = CHECKPOINT_TIME_FINITE_SIMULATION
                while next_cp <= STOP_CONDITION_FINITE_SIMULATION:
                    yield env.timeout(next_cp - env.now)
                    snap = system.snapshot(env.now, replica_id=rep_id)
                    snap["scenario"] = scenario
                    snap["mode"] = mode
                    snap["is_final"] = False
                    all_logs.append(snap)
                    append_row_stable(out_csv, snap, fieldnames)
                    next_cp += CHECKPOINT_TIME_FINITE_SIMULATION

                if env.now < STOP_CONDITION_FINITE_SIMULATION:
                    yield env.timeout(STOP_CONDITION_FINITE_SIMULATION - env.now)
                    snap = system.snapshot(env.now, replica_id=rep_id)
                    snap["scenario"] = scenario
                    snap["mode"] = mode
                    snap["is_final"] = True
                    all_logs.append(snap)
                    append_row_stable(out_csv, snap, fieldnames)

            env.process(checkpointer(env, system, rep))
            env.run(until=STOP_CONDITION_FINITE_SIMULATION)

            selectStream(RNG_STREAM)
            seeds.append(getSeed())

    return all_logs


def run_infinite_horizon(mode: str,
                         out_csv: str,
                         out_acs: str,
                         burn_in: int = 0,
                         arrival_p=None, arrival_l1=None, arrival_l2=None):
    """
    Esegue la simulazione a orizzonte infinito tramite il metodo dei Batch Means,
    calcolando le autocorrelazioni (ACF) sulle metriche di interesse.
    """
    if mode not in ("verification", "standard"):
        raise ValueError("mode must be 'verification' or 'standard'")

    if arrival_p is None:
        arrival_p  = ARRIVAL_P
        arrival_l1 = ARRIVAL_L1_x40
        arrival_l2 = ARRIVAL_L2_x40

    env = simpy.Environment()
    system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2)
    env.run()
    # system.report_bm()  # opzionale: abbiamo la versione con Analysis sopra
    print("\n==== START BATCH MEANS ====")

    csv_path, cols, k = export_bm_series_to_wide_csv(
        system,
        B=BATCH_SIZE,
        out_csv=out_csv,
        burn_in=burn_in,
        k_max=None
    )
    print(f"[OK] bm_series in {csv_path} ({k} righe). Colonne: {cols}")

    res_df = print_autocorrelation(
        file_path=csv_path,
        columns=cols,
        K_LAG=50,
        threshold=0.2,
        save_csv=out_acs
    )
    print(f"[OK] ACF salvata in: {out_acs}")

    return res_df
