import simpy
from library.rvgs import Exponential
from engineering.costants import MITIGATION_MEAN
from library.rngs import selectStream
from engineering.costants import RNG_STREAM_MITIGATION_SERVICE


class MitigationCenter:
    def __init__(self, env, name, capacity, metrics=None):
        self.env = env
        self.name = name
        self.capacity = capacity

        # Coda con capacità (controllo capacità fatto dal manager prima dell'arrivo)
        self.queue = simpy.Store(env, capacity=capacity)

        # Stato/metriche
        self.busy = False
        self.last_start = None

        self.busy_time = 0.0
        self.busy_periods = []          # [(start, end), ...] per utilizzo a finestre

        self.completed_jobs = []        # response time per job nel centro
        self.completion_times = []      # tempi di completamento (monotoni) per throughput a finestre

        self.total_completions = 0
        self.legal_completions = 0
        self.illegal_completions = 0

        # Dizionario metriche condiviso (per aggiornare mitigation_completions al completamento reale)
        self.metrics = metrics

        # Avvia il processo server
        self.env.process(self.server_process())

    def has_capacity(self):
        # NB: il manager chiama questo PRIMA di fare arrival()
        return len(self.queue.items) < self.capacity

    def arrival(self, job):
        # Inserisce in coda (capacità già verificata a monte)
        self.queue.put(job)

    def server_process(self):
        while True:
            job = yield self.queue.get()

            # Inizio servizio
            self.busy = True
            self.last_start = self.env.now

            selectStream(RNG_STREAM_MITIGATION_SERVICE)
            # Tempo di servizio del centro (esponenziale con media MITIGATION_MEAN)
            yield self.env.timeout(Exponential(MITIGATION_MEAN))

            # Fine servizio
            now = self.env.now
            # aggiorna tempi occupati e periodo
            self.busy_time += now - self.last_start
            self.busy_periods.append((self.last_start, now))

            # response time nel centro (tempo dalla "arrival" corrente del job)
            self.completed_jobs.append(now - job.arrival)
            self.completion_times.append(now)

            # contatori
            self.total_completions += 1
            if job.is_legal:
                self.legal_completions += 1
            else:
                self.illegal_completions += 1

            # aggiorna metrica globale su dict condiviso
            if self.metrics is not None:
                self.metrics["mitigation_completions"] = self.metrics.get("mitigation_completions", 0) + 1

            # chiudi stato busy
            self.busy = False
            self.last_start = None

    def update(self, now):
        """
        Da chiamare a fine simulazione per chiudere eventuale periodo occupato ancora aperto
        e contabilizzarlo in busy_time/busy_periods.
        """
        if self.busy and self.last_start is not None:
            self.busy_time += now - self.last_start
            self.busy_periods.append((self.last_start, now))
            self.last_start = now  # allinea