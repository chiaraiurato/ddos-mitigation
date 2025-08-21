import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time


class DDoSMarkovChain:
    """
    Modello analitico (CTMC) del sistema DDoS con:
      - Mitigation (M/M/1/K)
      - Analysis ML (M/M/c/c, c=4, NO queue)
      - Web (Processor Sharing)
      - Spike (Processor Sharing)
    Stato: (i, a, w, s)
      i = job in Mitigation           ∈ {0..K_mitigation}
      a = job in Analysis (occupati)  ∈ {0..K_analysis}  (= c)
      w = job nel Web                 ∈ {0..K_web}
      s = job nello Spike             ∈ {0..K_spike}
    """

    def __init__(self):
        # ====== Tassi di arrivo e servizio ======
        # Arrivi globali
        self.lambda_arrival = 6.666666  # job/s

        # Mitigation
        self.mu_mitigation = 909.0      # job/s  (centro veloce)

        # Analysis ML (c=4 core senza coda): per-core
        self.c_analysis = 4
        self.mu_analysis_core = 407.4425  # job/s per core (1629.77 / 4)

        # Web & Spike (Processor Sharing)
        self.mu_web = 6.25
        self.mu_spike = 6.25

        # ====== Capacità ======
        self.K_mitigation = 4           # M/M/1/K (buffer K, o server con capacità K?)
        self.K_analysis = self.c_analysis  # M/M/c/c (blocking)
        self.K_web = 20
        self.K_spike = 20

        # ====== Probabilità di classi ======
        # mix di traffico
        self.p_lecito = 0.10
        self.p_illecito = 0.90

        # False positive in Mitigation (drop prima di Analysis)
        self.p_fp = 0.01

        # Classificatore ML al centro Analysis
        self.p_tpr = 0.9938  # Pr(LECITO→LECITO) True Positive
        self.p_tnr = 0.9938  # Pr(ILLECITO→ILLECITO) True Negative

        # Derivate utili per Analysis
        # Drop per ML = FN (leciti) + TN (illeciti)
        self.p_drop_ml = self.p_lecito * (1.0 - self.p_tpr) + self.p_illecito * self.p_tnr
        # Passano (verso Web/Spike) = TP (leciti) + FP (illeciti)
        self.p_pass_ml = 1.0 - self.p_drop_ml

        # ====== Spazio degli stati ======
        self.states = []
        self.state_to_index = {}
        self._build_state_space()

    # ------------------------------------------------------------------ #
    # Costruzione spazio degli stati
    # ------------------------------------------------------------------ #
    def _build_state_space(self):
        idx = 0
        for i in range(self.K_mitigation + 1):
            for a in range(self.K_analysis + 1):
                for w in range(self.K_web + 1):
                    for s in range(self.K_spike + 1):
                        st = (i, a, w, s)
                        self.states.append(st)
                        self.state_to_index[st] = idx
                        idx += 1
        print(f"Spazio degli stati costruito: {len(self.states)} stati (dim: "
              f"{self.K_mitigation+1}×{self.K_analysis+1}×{self.K_web+1}×{self.K_spike+1})")

    # ------------------------------------------------------------------ #
    # Tasso di transizione tra stati
    # ------------------------------------------------------------------ #
    def _get_transition_rate(self, from_state, to_state):
        i1, a1, w1, s1 = from_state
        i2, a2, w2, s2 = to_state

        rate = 0.0

        # ----------------- (1) ARRIVI al Mitigation -----------------
        # (i, a, w, s) -> (i+1, a, w, s) se non pieno
        if i2 == i1 + 1 and a2 == a1 and w2 == w1 and s2 == s1:
            if i1 < self.K_mitigation:
                rate += self.lambda_arrival

        # ----------------- (2) COMPLETAMENTI al Mitigation -----------------
        # (i, a, w, s) -> (i-1, a, w, s) [drop FP]
        if i1 > 0 and i2 == i1 - 1 and a2 == a1 and w2 == w1 and s2 == s1:
            rate += self.mu_mitigation * self.p_fp

        # (i, a, w, s) -> (i-1, a+1, w, s) [inoltro verso Analysis se non pieno]
        if i1 > 0 and i2 == i1 - 1 and a2 == a1 + 1 and w2 == w1 and s2 == s1:
            if a1 < self.K_analysis:
                rate += self.mu_mitigation * (1.0 - self.p_fp)

        # (i, a, w, s) -> (i-1, a, w, s) [inoltro fallito per Analysis pieno → drop]
        if i1 > 0 and i2 == i1 - 1 and a2 == a1 and w2 == w1 and s2 == s1:
            if a1 >= self.K_analysis:
                rate += self.mu_mitigation * (1.0 - self.p_fp)

        # ----------------- (3) COMPLETAMENTI all'Analysis -----------------
        # tasso = min(a, c) * mu_core
        if a1 > 0:
            ana_rate = min(a1, self.c_analysis) * self.mu_analysis_core

            # (i, a, w, s) -> (i, a-1, w, s) [drop per ML]
            if i2 == i1 and a2 == a1 - 1 and w2 == w1 and s2 == s1:
                rate += ana_rate * self.p_drop_ml

            # (i, a, w, s) -> (i, a-1, w+1, s) [passa a Web se non pieno]
            if i2 == i1 and a2 == a1 - 1 and w2 == w1 + 1 and s2 == s1 and w1 < self.K_web:
                rate += ana_rate * self.p_pass_ml

            # (i, a, w, s) -> (i, a-1, w, s+1) [altrimenti a Spike se Web pieno e Spike non pieno]
            if i2 == i1 and a2 == a1 - 1 and w2 == w1 and s2 == s1 + 1:
                if w1 >= self.K_web and s1 < self.K_spike:
                    rate += ana_rate * self.p_pass_ml

            # (i, a, w, s) -> (i, a-1, w, s) [drop per full downstream]
            if i2 == i1 and a2 == a1 - 1 and w2 == w1 and s2 == s1:
                if w1 >= self.K_web and s1 >= self.K_spike:
                    rate += ana_rate * self.p_pass_ml

        # ----------------- (4) COMPLETAMENTI Web (PS) -----------------
        # transizione: (i, a, w, s) -> (i, a, w-1, s)
        if w1 > 0 and i2 == i1 and a2 == a1 and w2 == w1 - 1 and s2 == s1:
            # In PS somma dei tassi = mu_web quando w1>0
            rate += self.mu_web

        # ----------------- (5) COMPLETAMENTI Spike (PS) -----------------
        # transizione: (i, a, w, s) -> (i, a, w, s-1)
        if s1 > 0 and i2 == i1 and a2 == a1 and w2 == w1 and s2 == s1 - 1:
            rate += self.mu_spike

        return rate

    # ------------------------------------------------------------------ #
    # Matrice generatrice
    # ------------------------------------------------------------------ #
    def build_generator_matrix(self):
        n = len(self.states)
        print(f"Costruzione matrice generatrice Q ({n}×{n})...")
        Q = sp.lil_matrix((n, n))

        for i, from_state in enumerate(self.states):
            if i % 1000 == 0:
                print(f"  Stato {i}/{n}")
            row_sum = 0.0
            for j, to_state in enumerate(self.states):
                if i == j:
                    continue
                r = self._get_transition_rate(from_state, to_state)
                if r > 0.0:
                    Q[i, j] = r
                    row_sum += r
            Q[i, i] = -row_sum

        return Q.tocsr()

    # ------------------------------------------------------------------ #
    # Stazionarie
    # ------------------------------------------------------------------ #
    def solve_steady_state(self, Q):
        print("Risoluzione sistema lineare per probabilità stazionarie...")
        n = Q.shape[0]
        A = Q.T.tolil()
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        pi = spsolve(A.tocsr(), b)
        pi = np.maximum(pi, 0)
        pi = pi / np.sum(pi)
        return pi

    # ------------------------------------------------------------------ #
    # Metriche
    # ------------------------------------------------------------------ #
    def calculate_metrics(self, pi):
        print("Calcolo metriche di performance...")

        U = {'mitigation': 0.0, 'analysis': 0.0, 'web': 0.0, 'spike': 0.0}
        N = {'mitigation': 0.0, 'analysis': 0.0, 'web': 0.0, 'spike': 0.0}
        TH = {'mitigation': 0.0, 'analysis': 0.0, 'web': 0.0, 'spike': 0.0}

        # Probabilità di centro pieno
        P_full = {'mitigation': 0.0, 'analysis': 0.0}

        # Routing/Drop ai completamenti
        mit_depart = 0.0
        mit_to_analysis = 0.0
        mit_drop_fp = 0.0
        mit_drop_anafull = 0.0

        ana_depart = 0.0
        ana_to_web = 0.0
        ana_to_spike = 0.0
        ana_drop_ml = 0.0
        ana_drop_full = 0.0

        for idx, (i, a, w, s) in enumerate(self.states):
            p = pi[idx]

            # pieni?
            if i == self.K_mitigation:
                P_full['mitigation'] += p
            if a == self.K_analysis:
                P_full['analysis'] += p

            # "utilization"
            if i > 0:
                U['mitigation'] += p
            # analysis: frazione di core occupati media = E[min(a,c)]/c
            U['analysis'] += (min(a, self.c_analysis) / self.c_analysis) * p
            if w > 0:
                U['web'] += p
            if s > 0:
                U['spike'] += p

            # N medi
            N['mitigation'] += i * p
            N['analysis'] += a * p
            N['web'] += w * p
            N['spike'] += s * p

            # Throughput
            if i > 0:
                TH['mitigation'] += self.mu_mitigation * p
                # routing a valle di Mitigation
                mit_depart += self.mu_mitigation * p
                mit_drop_fp += self.mu_mitigation * p * self.p_fp
                if a < self.K_analysis:
                    mit_to_analysis += self.mu_mitigation * p * (1.0 - self.p_fp)
                else:
                    mit_drop_anafull += self.mu_mitigation * p * (1.0 - self.p_fp)

            if a > 0:
                ana_rate = min(a, self.c_analysis) * self.mu_analysis_core
                TH['analysis'] += ana_rate
                # routing da Analysis
                ana_depart += ana_rate
                ana_drop_ml += ana_rate * self.p_drop_ml
                # forward (se possibile)
                if self.p_pass_ml > 0:
                    if w < self.K_web:
                        ana_to_web += ana_rate * self.p_pass_ml
                    elif s < self.K_spike:
                        ana_to_spike += ana_rate * self.p_pass_ml
                    else:
                        ana_drop_full += ana_rate * self.p_pass_ml

            if w > 0:
                TH['web'] += self.mu_web * p
            if s > 0:
                TH['spike'] += self.mu_spike * p

        # Response time (Little)
        eps = 1e-15
        RT = {
            'mitigation': (N['mitigation'] / max(TH['mitigation'], eps)) if TH['mitigation'] > 0 else 0.0,
            'analysis':   (N['analysis']   / max(TH['analysis'],   eps)) if TH['analysis']   > 0 else 0.0,
            'web':        (N['web']        / max(TH['web'],        eps)) if TH['web']        > 0 else 0.0,
            'spike':      (N['spike']      / max(TH['spike'],      eps)) if TH['spike']      > 0 else 0.0,
        }

        # Routing prob condizionate
        mit_route = {
            'to_analysis': (mit_to_analysis / mit_depart) if mit_depart > 0 else 0.0,
            'drop_fp':     (mit_drop_fp     / mit_depart) if mit_depart > 0 else 0.0,
            'drop_anafull':(mit_drop_anafull/ mit_depart) if mit_depart > 0 else 0.0,
        }
        ana_route = {
            'to_web':   (ana_to_web   / ana_depart) if ana_depart > 0 else 0.0,
            'to_spike': (ana_to_spike / ana_depart) if ana_depart > 0 else 0.0,
            'drop_ml':  (ana_drop_ml  / ana_depart) if ana_depart > 0 else 0.0,
            'drop_full':(ana_drop_full/ ana_depart) if ana_depart > 0 else 0.0,
        }

        return {
            'utilizations': U,
            'avg_jobs': N,
            'throughput': TH,
            'response_times': RT,
            'prob_full': P_full,
            'routing_mit': mit_route,
            'routing_ana': ana_route,
            'rates': {
                'lambda_eff': self.lambda_arrival * (1.0 - P_full['mitigation']),
                'drop_mit_fp': mit_drop_fp,
                'drop_ana_cap': ana_drop_full,
                'drop_ml': ana_drop_ml,
                'drop_downstream': 0.0  # non servono altri drop qui
            }
        }

    # ------------------------------------------------------------------ #
    # Stampa risultati (con DIAGNOSTICA PS)
    # ------------------------------------------------------------------ #
    def print_results(self, metrics):
        U = metrics['utilizations']
        TH = metrics['throughput']
        RT = metrics['response_times']
        Pfull = metrics['prob_full']
        rM = metrics['routing_mit']
        rA = metrics['routing_ana']

        print("\n" + "=" * 70)
        print("MODELLO ANALITICO – Variante MIGLIORATIVA (con Analysis Center ML)")
        print("=" * 70)

        # UTILIZZAZIONI
        print("\nUTILIZZAZIONI:")
        print(f"  Mitigation   : {U['mitigation']:.6f}")
        print(f"  Analysis     : {U['analysis']:.6f}  (≈ E[min(a,c)]/c)")
        print(f"  Web          : {U['web']:.6f}")
        print(f"  Spike        : {U['spike']:.6f}")

        # THROUGHPUT
        print("\nTHROUGHPUT (job/s):")
        print(f"  Mitigation   : {TH['mitigation']:.6f}")
        print(f"  Analysis     : {TH['analysis']:.6f}")
        print(f"  Web          : {TH['web']:.6f}")
        # usa e-notation per grandezze piccole
        print(f"  Spike        : {TH['spike']:.3e}")

        # TEMPI DI RISPOSTA
        print("\nTEMPI DI RISPOSTA (s):")
        print(f"  Mitigation   : {RT['mitigation']:.6f}")
        print(f"  Analysis     : {RT['analysis']:.6f}")
        print(f"  Web          : {RT['web']:.6f}")
        if TH['spike'] < 1e-12:
            print(f"  Spike        : —  (throughput ~ 0)")
        else:
            print(f"  Spike        : {RT['spike']:.6f}")

        # ROUTING
        print("\nPROBABILITÀ DI ROUTING (condizionate al completamento del centro):")
        print("  Mitigation → ...")
        print(f"    to Analysis    : {rM['to_analysis']:.6f} ({rM['to_analysis']*100:.2f}%)")
        print(f"    drop (FP)      : {rM['drop_fp']:.6f} ({rM['drop_fp']*100:.2f}%)")
        print(f"    drop (AnaFull) : {rM['drop_anafull']:.6f} ({rM['drop_anafull']*100:.2f}%)")
        print("  Analysis → ...")
        print(f"    to Web         : {rA['to_web']:.6f} ({rA['to_web']*100:.2f}%)")
        print(f"    to Spike       : {rA['to_spike']:.6f} ({rA['to_spike']*100:.2f}%)")
        print(f"    drop (ML)      : {rA['drop_ml']:.6f} ({rA['drop_ml']*100:.2f}%)")
        print(f"    drop (Full)    : {rA['drop_full']:.6f} ({rA['drop_full']*100:.2f}%)")

        # ALTRE METRICHE
        print("\nALTRE METRICHE:")
        print(f"  P(Mitigation pieno) : {Pfull['mitigation']:.6f}")
        print(f"  P(Analysis pieno)   : {Pfull['analysis']:.6f}")
        print(f"  λ_eff (ingresso)    : {metrics['rates']['lambda_eff']:.6f} job/s")
        print(f"  Drop @Mit(FP)       : {metrics['rates']['drop_mit_fp']:.6f} job/s")
        print(f"  Drop @Ana(cap)      : {metrics['rates']['drop_ana_cap']:.6f} job/s")
        print(f"  Drop @ML(class)     : {metrics['rates']['drop_ml']:.6f} job/s")
        print(f"  Drop @downstream    : {metrics['rates']['drop_downstream']:.6f} job/s")

        # ------------ DIAGNOSTICA PS (condizionata al centro non vuoto) ------------
        eps = 1e-15
        P_web_busy = U['web']             # ≈ P{W>0}
        P_spk_busy = U['spike']           # ≈ P{S>0}
        # E[· | ·>0] = E[·] / P{·>0}
        E_W_cond = (metrics['avg_jobs']['web']   / max(P_web_busy, eps)) if P_web_busy > 0 else 0.0
        E_S_cond = (metrics['avg_jobs']['spike'] / max(P_spk_busy, eps)) if P_spk_busy > 0 else 0.0

        print("\nDIAGNOSTICA PS (condizionata a centro non vuoto):")
        print(f"  P(Web>0)   = {P_web_busy:.3e}")
        print(f"  E[W|W>0]   = {E_W_cond:.3f}  → job concorrenti quando Web è attivo")
        print(f"  Check RT_w ≈ E[W|W>0]/μ = {E_W_cond/self.mu_web if P_web_busy>0 else 0.0:.3f} s")
        print(f"  P(Spike>0) = {P_spk_busy:.3e}")
        print(f"  E[S|S>0]   = {E_S_cond:.3f}  → job concorrenti quando Spike è attivo")
        print(f"  Check RT_s ≈ E[S|S>0]/μ = {E_S_cond/self.mu_spike if P_spk_busy>0 else 0.0:.3f} s")


def main():
    print("Avvio risoluzione catena di Markov – modello MIGLIORATIVO (con Analysis Center ML)...")
    t0 = time.time()

    mc = DDoSMarkovChain()
    Q = mc.build_generator_matrix()
    pi = mc.solve_steady_state(Q)
    metrics = mc.calculate_metrics(pi)
    mc.print_results(metrics)

    print(f"\nTempo di esecuzione: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
