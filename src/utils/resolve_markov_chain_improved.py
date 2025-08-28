import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time


class DDoSMarkovChain:

    def __init__(self):
        self.lambda_arrival = 6.666666  
        self.mu_mitigation  = 909.0     
        self.mu_web   = 6.25            
        self.mu_spike = 6.25            

        self.c_ana        = 4                  
        self.mu_ana_core  = 407.4425           
        
        self.K_mitigation = 4
        self.K_analysis   = self.c_ana
        self.K_web        = 20
        self.K_spike      = 20

        self.p_fp       = 0.01

        self.p_drop_ml = 0.89504
        self.p_pass_ml = 1.0 - self.p_drop_ml

        self.states = []
        self.state_to_index = {}
        self._build_state_space()

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

    def _rate_mit_depart(self, i):
        return self.mu_mitigation if i > 0 else 0.0

    def _rate_ana_depart(self, a):
        return self.mu_ana_core * min(a, self.c_ana)

    def _rate_web_depart(self, w):
        return self.mu_web if w > 0 else 0.0

    def _rate_spi_depart(self, s):
        return self.mu_spike if s > 0 else 0.0

    def _get_transition_rate(self, st_from, st_to):
        i, a, w, s = st_from
        i2, a2, w2, s2 = st_to

        if (i2 == i + 1) and (a2 == a) and (w2 == w) and (s2 == s):
            if i < self.K_mitigation:
                return self.lambda_arrival
            return 0.0

        if (i2 == i - 1) and (a2 in (a, a + 1)) and (w2 == w) and (s2 == s) and (i > 0):
            dep = self._rate_mit_depart(i)
            if dep == 0.0:
                return 0.0

            if a2 == a:
                return dep * self.p_fp

            if (a2 == a + 1) and (a < self.K_analysis):
                return dep * (1.0 - self.p_fp)

            if (a2 == a) and (a == self.K_analysis):
                return dep * (1.0 - self.p_fp)

            return 0.0

        if (i2 == i) and (a2 in (a - 1, a)) and (w2 in (w, w + 1)) and (s2 in (s, s + 1)) and (a > 0):
            dep = self._rate_ana_depart(a)
            if dep == 0.0:
                return 0.0

            if (a2 == a - 1) and (w2 == w) and (s2 == s):
                return dep * self.p_drop_ml

            fwd = dep * self.p_pass_ml

            if (a2 == a - 1) and (w2 == w + 1) and (s2 == s) and (w < self.K_web):
                return fwd

            if (a2 == a - 1) and (w2 == w) and (s2 == s + 1) and (w >= self.K_web) and (s < self.K_spike):
                return fwd

            if (a2 == a - 1) and (w2 == w) and (s2 == s) and (w >= self.K_web) and (s >= self.K_spike):
                return fwd

            return 0.0

        if (i2 == i) and (a2 == a) and (w2 == w - 1) and (s2 == s) and (w > 0):
            return self._rate_web_depart(w)

        if (i2 == i) and (a2 == a) and (w2 == w) and (s2 == s - 1) and (s > 0):
            return self._rate_spi_depart(s)

        return 0.0

    def build_generator_matrix(self):
        n = len(self.states)
        print(f"Costruzione matrice generatrice Q ({n}×{n})...")
        Q = sp.lil_matrix((n, n))
        for i, st_from in enumerate(self.states):
            if i % 1000 == 0:
                print(f"  Stato {i}/{n}")
            row_sum = 0.0
            for j, st_to in enumerate(self.states):
                if i == j:
                    continue
                r = self._get_transition_rate(st_from, st_to)
                if r > 0:
                    Q[i, j] = r
                    row_sum += r
            Q[i, i] = -row_sum
        return Q.tocsr()

    def solve_steady_state(self, Q):
        print("Risoluzione sistema lineare per probabilità stazionarie...")
        n = Q.shape[0]
        A = Q.T.tolil()
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        pi = spsolve(A.tocsr(), b)
        pi = np.maximum(pi, 0)
        pi /= np.sum(pi)
        return pi

    def calculate_metrics(self, pi):
        print("Calcolo metriche di performance...")

        util_mit = util_web = util_spi = 0.0
        avg_i = avg_a = avg_w = avg_s = 0.0

        thr_mit = thr_ana = thr_web = thr_spi = 0.0

        p_mit_full = 0.0
        p_ana_full = 0.0
        p_ana_nonempty = 0.0          
        e_busy_ana = 0.0              

        rate_mit_to_ana   = 0.0
        rate_mit_drop_fp  = 0.0
        rate_mit_drop_cap = 0.0

        rate_ana_drop_ml   = 0.0
        rate_ana_to_web    = 0.0
        rate_ana_to_spike  = 0.0
        rate_ana_drop_full = 0.0

        for idx, (i, a, w, s) in enumerate(self.states):
            p = pi[idx]

            if i == self.K_mitigation: p_mit_full += p
            if a == self.K_analysis:   p_ana_full += p
            if a > 0:                  p_ana_nonempty += p

            if i > 0: util_mit += p
            if w > 0: util_web += p
            if s > 0: util_spi += p

            e_busy_ana += p * min(a, self.c_ana)

            avg_i += p * i
            avg_a += p * a
            avg_w += p * w
            avg_s += p * s

            if i > 0: thr_mit += p * self.mu_mitigation
            if w > 0: thr_web += p * self.mu_web
            if s > 0: thr_spi += p * self.mu_spike
            thr_ana += p * (self.mu_ana_core * min(a, self.c_ana))

            if i > 0:
                dep = p * self.mu_mitigation
                rate_mit_drop_fp  += dep * self.p_fp
                if a < self.K_analysis:
                    rate_mit_to_ana += dep * (1.0 - self.p_fp)
                else:
                    rate_mit_drop_cap += dep * (1.0 - self.p_fp)

            if a > 0:
                dep_ana = p * (self.mu_ana_core * min(a, self.c_ana))
                rate_ana_drop_ml += dep_ana * self.p_drop_ml
                fwd = dep_ana * self.p_pass_ml
                if w < self.K_web:
                    rate_ana_to_web += fwd
                elif s < self.K_spike:
                    rate_ana_to_spike += fwd
                else:
                    rate_ana_drop_full += fwd

        util_ana_per_core = e_busy_ana / (self.c_ana)        
        thr_ana_expected = self.mu_ana_core * e_busy_ana

        rt_mit = (avg_i / thr_mit) if thr_mit > 0 else 0.0
        rt_ana = (avg_a / thr_ana) if thr_ana > 0 else 0.0
        rt_web = (avg_w / thr_web) if thr_web > 0 else 0.0
        rt_spi = (avg_s / thr_spi) if thr_spi > 0 else 0.0

        lambda_eff = self.lambda_arrival * (1.0 - p_mit_full)

        mit_dep = thr_mit
        ana_dep = thr_ana

        p_mit_to_ana   = rate_mit_to_ana   / mit_dep if mit_dep > 0 else 0.0
        p_mit_drop_fp  = rate_mit_drop_fp  / mit_dep if mit_dep > 0 else 0.0
        p_mit_drop_cap = rate_mit_drop_cap / mit_dep if mit_dep > 0 else 0.0

        p_ana_to_web    = rate_ana_to_web    / ana_dep if ana_dep > 0 else 0.0
        p_ana_to_spike  = rate_ana_to_spike  / ana_dep if ana_dep > 0 else 0.0
        p_ana_drop_ml   = rate_ana_drop_ml   / ana_dep if ana_dep > 0 else 0.0
        p_ana_drop_full = rate_ana_drop_full / ana_dep if ana_dep > 0 else 0.0

        return {
            "utilizations": {
                "mitigation": util_mit,                   
                "analysis_per_core": util_ana_per_core,   
                "analysis_nonempty": p_ana_nonempty,      
                "web": util_web,                          
                "spike": util_spi,                        
                "analysis_avg_busy_cores": e_busy_ana     
            },
            "avg_jobs": {
                "mitigation": avg_i,
                "analysis":   avg_a,
                "web":        avg_w,
                "spike":      avg_s,
            },
            "throughput": {
                "mitigation": thr_mit,
                "analysis":   thr_ana,
                "web":        thr_web,
                "spike":      thr_spi,
                "analysis_expected": thr_ana_expected     
            },
            "response_times": {
                "mitigation": rt_mit,
                "analysis":   rt_ana,
                "web":        rt_web,
                "spike":      rt_spi,
            },
            "routing_probabilities": {
                "mitigation": {
                    "to_analysis": p_mit_to_ana,
                    "drop_fp":     p_mit_drop_fp,
                    "drop_ana_full": p_mit_drop_cap,
                },
                "analysis": {
                    "to_web":     p_ana_to_web,
                    "to_spike":   p_ana_to_spike,
                    "drop_ml":    p_ana_drop_ml,
                    "drop_full":  p_ana_drop_full,
                }
            },
            "rates": {
                "effective_arrival": lambda_eff,
                "drop_mit_fp":       rate_mit_drop_fp,
                "drop_ana_cap":      rate_mit_drop_cap,
                "drop_ml_class":     rate_ana_drop_ml,
                "drop_downstream":   rate_ana_drop_full,
            },
            "probabilities": {
                "mitigation_full": p_mit_full,
                "analysis_full":   p_ana_full,
            }
        }

    def print_report(self, m):
        print("\n" + "="*70)
        print("MODELLO ANALITICO – Variante MIGLIORATIVA (con Analysis Center ML)")
        print("="*70)

        print("\nUTILIZZAZIONI:")
        print(f"  Mitigation            : {m['utilizations']['mitigation']:.6f}  (P[I>0])")
        print(f"  Analysis (per-core)   : {m['utilizations']['analysis_per_core']:.6f}  (=E[min(A,c)]/c)")
        print(f"  Analysis P(A>0)       : {m['utilizations']['analysis_nonempty']:.6f}")
        print(f"  Analysis avg busy c.  : {m['utilizations']['analysis_avg_busy_cores']:.6f}  (=E[min(A,c)])")
        print(f"  Web                   : {m['utilizations']['web']:.6f}  (P[W>0])")
        print(f"  Spike                 : {m['utilizations']['spike']:.6f}  (P[S>0])")

        print("\nTHROUGHPUT (job/s):")
        print(f"  Mitigation   : {m['throughput']['mitigation']:.6f}")
        print(f"  Analysis     : {m['throughput']['analysis']:.6f}")
        print(f"  Web          : {m['throughput']['web']:.6f}")
        print(f"  Spike        : {m['throughput']['spike']:.6f}")

        print("\nTEMPI DI RISPOSTA (s):")
        print(f"  Mitigation   : {m['response_times']['mitigation']:.6f}")
        print(f"  Analysis     : {m['response_times']['analysis']:.6f}")
        print(f"  Web          : {m['response_times']['web']:.6f}")
        print(f"  Spike        : {m['response_times']['spike']:.6f}")

        print("\nPROBABILITÀ DI ROUTING (condizionate al completamento del centro):")
        r_m = m["routing_probabilities"]["mitigation"]
        r_a = m["routing_probabilities"]["analysis"]
        print("  Mitigation → ...")
        print(f"    to Analysis    : {r_m['to_analysis']:.6f} ({100*r_m['to_analysis']:.2f}%)")
        print(f"    drop (FP)      : {r_m['drop_fp']:.6f} ({100*r_m['drop_fp']:.2f}%)")
        print(f"    drop (AnaFull) : {r_m['drop_ana_full']:.6f} ({100*r_m['drop_ana_full']:.2f}%)")
        print("  Analysis → ...")
        print(f"    to Web         : {r_a['to_web']:.6f} ({100*r_a['to_web']:.2f}%)")
        print(f"    to Spike       : {r_a['to_spike']:.6f} ({100*r_a['to_spike']:.2f}%)")
        print(f"    drop (ML)      : {r_a['drop_ml']:.6f} ({100*r_a['drop_ml']:.2f}%)")
        print(f"    drop (Full)    : {r_a['drop_full']:.6f} ({100*r_a['drop_full']:.2f}%)")

        print("\nALTRE METRICHE:")
        print(f"  P(Mitigation pieno) : {m['probabilities']['mitigation_full']:.6f}")
        print(f"  P(Analysis pieno)   : {m['probabilities']['analysis_full']:.6f}")
        print(f"  λ_eff (ingresso)    : {m['rates']['effective_arrival']:.6f} job/s")
        print(f"  Drop @Mit(FP)       : {m['rates']['drop_mit_fp']:.6f} job/s")
        print(f"  Drop @Ana(cap)      : {m['rates']['drop_ana_cap']:.6f} job/s")
        print(f"  Drop @ML(class)     : {m['rates']['drop_ml_class']:.6f} job/s")
        print(f"  Drop @downstream    : {m['rates']['drop_downstream']:.6f} job/s")

        thr_ana = m['throughput']['analysis']
        util_core = m['utilizations']['analysis_per_core']
        thr_ana_chk = self.mu_ana_core * self.c_ana * util_core
        print("\nCHECK:")
        print(f"  Analysis thr  vs μ·c·util_core : {thr_ana:.6f}  vs  {thr_ana_chk:.6f}")
        if thr_ana > 0:
            avg_a = m['avg_jobs']['analysis']
            rt_ana = m['response_times']['analysis']
            print(f"  Little (Analysis): N≈λT → {avg_a:.6f} ≈ {thr_ana:.6f}·{rt_ana:.6f} = {thr_ana*rt_ana:.6f}")

def main():
    print("Avvio risoluzione catena di Markov – modello MIGLIORATIVO (con Analysis Center ML)...")
    t0 = time.time()
    mc = DDoSMarkovChain()
    Q = mc.build_generator_matrix()
    pi = mc.solve_steady_state(Q)
    metrics = mc.calculate_metrics(pi)
    mc.print_report(metrics)
    print(f"\nTempo di esecuzione: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
