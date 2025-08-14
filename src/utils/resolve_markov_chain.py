import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
from collections import defaultdict

class DDoSMarkovChain:
    def __init__(self):
        # Parametri del sistema
        self.lambda_arrival = 6.666666  # job/s - tasso di arrivo
        self.mu_mitigation = 909         # job/s - tasso servizio centro mitigazione
        self.mu_web = 6.25               # job/s - tasso servizio web server
        self.mu_spike = 6.25             # job/s - tasso servizio spike server

        # Capacità
        self.K_mitigation = 4          # Capacità centro mitigazione
        self.K_web = 20               # Capacità web server
        self.K_spike = 20             # Capacità spike server
        
        # Probabilità
        self.p_lecito = 0.10          # Probabilità pacchetto lecito
        self.p_illecito = 0.89        # Probabilità pacchetto illecito
        self.p_fp = 0.01              # Probabilità falso positivo
        
        # Spazio degli stati: (i, w, s)
        # i: jobs in mitigation (0-4), w: jobs in web (0-20), s: jobs in spike (0-20)
        self.states = []
        self.state_to_index = {}
        self._build_state_space()
        
    def _build_state_space(self):
        """Costruisce lo spazio degli stati"""
        index = 0
        for i in range(self.K_mitigation + 1):
            for w in range(self.K_web + 1):
                for s in range(self.K_spike + 1):
                    state = (i, w, s)
                    self.states.append(state)
                    self.state_to_index[state] = index
                    index += 1
        print(f"Spazio degli stati costruito: {len(self.states)} stati")
    
    def _get_transition_rate(self, from_state, to_state):
        """Calcola il tasso di transizione tra due stati"""
        i1, w1, s1 = from_state
        i2, w2, s2 = to_state
        
        rate = 0.0
        
        # 1. ARRIVI nel centro di mitigazione
        if i2 == i1 + 1 and w2 == w1 and s2 == s1:
            if i1 < self.K_mitigation:  # Centro non saturo
                rate += self.lambda_arrival
        
        # 2. COMPLETAMENTI nel centro di mitigazione
        if i2 == i1 - 1 and i1 > 0:
            # Il job può essere:
            # a) Scartato per falso positivo
            if w2 == w1 and s2 == s1:
                # Probabilità di essere scartato
                p_discard = self.p_lecito * self.p_fp + self.p_illecito * self.p_fp
                rate += self.mu_mitigation * p_discard
            
            # b) Inoltrato al web server (se non saturo)
            elif w2 == w1 + 1 and s2 == s1 and w1 < self.K_web:
                # Probabilità di essere inoltrato (non scartato)
                p_forward = 1 - (self.p_lecito * self.p_fp + self.p_illecito * self.p_fp)
                rate += self.mu_mitigation * p_forward
            
            # c) Inoltrato al spike server (se web saturo e spike non saturo)
            elif w2 == w1 and s2 == s1 + 1 and w1 >= self.K_web and s1 < self.K_spike:
                p_forward = 1 - (self.p_lecito * self.p_fp + self.p_illecito * self.p_fp)
                rate += self.mu_mitigation * p_forward
            
            # d) Scartato perché entrambi i server sono saturi
            elif w2 == w1 and s2 == s1 and w1 >= self.K_web and s1 >= self.K_spike:
                p_forward = 1 - (self.p_lecito * self.p_fp + self.p_illecito * self.p_fp)
                rate += self.mu_mitigation * p_forward
        
        # 3. COMPLETAMENTI nel web server (Processor Sharing)
        if i2 == i1 and w2 == w1 - 1 and s2 == s1 and w1 > 0:
            # In PS, ogni job riceve μ_web/w1 del tasso di servizio
            rate += self.mu_web
        
        # 4. COMPLETAMENTI nel spike server (Processor Sharing)
        if i2 == i1 and w2 == w1 and s2 == s1 - 1 and s1 > 0:
            # In PS, ogni job riceve μ_spike/s1 del tasso di servizio
            rate += self.mu_spike
        
        return rate
    
    def build_generator_matrix(self):
        """Costruisce la matrice generatrice Q"""
        n_states = len(self.states)
        print(f"Costruzione matrice generatrice {n_states}x{n_states}...")
        
        # Usa matrice sparsa per efficienza
        Q = sp.lil_matrix((n_states, n_states))
        
        for i, from_state in enumerate(self.states):
            if i % 500 == 0:
                print(f"Processando stato {i}/{n_states}")
            
            row_sum = 0.0
            for j, to_state in enumerate(self.states):
                if i != j:
                    rate = self._get_transition_rate(from_state, to_state)
                    if rate > 0:
                        Q[i, j] = rate
                        row_sum += rate
            
            # Elemento diagonale (negativo della somma della riga)
            Q[i, i] = -row_sum
        
        return Q.tocsr()
    
    def solve_steady_state(self, Q):
        """Risolve per le probabilità stazionarie"""
        print("Risoluzione sistema lineare per probabilità stazionarie...")
        
        n_states = len(self.states)
        
        # Sostituisce l'ultima equazione con la condizione di normalizzazione
        A = Q.T.tolil()
        A[-1, :] = 1.0
        
        # Vettore termini noti
        b = np.zeros(n_states)
        b[-1] = 1.0
        
        # Risolve il sistema
        pi = spsolve(A.tocsr(), b)
        
        # Verifica che le probabilità siano positive e somma a 1
        pi = np.maximum(pi, 0)  # Assicura non-negatività
        pi = pi / np.sum(pi)    # Normalizza
        
        return pi
    
    def calculate_metrics(self, pi):
        """Calcola le metriche di performance (incluso il Mitigation)."""
        print("Calcolo metriche di performance...")

        # Utilizzazioni
        util_mitigation = 0.0
        util_web = 0.0
        util_spike = 0.0

        # Numero medio di job
        avg_jobs_mitigation = 0.0
        avg_jobs_web = 0.0
        avg_jobs_spike = 0.0

        # Throughput (tasso di completamento, per PS è μ * P{n>0})
        throughput_mitigation = 0.0
        throughput_web = 0.0
        throughput_spike = 0.0

        # Probabilità di routing dal mitigation center (condizionate a una partenza dal centro)
        prob_route_to_web = 0.0
        prob_route_to_spike = 0.0
        prob_discarded_fp = 0.0
        prob_discarded_full = 0.0
        total_mitigation_departures = 0.0

        # Probabilità che il centro mitigazione sia pieno (per arrivi bloccati)
        prob_mitigation_full = 0.0

        for idx, (i, w, s) in enumerate(self.states):
            prob = pi[idx]

            # Mitigation pieno?
            if i == self.K_mitigation:
                prob_mitigation_full += prob

            # Utilizzazioni (P{n>0})
            if i > 0:
                util_mitigation += prob
            if w > 0:
                util_web += prob
            if s > 0:
                util_spike += prob

            # Numero medio di job
            avg_jobs_mitigation += prob * i
            avg_jobs_web += prob * w
            avg_jobs_spike += prob * s

            # Throughput istantaneo nei server (per PS è μ se n>0, altrimenti 0)
            if i > 0:
                throughput_mitigation += prob * self.mu_mitigation
            if w > 0:
                throughput_web += prob * self.mu_web
            if s > 0:
                throughput_spike += prob * self.mu_spike

            # Routing/Drop alla partenza dal Mitigation
            if i > 0:
                # tasso di partenza dal centro mitigazione nello stato (i,w,s)
                departure_rate = prob * self.mu_mitigation
                total_mitigation_departures += departure_rate

                # drop per falso positivo (vale sia per lecito che illecito)
                p_fp_discard = self.p_fp  # perché p_lecito+p_illecito≈1
                prob_discarded_fp += departure_rate * p_fp_discard

                # inoltro (non FP)
                p_forward = 1.0 - p_fp_discard
                if w < self.K_web:
                    prob_route_to_web += departure_rate * p_forward
                elif s < self.K_spike:
                    prob_route_to_spike += departure_rate * p_forward
                else:
                    prob_discarded_full += departure_rate * p_forward

        # Tasso di arrivo effettivo al sistema (blocchi se Mitigation è pieno)
        effective_arrival_rate = self.lambda_arrival * (1.0 - prob_mitigation_full)

        # Normalizza le probabilità di routing (condizionate alla partenza dal Mitigation)
        if total_mitigation_departures > 0:
            prob_route_to_web   /= total_mitigation_departures
            prob_route_to_spike /= total_mitigation_departures
            prob_discarded_fp   /= total_mitigation_departures
            prob_discarded_full /= total_mitigation_departures

        # Tempi di risposta (Little: T = N / λ_eff_out)
        resp_time_mitigation = (avg_jobs_mitigation / throughput_mitigation) if throughput_mitigation > 0 else 0.0
        resp_time_web        = (avg_jobs_web        / throughput_web)        if throughput_web        > 0 else 0.0
        resp_time_spike      = (avg_jobs_spike      / throughput_spike)      if throughput_spike      > 0 else 0.0

        # Tasso di job scartati per FP (tra le partenze dal Mitigation)
        jobs_discarded_fp_rate = throughput_mitigation * prob_discarded_fp if total_mitigation_departures > 0 else 0.0

        return {
            'utilizations': {
                'mitigation': util_mitigation,
                'web': util_web,
                'spike': util_spike
            },
            'avg_jobs': {
                'mitigation': avg_jobs_mitigation,
                'web': avg_jobs_web,
                'spike': avg_jobs_spike
            },
            'throughput': {
                'mitigation': throughput_mitigation,
                'web': throughput_web,
                'spike': throughput_spike
            },
            'response_times': {
                'mitigation': resp_time_mitigation,
                'web': resp_time_web,
                'spike': resp_time_spike
            },
            'routing_probabilities': {
                'to_web': prob_route_to_web,
                'to_spike': prob_route_to_spike,
                'discarded_fp': prob_discarded_fp,
                'discarded_full': prob_discarded_full
            },
            'rates': {
                'effective_arrival': effective_arrival_rate,
                'jobs_discarded_fp': jobs_discarded_fp_rate
            },
            'probabilities': {
                'mitigation_full': prob_mitigation_full
            }
        }

    def print_comparison(self, metrics, simulation_data):
        """Confronta i risultati analitici con la simulazione (ora include il Mitigation)."""
        print("\n" + "="*60)
        print("CONFRONTO RISULTATI: MODELLO ANALITICO")
        print("="*60)

        # Dati simulazione 
        sim_web_completions = 59215
        sim_spike_completions = 4728
        sim_fp_dropped = 685

        # UTILIZZAZIONI
        print(f"\nUTILIZZAZIONI:")
        print(f"Mitigation     - Analitico: {metrics['utilizations']['mitigation']:.6f}")
        print(f"Web Server     - Analitico: {metrics['utilizations']['web']:.6f}")
        print(f"Spike Server   - Analitico: {metrics['utilizations']['spike']:.6f}")

        # THROUGHPUT
        print(f"\nTHROUGHPUT (job/s):")
        print(f"Mitigation     - Analitico: {metrics['throughput']['mitigation']:.6f}")
        print(f"Web Server     - Analitico: {metrics['throughput']['web']:.6f}")
        print(f"Spike Server   - Analitico: {metrics['throughput']['spike']:.6f}")

        # TEMPI DI RISPOSTA
        print(f"\nTEMPI DI RISPOSTA (s):")
        print(f"Mitigation     - Analitico: {metrics['response_times']['mitigation']:.6f}")
        print(f"Web Server     - Analitico: {metrics['response_times']['web']:.6f}")
        print(f"Spike Server   - Analitico: {metrics['response_times']['spike']:.6f}")

        # ALTRE METRICHE
        print(f"\nALTRE METRICHE:")
        print(f"P(Mitigation pieno)    - Analitico: {metrics['probabilities']['mitigation_full']:.6f}")
        print(f"Tasso arrivi effettivi - Analitico: {metrics['rates']['effective_arrival']:.6f}")
        print(f"Job scartati per FP    - Analitico: {metrics['rates']['jobs_discarded_fp']:.6f}")

        # ROUTING DAL MITIGATION
        print(f"\nPROBABILITÀ DI ROUTING DAL MITIGATION CENTER (condizionate a una partenza):")
        print(f"→ Web Server          - {metrics['routing_probabilities']['to_web']:.6f} ({metrics['routing_probabilities']['to_web']*100:.2f}%)")
        print(f"→ Spike Server        - {metrics['routing_probabilities']['to_spike']:.6f} ({metrics['routing_probabilities']['to_spike']*100:.2f}%)")
        print(f"→ Scartato (FP)       - {metrics['routing_probabilities']['discarded_fp']:.6f} ({metrics['routing_probabilities']['discarded_fp']*100:.2f}%)")
        print(f"→ Scartato (Full)     - {metrics['routing_probabilities']['discarded_full']:.6f} ({metrics['routing_probabilities']['discarded_full']*100:.2f}%)")

        # Confronto percentuali con la simulazione (grezzo, sui conteggi finali)
        sim_total_departures_from_mit = sim_web_completions + sim_spike_completions + sim_fp_dropped
        sim_pct_web = (sim_web_completions / sim_total_departures_from_mit) * 100 if sim_total_departures_from_mit > 0 else 0.0
        sim_pct_spike = (sim_spike_completions / sim_total_departures_from_mit) * 100 if sim_total_departures_from_mit > 0 else 0.0
        sim_pct_fp = (sim_fp_dropped / sim_total_departures_from_mit) * 100 if sim_total_departures_from_mit > 0 else 0.0

        print(f"\nCONFRONTO PERCENTUALI:")
        print(f"Jobs → Web Server     - {metrics['routing_probabilities']['to_web']*100:.2f}%")
        print(f"Jobs → Spike Server   - {metrics['routing_probabilities']['to_spike']*100:.2f}%")
        print(f"Jobs scartati (FP)    - {metrics['routing_probabilities']['discarded_fp']*100:.2f}%")

def main():
    print("Avvio risoluzione catena di Markov per DDoS Mitigation System...")
    start_time = time.time()
    
    
    markov = DDoSMarkovChain()
    
    # Costruisce la matrice generatrice
    Q = markov.build_generator_matrix()
    
    # Risolve per le probabilità stazionarie
    pi = markov.solve_steady_state(Q)
    
    # Calcola le metriche
    metrics = markov.calculate_metrics(pi)
    
    # Stampa i risultati e confronto
    markov.print_comparison(metrics, None)
    
    end_time = time.time()
    print(f"\nTempo di esecuzione: {end_time - start_time:.2f} secondi")

if __name__ == "__main__":
    main()