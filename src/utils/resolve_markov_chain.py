import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
from collections import defaultdict

class DDoSMarkovChain:
    def __init__(self):
        
        self.lambda_arrival = 6.666666  
        self.mu_mitigation = 909        
        self.mu_web = 6.25              
        self.mu_spike = 6.25            

        self.K_mitigation = 4        
        self.K_web = 20              
        self.K_spike = 20            
        
        self.p_discard = 0.01       
        self.p_forward = 0.99       

        self.states = []
        self.state_to_index = {}
        self._build_state_space()
        
    def _build_state_space(self):
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
        i1, w1, s1 = from_state
        i2, w2, s2 = to_state
        
        rate = 0.0
        
        if i2 == i1 + 1 and w2 == w1 and s2 == s1:
            if i1 < self.K_mitigation:
                rate += self.lambda_arrival
        
        if i2 == i1 - 1 and i1 > 0:
            if w2 == w1 and s2 == s1:
                rate += self.mu_mitigation * self.p_discard
            
            elif w2 == w1 + 1 and s2 == s1 and w1 < self.K_web:
                rate += self.mu_mitigation * self.p_forward
            
            elif w2 == w1 and s2 == s1 + 1 and w1 >= self.K_web and s1 < self.K_spike:
                rate += self.mu_mitigation * self.p_forward
            
            elif w2 == w1 and s2 == s1 and w1 >= self.K_web and s1 >= self.K_spike:
                rate += self.mu_mitigation * self.p_forward
        
        if i2 == i1 and w2 == w1 - 1 and s2 == s1 and w1 > 0:
            rate += self.mu_web
        
        if i2 == i1 and w2 == w1 and s2 == s1 - 1 and s1 > 0:
            rate += self.mu_spike
        
        return rate
    
    def build_generator_matrix(self):
        n_states = len(self.states)
        print(f"Costruzione matrice generatrice {n_states}x{n_states}...")
        
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
            
            Q[i, i] = -row_sum
        
        return Q.tocsr()
    
    def solve_steady_state(self, Q):
        print("Risoluzione sistema lineare per probabilità stazionarie...")
        
        n_states = len(self.states)
        
        A = Q.T.tolil()
        A[-1, :] = 1.0
        
        b = np.zeros(n_states)
        b[-1] = 1.0
        
        pi = spsolve(A.tocsr(), b)
        
        
        pi = np.maximum(pi, 0)  
        pi = pi / np.sum(pi)    
        
        return pi
    
    def calculate_metrics(self, pi):
        print("Calcolo metriche di performance...")

        util_mitigation = 0.0
        util_web = 0.0
        util_spike = 0.0

        avg_jobs_mitigation = 0.0
        avg_jobs_web = 0.0
        avg_jobs_spike = 0.0

        throughput_mitigation = 0.0
        throughput_web = 0.0
        throughput_spike = 0.0

        prob_route_to_web = 0.0
        prob_route_to_spike = 0.0
        prob_discarded = 0.0
        prob_discarded_full = 0.0
        total_mitigation_departures = 0.0

        prob_mitigation_full = 0.0

        for idx, (i, w, s) in enumerate(self.states):
            prob = pi[idx]

            if i == self.K_mitigation:
                prob_mitigation_full += prob

            if i > 0:
                util_mitigation += prob
            if w > 0:
                util_web += prob
            if s > 0:
                util_spike += prob

            avg_jobs_mitigation += prob * i
            avg_jobs_web += prob * w
            avg_jobs_spike += prob * s

            if i > 0:
                throughput_mitigation += prob * self.mu_mitigation
            if w > 0:
                throughput_web += prob * self.mu_web
            if s > 0:
                throughput_spike += prob * self.mu_spike

            if i > 0:
                departure_rate = prob * self.mu_mitigation
                total_mitigation_departures += departure_rate

                prob_discarded += departure_rate * self.p_discard

                if w < self.K_web:
                    prob_route_to_web += departure_rate * self.p_forward
                elif s < self.K_spike:
                    prob_route_to_spike += departure_rate * self.p_forward
                else:
                    prob_discarded_full += departure_rate * self.p_forward

        effective_arrival_rate = self.lambda_arrival * (1.0 - prob_mitigation_full)

        if total_mitigation_departures > 0:
            prob_route_to_web   /= total_mitigation_departures
            prob_route_to_spike /= total_mitigation_departures
            prob_discarded   /= total_mitigation_departures
            prob_discarded_full /= total_mitigation_departures

        resp_time_mitigation = (avg_jobs_mitigation / throughput_mitigation) if throughput_mitigation > 0 else 0.0
        resp_time_web        = (avg_jobs_web        / throughput_web)        if throughput_web        > 0 else 0.0
        resp_time_spike      = (avg_jobs_spike      / throughput_spike)      if throughput_spike      > 0 else 0.0

        jobs_discarded_rate = throughput_mitigation * self.p_discard if total_mitigation_departures > 0 else 0.0

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
                'discarded': prob_discarded,
                'discarded_full': prob_discarded_full
            },
            'rates': {
                'effective_arrival': effective_arrival_rate,
                'jobs_discarded': jobs_discarded_rate
            },
            'probabilities': {
                'mitigation_full': prob_mitigation_full
            }
        }

    def print_comparison(self, metrics, simulation_data):
        print("\n" + "="*60)
        print("CONFRONTO RISULTATI: MODELLO ANALITICO")
        print("="*60)

        sim_web_completions = 59215
        sim_spike_completions = 4728
        sim_fp_dropped = 685

        print(f"\nUTILIZZAZIONI:")
        print(f"Mitigation     - Analitico: {metrics['utilizations']['mitigation']:.6f}")
        print(f"Web Server     - Analitico: {metrics['utilizations']['web']:.6f}")
        print(f"Spike Server   - Analitico: {metrics['utilizations']['spike']:.6f}")

        print(f"\nTHROUGHPUT (job/s):")
        print(f"Mitigation     - Analitico: {metrics['throughput']['mitigation']:.6f}")
        print(f"Web Server     - Analitico: {metrics['throughput']['web']:.6f}")
        print(f"Spike Server   - Analitico: {metrics['throughput']['spike']:.6f}")

        print(f"\nTEMPI DI RISPOSTA (s):")
        print(f"Mitigation     - Analitico: {metrics['response_times']['mitigation']:.6f}")
        print(f"Web Server     - Analitico: {metrics['response_times']['web']:.6f}")
        print(f"Spike Server   - Analitico: {metrics['response_times']['spike']:.6f}")

        print(f"\nALTRE METRICHE:")
        print(f"P(Mitigation pieno)    - Analitico: {metrics['probabilities']['mitigation_full']:.6f}")
        print(f"Tasso arrivi effettivi - Analitico: {metrics['rates']['effective_arrival']:.6f}")
        print(f"Job scartati poiché classificati illeciti    - Analitico: {metrics['rates']['jobs_discarded']:.6f}")

        print(f"\nPROBABILITÀ DI ROUTING DAL MITIGATION CENTER (condizionate a una partenza):")
        print(f"→ Web Server                 - {metrics['routing_probabilities']['to_web']:.6f} ({metrics['routing_probabilities']['to_web']*100:.2f}%)")
        print(f"→ Spike Server               - {metrics['routing_probabilities']['to_spike']:.6f} ({metrics['routing_probabilities']['to_spike']*100:.2f}%)")
        print(f"→ Rifiutato da entrambi      - {metrics['routing_probabilities']['discarded_full']:.6f} ({metrics['routing_probabilities']['discarded_full']*100:.2f}%)")

        sim_total_departures_from_mit = sim_web_completions + sim_spike_completions + sim_fp_dropped
        sim_pct_web = (sim_web_completions / sim_total_departures_from_mit) * 100 if sim_total_departures_from_mit > 0 else 0.0
        sim_pct_spike = (sim_spike_completions / sim_total_departures_from_mit) * 100 if sim_total_departures_from_mit > 0 else 0.0
        sim_pct_fp = (sim_fp_dropped / sim_total_departures_from_mit) * 100 if sim_total_departures_from_mit > 0 else 0.0

        print(f"\nCONFRONTO PERCENTUALI:")
        print(f"Jobs → Web Server     - {metrics['routing_probabilities']['to_web']*100:.2f}%")
        print(f"Jobs → Spike Server   - {metrics['routing_probabilities']['to_spike']*100:.2f}%")
        print(f"Jobs scartati        - {metrics['routing_probabilities']['discarded']*100:.2f}%")

def main():
    print("Avvio risoluzione catena di Markov per DDoS Mitigation System...")
    start_time = time.time()
        
    markov = DDoSMarkovChain()
    
    Q = markov.build_generator_matrix()
    
    pi = markov.solve_steady_state(Q)
    
    metrics = markov.calculate_metrics(pi)
    
    markov.print_comparison(metrics, None)
    
    end_time = time.time()
    print(f"\nTempo di esecuzione: {end_time - start_time:.2f} secondi")

if __name__ == "__main__":
    main()