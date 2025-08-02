import simpy
import numpy as np
from tqdm import trange
from costants import *
from library.rngs import random
from library.rvgs import Exponential


def batch_means_integrated(num_batches=512, batch_time=1000):
    """
    Analisi Batch Means per il sistema integrato
    
    Il metodo batch means divide la simulazione in batch indipendenti per:
    1. Ridurre l'autocorrelazione tra osservazioni
    2. Fornire stime più accurate degli intervalli di confidenza
    3. Valutare la stabilità del sistema nel tempo
    """
    try:
        iterator = trange(num_batches, desc="Running integrated batches")
    except ImportError:
        iterator = range(num_batches)
    
    print(f"\n{'='*70}")
    print(f"BATCH MEANS ANALYSIS - {num_batches} batches of {batch_time} time units each")
    print(f"{'='*70}")
    
    # Collezioni per le metriche di ogni batch
    batch_metrics = {
        'single_resp': [],
        'web_resp': [],
        'spike_resp': [],
        'single_util': [],
        'web_util': [],
        'spike_util': [],
        'single_tput': [],
        'web_tput': [],
        'spike_tput': [],
        'total_tput': [],
        'overflow_rate': [],
        'avg_queue_single': [],
        'avg_queue_web': [],
        'avg_queue_spike': []
    }
    
    for batch_num in iterator:
        env = simpy.Environment()
        system = SystemSimulation(env)
        env.run(until=batch_time)
        
        # Raccogli metriche per questo batch
        total_time = env.now
        
        # Response times
        batch_metrics['single_resp'].append(
            np.mean(system.single_stats) if system.single_stats else 0.0
        )
        batch_metrics['web_resp'].append(
            np.mean(system.web_stats) if system.web_stats else 0.0
        )
        batch_metrics['spike_resp'].append(
            np.mean(system.spike_stats) if system.spike_stats else 0.0
        )
        
        # Utilizations
        batch_metrics['single_util'].append(system.busy_single / total_time)
        batch_metrics['web_util'].append(system.busy_web / total_time)
        batch_metrics['spike_util'].append(system.busy_spike / total_time)
        
        # Throughputs
        batch_metrics['single_tput'].append(system.single_completions / total_time)
        batch_metrics['web_tput'].append(system.web_completions / total_time)
        batch_metrics['spike_tput'].append(system.spike_completions / total_time)
        
        total_completions = (system.single_completions + 
                           system.web_completions + 
                           system.spike_completions)
        batch_metrics['total_tput'].append(total_completions / total_time)
        
        # Overflow rate
        batch_metrics['overflow_rate'].append(
            system.overflow_jobs / system.total_arrivals if system.total_arrivals > 0 else 0.0
        )
        
        # Average queue lengths
        batch_metrics['avg_queue_single'].append(system.area_single / total_time)
        batch_metrics['avg_queue_web'].append(system.area_web / total_time)
        batch_metrics['avg_queue_spike'].append(system.area_spike / total_time)
    
    def compute_batch_ci(data, metric_name, confidence_level=CONF_LEVEL):
        """Calcola intervallo di confidenza per batch means"""
        if len(data) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        se = std / np.sqrt(len(data))  # Standard error
        margin = confidence_level * se
        ci_low = mean - margin
        ci_high = mean + margin
        
        return mean, std, se, ci_low, ci_high
    
    def print_metric_results(metric_name, data, unit=""):
        """Stampa risultati dettagliati per una metrica"""
        mean, std, se, ci_low, ci_high = compute_batch_ci(data, metric_name)
        margin = mean - ci_low
        
        print(f"{metric_name:<25}: {mean:.6f}{unit} ± {margin:.6f}")
        print(f"{'  99% CI':<25}: [{ci_low:.6f}, {ci_high:.6f}]{unit}")
        print(f"{'  Std Dev':<25}: {std:.6f}{unit}")
        print(f"{'  Std Error':<25}: {se:.6f}{unit}")
        print()
    
    # Analisi e stampa risultati dettagliati
    print("\nRESPONSE TIMES:")
    print("-" * 50)
    print_metric_results("Single Center", batch_metrics['single_resp'], " time units")
    print_metric_results("Web Pool", batch_metrics['web_resp'], " time units")
    print_metric_results("Spike Pool", batch_metrics['spike_resp'], " time units")
    
    print("UTILIZATIONS:")
    print("-" * 50)
    print_metric_results("Single Center", batch_metrics['single_util'])
    print_metric_results("Web Pool", batch_metrics['web_util'])
    print_metric_results("Spike Pool", batch_metrics['spike_util'])
    
    print("THROUGHPUTS:")
    print("-" * 50)
    print_metric_results("Single Center", batch_metrics['single_tput'], " jobs/time")
    print_metric_results("Web Pool", batch_metrics['web_tput'], " jobs/time")
    print_metric_results("Spike Pool", batch_metrics['spike_tput'], " jobs/time")
    print_metric_results("Total System", batch_metrics['total_tput'], " jobs/time")
    
    print("QUEUE LENGTHS:")
    print("-" * 50)
    print_metric_results("Single Center Avg", batch_metrics['avg_queue_single'], " jobs")
    print_metric_results("Web Pool Avg", batch_metrics['avg_queue_web'], " jobs")
    print_metric_results("Spike Pool Avg", batch_metrics['avg_queue_spike'], " jobs")
    
    print("SYSTEM METRICS:")
    print("-" * 50)
    print_metric_results("Overflow Rate", batch_metrics['overflow_rate'])
    
    # Analisi di convergenza (opzionale)
    print("CONVERGENCE ANALYSIS:")
    print("-" * 50)
    
    # Calcola medie cumulative per verificare convergenza
    cumulative_means = np.cumsum(batch_metrics['total_tput']) / np.arange(1, len(batch_metrics['total_tput']) + 1)
    final_mean = cumulative_means[-1]
    convergence_threshold = 0.01  # 1% di variazione
    
    # Trova quando la media si stabilizza
    stable_from = 0
    for i in range(len(cumulative_means) - 1, max(10, len(cumulative_means) // 4), -1):
        if abs(cumulative_means[i] - final_mean) / final_mean > convergence_threshold:
            stable_from = i + 1
            break
    
    print(f"Total throughput converged after batch {stable_from}/{num_batches}")
    print(f"Final throughput estimate: {final_mean:.6f} ± {(final_mean - compute_batch_ci(batch_metrics['total_tput'], 'total_tput')[3]):.6f}")
    
    return batch_metrics

def run_integrated_simulation_fast():
    """Versione veloce per test rapidi"""
    env = simpy.Environment()
    system = SystemSimulation(env)
    env.run()  # Usa N_ARRIVALS ridotto
    system.report()

def quick_batch_analysis():
    """Analisi batch rapida per test"""
    print("Running QUICK batch means analysis...")
    return batch_means_integrated(num_batches=FAST_NUM_BATCHES, batch_time=FAST_BATCH_SIZE)

def test_different_scenarios():
    """Testa diversi scenari di carico per vedere il dual-pool in azione"""
    scenarios = [
        {
            'name': 'Low Capacity',
            'capacity': 5,
            'arrival_mean': 0.001,
            'service_mean': 0.0012,
            'description': 'Capacità molto bassa per forzare overflow'
        },
        {
            'name': 'Medium Load',
            'capacity': 25,
            'arrival_mean': 0.0008,
            'service_mean': 0.001,
            'description': 'Carico medio con overflow moderato'
        },
        {
            'name': 'High Load',
            'capacity': 50,
            'arrival_mean': 0.0005,
            'service_mean': 0.001,
            'description': 'Carico alto con overflow significativo'
        }
    ]
    
    print("\n" + "="*70)
    print("TESTING DIFFERENT LOAD SCENARIOS")
    print("="*70)
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']} - {scenario['description']}")
        print("-" * 50)
        
        # Simula temporaneamente con questi parametri
        global SINGLE_CENTER_CAPACITY, INTERARRIVAL_MEAN, SERVICE_MEAN_SINGLE
        old_capacity = SINGLE_CENTER_CAPACITY
        old_arrival = INTERARRIVAL_MEAN
        old_service = SERVICE_MEAN_SINGLE
        
        SINGLE_CENTER_CAPACITY = scenario['capacity']
        INTERARRIVAL_MEAN = scenario['arrival_mean']
        SERVICE_MEAN_SINGLE = scenario['service_mean']
        
        try:
            env = simpy.Environment()
            system = SystemSimulation(env)
            env.run()
            
            overflow_rate = system.overflow_jobs / system.total_arrivals * 100
            load_factor = SERVICE_MEAN_SINGLE / INTERARRIVAL_MEAN
            
            print(f"Load factor: {load_factor:.2f}")
            print(f"Overflow rate: {overflow_rate:.1f}%")
            print(f"Single completions: {system.single_completions}")
            print(f"Web completions: {system.web_completions}")
            print(f"Spike completions: {system.spike_completions}")
            
        except Exception as e:
            print(f"Error in scenario: {e}")
        
        # Ripristina parametri originali
        SINGLE_CENTER_CAPACITY = old_capacity
        INTERARRIVAL_MEAN = old_arrival
        SERVICE_MEAN_SINGLE = old_service

if __name__ == "__main__":
    import time
    
    print("TESTING COMPLETE INTEGRATED SYSTEM")
    print("="*50)
    print(f"Arrival rate: {1/INTERARRIVAL_MEAN:.0f} rps")
    print(f"Service rate: {1/SERVICE_MEAN_SINGLE:.0f} rps") 
    print(f"Single center capacity: {SINGLE_CENTER_CAPACITY}")
    print(f"Load factor: {SERVICE_MEAN_SINGLE/INTERARRIVAL_MEAN:.2f}")
    print(f"Expected overflow: {'YES' if SINGLE_CENTER_CAPACITY < 50 else 'MAYBE'}")
    print(f"Arrivals per simulation: {N_ARRIVALS:,}")
    print(f"Batch size: {FAST_BATCH_SIZE}")
    print(f"Number of batches: {FAST_NUM_BATCHES}")
    print("="*50)
    
    # Test scenario multipli
    test_different_scenarios()
    
    # Test singolo veloce con parametri attuali
    print("\n1. Single simulation run with current parameters:")
    start_time = time.time()
    run_integrated_simulation_fast()
    single_time = time.time() - start_time
    print(f"Time for single run: {single_time:.2f} seconds")
    
    # Batch means veloce solo se c'è overflow
    env = simpy.Environment()
    test_system = SystemSimulation(env)
    env.run()
    
    if test_system.overflow_jobs > 0:
        print(f"\n2. Quick batch means analysis (overflow detected):")
        start_time = time.time()
        quick_batch_analysis()
        batch_time = time.time() - start_time
        print(f"Time for batch analysis: {batch_time:.2f} seconds")
    else:
        print(f"\n2. Skipping batch means - no overflow detected")
        print("   Try running with lower SINGLE_CENTER_CAPACITY for overflow testing")
    
    print("\n" + "="*50)
    print("SYSTEM DESIGN ANALYSIS:")
    print("="*50)
    print(f"• Single center capacity: {SINGLE_CENTER_CAPACITY} jobs")
    print(f"• Load factor: {SERVICE_MEAN_SINGLE/INTERARRIVAL_MEAN:.2f}")
    print(f"• Arrival rate: {1/INTERARRIVAL_MEAN:.0f} rps")
    print(f"• Service rate: {1/SERVICE_MEAN_SINGLE:.0f} rps")
    if SINGLE_CENTER_CAPACITY < 50:
        print("• EXPECTED: Significant overflow to dual-pool system")
    else:
        print("• EXPECTED: Minimal overflow (may need higher load)")
    print("="*50)