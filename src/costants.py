# Parametri ottimizzati per performance
SINGLE_CENTER_CAPACITY = 10   # Ridotto drasticamente da 100
INTERARRIVAL_MEAN = 0.001      # 1,000 rps
SERVICE_MEAN_SINGLE = 0.0012   # Servizio leggermente più lento
P_FEEDBACK = 0.02              # Feedback al 2%          

# Parametri del sistema dual-pool
SI_MAX = 20
ARRIVAL_P = 0.03033
ARRIVAL_L1 = 0.4044
ARRIVAL_L2 = 12.9289
SERVICE_P = 0.03033
SERVICE_L1 = 0.3791
SERVICE_L2 = 12.1208

# Parametri simulazione
SIM_TIME = 5000              
CONF_LEVEL = 2.576
N_ARRIVALS = 5000         

# Parametri batch means 
FAST_BATCH_SIZE = 512        
FAST_NUM_BATCHES = 256    