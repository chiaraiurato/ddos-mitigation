# PARAMETRI VECCHI 

# # Parametri ottimizzati per performance
# SINGLE_CENTER_CAPACITY = 10   # Ridotto drasticamente da 100
# INTERARRIVAL_MEAN = 0.001      # 1,000 rps
# SERVICE_MEAN_SINGLE = 0.0012   # Servizio leggermente più lento
# P_FEEDBACK = 0.02              # Feedback al 2%          

# # Parametri del sistema dual-pool
# SI_MAX = 20
# ARRIVAL_P = 0.03033
# ARRIVAL_L1 = 0.4044
# ARRIVAL_L2 = 12.9289
# SERVICE_P = 0.03033
# SERVICE_L1 = 0.3791
# SERVICE_L2 = 12.1208

# # Parametri simulazione
# SIM_TIME = 5000              
# CONF_LEVEL = 2.576
# N_ARRIVALS = 5000         

# # Parametri batch means 
# FAST_BATCH_SIZE = 512        
# FAST_NUM_BATCHES = 256    

# Paramtri in sim2

# Parameters (based on literature data)
# Source: https://www.researchgate.net/figure/Average-waiting-time-of-requests-in-complex-DDoS-scenarios-with-or-without-the-DDoS_fig4_335954299
ARRIVAL_P = 0.03
ARRIVAL_L1 = 0.4044
ARRIVAL_L2 = 12.9289

# Source: Hyperexponential values from original model
SERVICE_P = 0.03033
SERVICE_L1 = 0.3791
SERVICE_L2 = 12.1208

# Source: Typical mitigation time: ~1ms (Cloudflare)
# Source: https://www.cloudflare.com/ddos/ddos-mitigation/?utm_source=chatgpt.com
MITIGATION_MEAN = 0.001

# False positive rate in literature: 0.01–0.03
# Source: https://www.researchgate.net/publication/335954299
P_FEEDBACK = 0.02
P_FALSE_POSITIVE = 0.01
P_LECITO = 0.1

# Server configuration
MAX_WEB_CAPACITY = 20
MAX_SPIKE_CAPACITY = 20
SCALE_THRESHOLD = 20
N_ARRIVALS = 5000000