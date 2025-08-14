# Batch Means Parameters
BATCH_SIZE = 1024
N_BATCH = 256
CONFIDENCE_LEVEL = 0.95


# Parameters 
# Source: Performance Engineering - Learning Through Applications Using JMT 
ARRIVAL_P_VERIFICATION = 0.03033
ARRIVAL_L1_VERIFICATION = 0.4044
ARRIVAL_L2_VERIFICATION = 12.9289

ARRIVAL_P = 0.03033
ARRIVAL_L1 = 0.4044
ARRIVAL_L2 = 12.9289

# Dati Paper x 2
ARRIVAL_L1_x2 = 0.8088
ARRIVAL_L2_x2 = 25.8578

# Dati Paper x 5
ARRIVAL_L1_x5 = 2.022
ARRIVAL_L2_x5 = 64.6445

# Dati Paper x 10
ARRIVAL_L1_x10 = 4.044
ARRIVAL_L2_x10 = 129.289

# Dati Paper x 40 - taken from "A Simulation Model for the Analysis of DDoS Ampliﬁcation Attacks"
ARRIVAL_L1_x40 = 16.176
ARRIVAL_L2_x40 = 517.156

SERVICE_P = 0.03033
SERVICE_L1 = 0.3791
SERVICE_L2 = 12.1208

# Source: Typical mitigation time: ~1ms (Cisco)
# Source: https://www.cisco.com/c/en/us/td/docs/security/secure-firewall/management-center/device-config/720/management-center-device-config-72/intrusion-performance.html
MITIGATION_MEAN = 0.0011
# Buffer capacity for M/M/1/K model
# Typical enterprise firewall buffer sizes
MITIGATION_CAPACITY = 2000

# Verification
MITIGATION_CAPACITY_VERIFICATION = 4
P_FEEDBACK_VERIFICATION = 0

# False positive rate in literature: 0.01–0.03
# Source: https://www.researchgate.net/publication/335954299
P_FEEDBACK = 0.02
P_FALSE_POSITIVE = 0.01
P_LECITO = 0.1

# Server configuration
MAX_WEB_CAPACITY = 20
MAX_SPIKE_CAPACITY = 20
SCALE_THRESHOLD = 20

MAX_SPIKE_NUMBER = 4

# Arrivals Number
# N_ARRIVALS = 2585120
N_ARRIVALS = 1000000
N_ARRIVALS_VERIFICATION = 64628
# Simulation Time
MAX_SIMULATION_TIME = 20000  

# Variable used for batch means
TIME_WINDOW = 30.0          # seconds per time window
TIME_WINDOWS_PER_BATCH_VALIDATION = 32
TIME_WINDOWS_PER_BATCH = 8  # how many windows form a batch for CI on util/throughput

# Multistream 
RNG_SEED_VERIFICATION = 123456789
RNG_SEED_STANDARD     = 42

RNG_STREAM_ARRIVALS            = 0
RNG_STREAM_MITIGATION_SERVICE  = 1
RNG_STREAM_SERVICE_TIMES       = 2   # service time dei job (web/spike)
RNG_STREAM_FALSE_POSITIVE      = 3
RNG_STREAM_FEEDBACK            = 4


