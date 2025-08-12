from library.rvgs import Exponential, Hyperexponential
from engineering.costants import *

def get_interarrival_time(mode):
    if mode == "verification":
        return Exponential(0.15)  # 1 / 6.666666
    else:
        return Hyperexponential(ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2)

def get_service_time(mode):
    if mode == "verification":
        return Exponential(0.16)  # 1 / 6.25
    else:
        return Hyperexponential(SERVICE_P, SERVICE_L1, SERVICE_L2)
