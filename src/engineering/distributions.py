from library.rvgs import Exponential, Hyperexponential
from engineering.costants import *

def get_interarrival_time(mode):
    if mode == "verification":
        return Exponential(0.15)  # 1 / 6.666666
    else:
        return Hyperexponential(ARRIVAL_P_VERIFICATION, ARRIVAL_L1_VERIFICATION, ARRIVAL_L2_VERIFICATION)

def get_service_time(mode):
    if mode == "verification":
        return Exponential(0.16)  # 1 / 6.25
    else:
        return Hyperexponential(SERVICE_P, SERVICE_L1, SERVICE_L2)
