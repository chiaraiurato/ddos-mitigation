from library.rvgs import Exponential, Hyperexponential
from engineering.costants import *
from library.rngs import selectStream

def get_interarrival_time(mode, arrival_p=None, arrival_l1=None, arrival_l2=None):
    selectStream(RNG_STREAM_ARRIVALS)
    if mode == "verification":
        return Exponential(0.15)  # 1 / 6.666666
    else:
        if arrival_p == None:
            arrival_p = ARRIVAL_P
            arrival_l1 = ARRIVAL_L1
            arrival_l2 = ARRIVAL_L2
        return Hyperexponential(arrival_p, arrival_l1, arrival_l2)

def get_service_time(mode):
    selectStream(RNG_STREAM_SERVICE_TIMES)
    if mode == "verification":
        return Exponential(0.16)  # 1 / 6.25
    else:
        return Hyperexponential(SERVICE_P, SERVICE_L1, SERVICE_L2)