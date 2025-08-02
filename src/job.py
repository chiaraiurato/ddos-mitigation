class Job:
    def __init__(self, arrival_time, service_time, source='overflow'):
        self.arrival = arrival_time
        self.remaining = service_time
        self.last_updated = arrival_time
        self.source = source  