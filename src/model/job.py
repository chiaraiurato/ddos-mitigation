class Job:
    def __init__(self, job_id, arrival_time, service_time, is_legal):
        self.id = job_id
        self.arrival = arrival_time
        self.remaining = service_time
        self.original_service = service_time
        self.last_updated = arrival_time
        self.is_legal = is_legal