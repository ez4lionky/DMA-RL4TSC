from . import BaseMetric
import numpy as np


class ThroughputMetric(BaseMetric):
    """
    Calculate Average Throughput.
    
    """

    def __init__(self, world):
        self.world = world
        self.world.subscribe(["lane_vehicles", "time"])
        self.name = "Average Throughput"
        self.reset()

    def update(self, done=False):
        lane_vehicles = self.world.get_info("lane_vehicles")
        current_time = self.world.get_info("time")

        for lane_id, lane in lane_vehicles.items():
            if lane_id not in self.lane_throughput:
                self.lane_throughput[lane_id] = []
            if lane_id not in self.last_lane_vehicles:
                self.last_lane_vehicles[lane_id] = []
            new_exited_vehicles = set(self.last_lane_vehicles[lane_id]) - set(lane)
            self.lane_throughput[lane_id] += list(new_exited_vehicles)
            self.last_lane_vehicles[lane_id] = lane

        return self.eval()

    def eval(self):
        return np.sum([len(lane) for lane_id, lane in self.lane_throughput.items()])

    def reset(self):
        self.last_lane_vehicles = {}
        self.lane_throughput = {}
