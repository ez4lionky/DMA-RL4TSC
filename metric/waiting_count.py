from . import BaseMetric
import numpy as np


class WaitingCountMeric(BaseMetric):
    """
    Calculate average travel time of all vehicles.
    For each vehicle, travel time measures time between it entering and leaving the roadnet.
    """

    def __init__(self, world):
        super(WaitingCountMeric, self).__init__(world)
        self.world = world
        self.world.subscribe(["lane_waiting_count"])
        self.name = "Lane waiting count"
        self.lane_waiting_count = []
        self.reset()

    def update(self, done=False):
        self.lane_waiting_count = self.world.get_info("lane_waiting_count")
        return self.eval()

    def eval(self):
        return sum(self.lane_waiting_count.values())

    def reset(self):
        # self.vehicle_enter_time = {}
        self.lane_waiting_count = []
