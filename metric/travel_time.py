from . import BaseMetric
import numpy as np


class TravelTimeMetric(BaseMetric):
    """
    Calculate average travel time of all vehicles.
    For each vehicle, travel time measures time between it entering and leaving the roadnet.
    """

    def __init__(self, world):
        self.world = world
        self.world.subscribe(["avg_travel_time"])
        self.name = "Average Travel Time"
        self.reset()

    def update(self, done=False):
        self.travel_time = self.world.get_info("avg_travel_time")

        return self.eval()

    def eval(self):
        return self.travel_time

    def reset(self):
        # self.vehicle_enter_time = {}
        self.travel_time = []
