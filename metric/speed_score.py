from . import BaseMetric
import numpy as np


class SpeedScoreMetric(BaseMetric):
    """
    Calculate average Speed of all vehicles.

    """
    def __init__(self, world, speed_limit=11.11):
        self.world = world
        self.world.subscribe(["vehicle_speed"])
        self.name = "Average Speed Score"
        self.speed_limit = speed_limit
        self.reset()

    def update(self, done=False):
        list_vehicles = list(self.world.get_info("vehicle_speed").values())
        
        if len(list_vehicles):
            medias = np.mean(list_vehicles)
            self.vehicles_speed.append(medias) 

            return self.eval()
        return 0

    def eval(self):
        avg_speed = np.nanmean(self.vehicles_speed)
        return avg_speed/self.speed_limit

    def reset(self):
        self.vehicles_speed = []
