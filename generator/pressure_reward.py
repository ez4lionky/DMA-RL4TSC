import numpy as np


class PressureRewardGenerator:
    """
    Generate Reward based on statistics of lane vehicles.

    Parameters
    ----------
    world : World object
    I : Intersection object
    fns : "pressure"
    scale : int, reward * scale
    negative : boolean, whether return negative values 
    """

    def __init__(self, world, I, scale=1, negative=False):
        self.world = world
        self.I = I

        # subscribe functions
        self.world.subscribe("pressure")
        self.fns = ["pressure"]

        # calculate result dimensions
        # size = sum(len(x) for x in self.lanes)
        self.scale = scale * -1 if negative else scale

    def generate(self):
        ret = {fn: self.world.get_info(fn) for fn in self.fns}
        ret = ret[self.fns[0]].get(self.I.id, 0)
        ret = ret * self.scale
        return [ret]


if __name__ == "__main__":
    from world import World

    world = World("../envs/real_1x1/config.json", thread_num=1)
    laneVehicle = PressureRewardGenerator(world, world.intersections[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())
