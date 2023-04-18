import numpy as np


class PhaseVehicleGenerator:
    """
    Generate State or Reward based on statistics of lane vehicles.

    Parameters
    ----------
    world : World object
    I : Intersection object
    fns : list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay" and "pressure"
    in_only : boolean, whether to compute incoming lanes only
    average : None or str
        None means no averaging
        "road" means take average of lanes on each road
        "all" means take average of all lanes
    negative : boolean, whether return negative values (mostly for Reward)
    """

    def __init__(self, world, I, fns, in_only=False, average=None, negative=False, scale=1):
        self.world = world
        self.I = I

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        self.ob_length = len(fns) * 8

        self.scale = (-1) * scale if negative else scale

    def generate(self):
        results = [self.world.get_info(fn) for fn in self.fns]

        ret = np.array([])
        for i in range(len(self.fns)):
            ret = np.append(ret, results[i][self.I.id])

        ret *= self.scale

        return ret


if __name__ == "__main__":
    from world import World

    world = World("examples/config.json", thread_num=1)
    laneVehicle = LaneVehicleGenerator(world, world.intersections[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())
