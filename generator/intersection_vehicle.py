import numpy as np
import math


class IntersectionVehicleGenerator:
    """
    Generate State or Reward based on statistics of intersection vehicles.

    Parameters
    ----------
    world : World object
    I : Intersection object
    fns : list of statistics to get,"vehicle_trajectory", "lane_vehicles", "history_vehicles" is needed for result "passed_count" and "passed_time_count",
                                    "vehicle_distance", "lane_vehicles" is needed for result "vehicle_map"
    targets : list of results to return, currently support "vehicle_map": map of vehicles: an image representation of vehicles’ position in this intersection
                                                           "passed_count": total number of vehicles that passed the intersection during time interval ∆t after the last action
                                                           "passed_time_count": total time (in minutes) spent on approaching lanes of vehicles that passed the intersection during time interval ∆t after the last action
             See section 4.2 of the intelliLight paper[Hua Wei et al, KDD'18] for more detailed description on these targets.
    negative : boolean, whether return negative values (mostly for Reward)
    time_interval: use to calculate
    """

    def __init__(self, params, world, I, fns=("vehicle_trajectory", "lane_vehicles", "history_vehicles", "vehicle_distance"),
                 targets=("vehicle_map"), negative=False):
        self.params = params
        self.world = world
        self.I = I

        # get lanes of intersections
        self.lanes = []
        self.in_lanes = []
        self.road_starting_points = {}
        roads = I.roads
        # for road in roads:
        #     from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
        #     self.road_starting_points[road["id"]] = road["points"][0]
        #     self.lanes.append(
        #         [road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
        #     if from_zero:
        #         self.in_lanes.append(
        #             [road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
        for road in roads:
            if_in = road["endIntersection"] == I.id
            self.road_starting_points[road["id"]] = road["points"][0]
            self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))])
            if if_in:
                self.in_lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))])

        self.all_lanes = [n for a in self.lanes for n in a]
        self.all_in_lanes = [n for a in self.in_lanes for n in a]

        # print(self.all_lanes)
        # print(self.all_in_lanes)

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns
        self.targets = targets

        self.result_functions = {
            "passed_count": self.passed_count,
            "passed_time_count": self.passed_time_count,
            "vehicle_map": self.vehicle_map,
            "current_phase": self.current_phase,
        }

        result_shape = {
            "vehicle_map": [2, len(self.all_in_lanes), params['grid_num']],
            "passed_count": [1],
            "passed_time_count": [1],
            "current_phase": [len(I.phases)],
        }
        if 'vehicle_map' in targets:
            self.map_shape = result_shape['vehicle_map']
        if 'current_phase' in targets:
            self.phase_shape = result_shape['current_phase'][0]
            
        self.negative = negative

    def if_vehicle_passed_intersection(self, vehicle_trajectory, current_time, action_interval):
        def get_target_trajectory(trajectory, last_time, current_time):
            target = []

            if len(trajectory) == 0:
                return target

            if trajectory[-1][1] + trajectory[-1][2] < current_time and trajectory[-1][1] + trajectory[-1][
                2] > last_time:  # means this vehicle left the simulator or currently in an intersection in this period
                target.append(trajectory[-1])
                target.append(['left'])
                return target

            for i, traj in enumerate(trajectory):
                if traj[1] > last_time:
                    target.append(traj)
                    if traj[1] - traj[2] > last_time and i > 0:
                        target.append(trajectory[i - 1])

            return target
        # action_interval = self.I.phase_duration_time  #!

        last_time = current_time - action_interval

        current_trajectory = get_target_trajectory(vehicle_trajectory, last_time, current_time)
        for i in range(len(current_trajectory)):
            if current_trajectory[i][0] in self.all_in_lanes:
                if i + 1 < len(current_trajectory):
                    return 1
        return 0

    def get_passed_vehicles(self, fns):
        vehicle_trajectory = fns["vehicle_trajectory"]
        history_vehicles = fns["history_vehicles"]

        passed_vehicles = []
        for vehicle in history_vehicles:
            if self.if_vehicle_passed_intersection(vehicle_trajectory[vehicle], current_time=self.time,
                                                   action_interval=self.action_interval):
                passed_vehicles.append(vehicle)
        return passed_vehicles

    def get_vehicle_position(self, distance, lane):
        road = lane[:-2]
        # start_point = list(self.world.all_roads_starting_point[road].values())
        start_point = list(self.road_starting_points[road].values())

        # 0 right, 1 up, 2 left, 3 down
        direction_code = int(road[-1])

        # 0 x-axis, 1 y-axis
        way = direction_code % 2

        # 0 1 -> 1,    2 3 -> -1
        direction = -((direction_code * 2 - 3) / abs(direction_code * 2 - 3))

        cur_position = start_point.copy()
        distance = int(distance)

        cur_position[way] += direction * distance

        return tuple(cur_position), way, direction

    def passed_count(self, fns):
        passed_vehicles = self.get_passed_vehicles(fns)
        return len(passed_vehicles)

    def passed_time_count(self, fns):
        vehicle_trajectory = fns["vehicle_trajectory"]
        passed_vehicles = self.get_passed_vehicles(fns)
        # for i in passed_vehicles:
        #     print(vehicle_trajectory[i])
        passed_time_count = 0
        for vehicle in passed_vehicles:
            passed_time_count += vehicle_trajectory[vehicle][-2][2]
        return passed_time_count

    def vehicle_map(self, fns):
        # map vehicle into cells with position and velocity information
        speed_dict = self.vehicle_velocity()
        grid_length = self.params['grid_length']  # hyperparameter, decides the density of the grid
        grid_num = self.params['grid_num']  # hyperparameter, decides the total sense area of each intersection
        len_in = len(self.all_in_lanes)  # len_in = 12
        # divide the full lane into cells
        i_pos = self.I.pos

        pos_mat = np.zeros((len_in, grid_num))
        vel_mat = np.zeros((len_in, grid_num))

        vehicle_distance = fns["vehicle_distance"]
        lane_vehicles = fns["lane_vehicles"]
        # vehicle_nums = sum(len(v) for k, v in lane_vehicles.items())
        # print(vehicle_nums)
        # for k, v in lane_vehicles.items():
        #     if len(v) > 0:
        #         print(k)

        # print(f'{self.I.id}-i_pos:{self.I.pos}-all_in_lanes:{self.all_in_lanes}')
        for i, lane in enumerate(self.all_in_lanes):
            max_speed = self.world.lanes_max_speed[lane]
            lane_length = self.world.lanes_length[lane]
            offset = self.I.width
            for vehicle in lane_vehicles[lane]:
                distance = vehicle_distance[vehicle]
                # print('distance: ', distance)
                # print(f'self.I.pos: {i_pos} ')
                # vehicle_position, way, direction = self.get_vehicle_position(distance, lane)  # (double,double),tuple
                # if vehicle_position is None:
                #     continue
                # ind = int(abs(vehicle_position[way] - i_pos[way]) // grid_length)
                ind = int((lane_length - distance - offset) // grid_length)
                # ind = int((lane_length - distance) // grid_length)
                assert ind >= 0
                # print(f'vehicle position: {vehicle_position}, way: {way}, direction: {direction} ')
                if ind < grid_num:
                    # print('ind: ', ind)
                    pos_mat[i, ind] = 1
                    # print(f'speed: {speed_dict[vehicle]}, max_speed: {max_speed}')
                    vel_mat[i, ind] = speed_dict[vehicle] / max_speed
        mat = np.concatenate([pos_mat[..., None], vel_mat[..., None]], axis=-1)
        mat = mat.transpose((2, 0, 1))
        return mat
    
    def vehicle_velocity(self):
        return self.world.eng.get_vehicle_speed()
    
    def current_phase(self, fns):
        # cur_phase = self.I._current_phase
        # cur_vec = self.world.phase_to_vector[cur_phase]
        cur_phase = self.I.current_phase
        cur_vec = np.zeros(len(self.I.phases))
        cur_vec[cur_phase] = 1
        return cur_vec
    
    def generate(self, action_interval=10):
        self.action_interval = action_interval
        self.time = self.world.eng.get_current_time()

        fns = {fn: self.world.get_info(fn) for fn in self.fns}
        ret = [self.result_functions[res](fns) for res in self.targets]
        if len(ret) == 1:
            ret = ret[0]
        if self.negative:
            ret = ret * (-1)

        return ret


if __name__ == "__main__":
    from world import World

    world = World("examples/config.json", thread_num=1)
    laneVehicle = IntersectionVehicleGenerator(world, world.intersections[0],
                                               ["vehicle_trajectory", "lane_vehicles", "vehicle_distance"],
                                               ["passed_time_count", "passed_count", "vehicle_map"])
    for _ in range(1, 301):
        world.step([_ % 3])
        ret = laneVehicle.generate()

        if _ % 10 != 0:
            continue
        print(ret)
