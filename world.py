import json
import os.path as osp
import networkx as nx
import cityflow

import numpy as np
from math import atan2, pi
import sys
from envs.environment import RoadGraph


def _get_direction(road, out=True):
    if out:
        x = road["points"][1]["x"] - road["points"][0]["x"]
        y = road["points"][1]["y"] - road["points"][0]["y"]
    else:
        x = road["points"][-2]["x"] - road["points"][-1]["x"]
        y = road["points"][-2]["y"] - road["points"][-1]["y"]
    tmp = atan2(x, y)
    return tmp if tmp >= 0 else (tmp + 2 * pi)


class Intersection(object):
    def __init__(self, intersection, world, action_interval=10):
        self.id = intersection["id"]
        self.eng = world.eng
        self.action_interval = action_interval
        self.width = intersection['width']
        self.pos = [intersection['point']['x'], intersection['point']['y']]
        # self.min_phase_time, self.max_phase_time = world.min_phase_time, world.max_phase_time
        self.data_chunk_length = 10

        # incoming and outgoing roads of each intersection, clock-wise order from North
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = None
        self.in_roads = None

        # links and phase information of each intersection
        self.roadlinks = []
        self.lanelinks_of_roadlink = []
        self.startlanes = []
        self.lanelinks = []
        self.phase_available_roadlinks = []
        self.phase_available_lanelinks = []
        self.phase_available_startlanes = []

        # define yellow phases, currently default to 0
        self.yellow_phase_id = [0]
        self.yellow_phase_time = world.yellow_time

        # parsing links and phases
        for roadlink in intersection["roadLinks"]:
            self.roadlinks.append((roadlink["startRoad"], roadlink["endRoad"]))
            lanelinks = []
            for lanelink in roadlink["laneLinks"]:
                startlane = roadlink["startRoad"] + "_" + str(lanelink["startLaneIndex"])
                self.startlanes.append(startlane)
                endlane = roadlink["endRoad"] + "_" + str(lanelink["endLaneIndex"])
                lanelinks.append((startlane, endlane))
            self.lanelinks.extend(lanelinks)
            self.lanelinks_of_roadlink.append(lanelinks)

        self.startlanes = list(set(self.startlanes))

        phases = intersection["trafficLight"]["lightphases"]
        self.phases = [i for i in range(len(phases)) if not i in self.yellow_phase_id]
        for i in self.phases:
            phase = phases[i]
            self.phase_available_roadlinks.append(phase["availableRoadLinks"])
            phase_available_lanelinks = []
            phase_available_startlanes = []
            for roadlink_id in phase["availableRoadLinks"]:
                lanelinks_of_roadlink = self.lanelinks_of_roadlink[roadlink_id]
                phase_available_lanelinks.extend(lanelinks_of_roadlink)
                for lanelinks in lanelinks_of_roadlink:
                    phase_available_startlanes.append(lanelinks[0])
            self.phase_available_lanelinks.append(phase_available_lanelinks)
            phase_available_startlanes = list(set(phase_available_startlanes))
            self.phase_available_startlanes.append(phase_available_startlanes)
        self.reset()

    def insert_road(self, road, out):
        self.roads.append(road)
        self.outs.append(out)
        self.directions.append(_get_direction(road, out))

    def sort_roads(self, RIGHT):
        order = sorted(range(len(self.roads)),
                       key=lambda i: (self.directions[i], self.outs[i] if RIGHT else not self.outs[i]))
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]

    def _change_phase(self, phase, interval):
        self.eng.set_tl_phase(self.id, phase)
        self._current_phase = phase
        self.current_phase_time = interval

    def step(self, action, interval):
        if self._current_phase in self.yellow_phase_id:
            if self.current_phase_time >= self.yellow_phase_time:
                self._change_phase(self.phases[self.action_before_yellow], interval)
                self.current_phase = self.action_before_yellow
            else:
                self.current_phase_time += interval
        else:
            if action == self.current_phase:
                self.current_phase_time += interval
            else:
                if self.yellow_phase_time > 0:
                    self._change_phase(self.yellow_phase_id[0], interval)
                    self.action_before_yellow = action
                else:
                    self._change_phase(self.phases[action], interval)
                    self.current_phase = action

        # no yellow
        # if action == self.current_phase:
        #     self.current_phase_time += interval
        # else:
        #     self._change_phase(action, interval)
        #     self.current_phase = action

    def reset(self):
        # record phase info
        self.current_phase = 0  # phase id in self.phases (excluding yellow)
        self._current_phase = self.phases[0]  # true phase id (including yellow)
        self.eng.set_tl_phase(self.id, self._current_phase)
        self.current_phase_time = 0
        self.action_before_yellow = None
        self.phase_duration_time = self.action_interval


class World(object):
    """
    Create a CityFlow engine and maintain informations about CityFlow world
    """
    phase_to_vector = {
        0: [0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0],
        5: [1, 1, 0, 0, 0, 0, 0, 0],
        6: [0, 0, 1, 1, 0, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 1, 1],
        8: [0, 0, 0, 0, 1, 1, 0, 0]
    }
    accelerated_distance = 0.5 * 2 * (11.11 / 2) ** 2
    max_speed = 11.111

    def __init__(self, cityflow_config, thread_num, action_interval=10, yellow_time=0, verbose=False):
        print("building world...")
        self.eng = cityflow.Engine(cityflow_config, thread_num=thread_num)
        with open(cityflow_config) as f:
            cityflow_config = json.load(f)
        self.cityflow_config = cityflow_config
        self.roadnet = self._get_roadnet(cityflow_config)
        self.road_graph = RoadGraph(cityflow_config)
        self.verbose = verbose
        self.RIGHT = True  # vehicles moves on the right side, currently always set to true due to CityFlow's mechanism
        self.interval = cityflow_config["interval"]
        self.yellow_time = yellow_time

        # get all non virtual intersections
        self.intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersection_ids = [i["id"] for i in self.intersections]

        # create non-virtual Intersections
        print("creating intersections...")
        non_virtual_intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersections = [Intersection(i, self, action_interval) for i in non_virtual_intersections]
        self.intersection_ids = [i["id"] for i in non_virtual_intersections]
        self.id2intersection = {i.id: i for i in self.intersections}
        self.id2idx = {i.id: idx for idx, i in enumerate(self.intersections)}
        print("intersections created.")

        # id of all roads and lanes
        graph = nx.DiGraph()
        graph.add_nodes_from({idx: i.id for idx, i in enumerate(self.intersections)})
        print("parsing roads...")
        self.all_roads = []
        self.all_lanes = []
        self.lanes_max_speed = {}
        self.lanes_length = {}

        for road in self.roadnet["roads"]:
            self.all_roads.append(road["id"])
            i = 0
            st_p, end_p = road['points']
            st_p, end_p = [st_p['x'], st_p['y']], [end_p['x'], end_p['y']]
            length = np.sqrt(np.subtract(st_p, end_p).sum() ** 2)
            for _ in road["lanes"]:
                lane_id = road["id"] + "_" + str(i)
                self.all_lanes.append(lane_id)
                self.lanes_max_speed[lane_id] = _['maxSpeed']
                self.lanes_length[lane_id] = length
                i += 1

            s_iid = road["startIntersection"]
            if s_iid in self.intersection_ids:
                self.id2intersection[s_iid].insert_road(road, True)
            e_iid = road["endIntersection"]
            if e_iid in self.intersection_ids:
                self.id2intersection[e_iid].insert_road(road, False)
            if s_iid in self.id2idx.keys() and e_iid in self.id2idx.keys():
                # print(self.id2idx[s_iid], self.id2idx[e_iid])
                graph.add_edge(self.id2idx[s_iid], self.id2idx[e_iid])
        self.graph = graph

        for i in self.intersections:
            i.sort_roads(self.RIGHT)
        print("roads parsed.")

        # initializing info functions
        self.info_functions = {
            "vehicles": (lambda: self.eng.get_vehicles(include_waiting=True)),
            "lane_count": self.eng.get_lane_vehicle_count,
            "lane_waiting_count": self.eng.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.eng.get_lane_vehicles,
            "time": self.eng.get_current_time,
            "vehicle_distance": self.eng.get_vehicle_distance,
            "vehicle_speed": self.eng.get_vehicle_speed,
            "avg_travel_time": self.eng.get_average_travel_time,
            "pressure": self.get_pressure,
            "speed_score": self.get_speed_score,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "waiting_time_count": self.get_vehicle_waiting_time,
            "lane_delay": self.get_lane_delay,
            "vehicle_trajectory": self.get_vehicle_trajectory,
            "history_vehicles": self.get_history_vehicles,
            "car_count": self.get_car_count,
            "state_of_three": self.get_state_of_three,
            "phase_vehicles": self.get_phase_vehicles
        }
        self.fns = []
        self.info = {}

        self.vehicle_waiting_time = {}  # key: vehicle_id, value: the waiting time of this vehicle since last halt.
        self.vehicle_trajectory = {}  # key: vehicle_id, value: [[lane_id_1, enter_time, time_spent_on_lane_1], ... , [lane_id_n, enter_time, time_spent_on_lane_n]]
        self.history_vehicles = set()

        print("world built.")

    def get_pressure(self):
        vehicles = self.eng.get_lane_vehicle_count()
        pressures = {}
        for i in self.intersections:
            pressure = 0
            in_lanes = []
            for road in i.in_roads:
                from_zero = (road["startIntersection"] == i.id) if self.RIGHT else (
                        road["endIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    in_lanes.append(road["id"] + "_" + str(n))
            out_lanes = []
            for road in i.out_roads:
                from_zero = (road["endIntersection"] == i.id) if self.RIGHT else (
                        road["startIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    out_lanes.append(road["id"] + "_" + str(n))

            for lane in vehicles.keys():
                if lane in in_lanes:
                    pressure += vehicles[lane]
                if lane in out_lanes:
                    pressure -= vehicles[lane]
            pressures[i.id] = pressure
        return pressures

    def get_speed_score(self):
        speed_score = {}
        info = self.info
        vehicle_speed = self.eng.get_vehicle_speed()
        vehicle_distance = info["vehicle_distance"]
        lane_vehicles = info["lane_vehicles"]
        for lane in self.all_lanes:
            speed_score[lane] = 0
            for vehicle in lane_vehicles[lane]:
                cur_distance = vehicle_distance[vehicle]
                if cur_distance > self.accelerated_distance:
                    cur_speed = vehicle_speed[vehicle]
                    score = 1 - cur_speed / self.max_speed
                else:
                    score = 0
                speed_score[lane] += score
            # speed_score[lane] /= len(lane_vehicles[lane])
        return speed_score

    def get_car_count(self):
        vehicles = self.eng.get_lane_vehicle_count()
        car_count = {}
        for i in self.intersections:
            count = 0
            in_lanes = []
            for road in i.in_roads:
                from_zero = (road["startIntersection"] == i.id) if self.RIGHT else (
                        road["endIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    in_lanes.append(road["id"] + "_" + str(n))

            for lane in vehicles.keys():
                if lane in in_lanes:
                    count += vehicles[lane]

            car_count[i.id] = count
        return car_count

    # return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
    # [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def get_vehicle_lane(self):
        # get the current lane of each vehicle. {vehicle_id: lane_id}
        vehicle_lane = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        for lane in self.all_lanes:
            for vehicle in lane_vehicles[lane]:
                vehicle_lane[vehicle] = lane
        return vehicle_lane

    def get_vehicle_waiting_time(self):
        # the waiting time of vehicle since last halt.
        vehicles = self.eng.get_vehicles(include_waiting=False)
        vehicle_speed = self.eng.get_vehicle_speed()
        for vehicle in vehicles:
            if vehicle not in self.vehicle_waiting_time.keys():
                self.vehicle_waiting_time[vehicle] = 0
            if vehicle_speed[vehicle] < 0.1:
                self.vehicle_waiting_time[vehicle] += 1
            else:
                self.vehicle_waiting_time[vehicle] = 0
        return self.vehicle_waiting_time

    def get_lane_waiting_time_count(self):
        # the sum of waiting times of vehicles on the lane since their last halt.
        lane_waiting_time = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        vehicle_waiting_time = self.get_vehicle_waiting_time()
        for lane in self.all_lanes:
            lane_waiting_time[lane] = 0
            for vehicle in lane_vehicles[lane]:
                lane_waiting_time[lane] += vehicle_waiting_time[vehicle]
        return lane_waiting_time

    def get_lane_delay(self):
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        # set speed limit to 11.11 by default
        speed_limit = 11.11
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_delay = {}
        lanes = self.all_lanes
        vehicle_speed = self.eng.get_vehicle_speed()

        for lane in lanes:
            vehicles = lane_vehicles[lane]
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            for vehicle in vehicles:
                speed = vehicle_speed[vehicle]
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = speed_limit
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[lane] = 1 - lane_avg_speed / speed_limit
        return lane_delay

    def get_phase_vehicles(self):
        phase_vehicles = {}

        for i in self.intersections:
            vehicles = self.eng.get_lane_waiting_vehicle_count()
            v = []

            for idx in range(len(i.phase_available_startlanes)):
                v.append(np.sum([vehicles[key] for key in vehicles if key in i.phase_available_startlanes[idx]]))

            phase_vehicles[i.id] = v

        return phase_vehicles

    def get_state_of_three(self):
        state_of_three = {}
        for i in self.intersections:
            ##Pega lanes da fase atual
            act_phase = i.current_phase
            act_roads = i.phase_available_startlanes[act_phase]
            ##Pega ruas da proxima fase
            nxt_phase = (act_phase + 1) % len(i.phases)
            nxt_roads = i.phase_available_startlanes[nxt_phase]
            ##Pega ruas das outras fases (excluindo as que ja foram incluidas no act_roads e nxt_roads)
            otr_roads = [road for road in i.startlanes if (road not in act_roads) and (road not in nxt_roads)]
            # print(act_roads)
            # print(nxt_roads)
            # print(otr_roads)
            vehicles = self.eng.get_lane_waiting_vehicle_count()

            act_vehicles = np.sum([vehicles[key] for key in vehicles if key in act_roads])
            nxt_vehicles = np.sum([vehicles[key] for key in vehicles if key in nxt_roads])
            otr_vehicles = np.sum([vehicles[key] for key in vehicles if key in otr_roads])

            state_of_three[i.id] = [act_vehicles, nxt_vehicles, otr_vehicles]

        # print(state_of_three)
        return state_of_three

    def get_state_of_three_by_phase(self, intersection, current_phase):
        for i in self.intersections:
            if i == intersection:
                ##Pega lanes da fase atual
                act_phase = current_phase
                act_roads = i.phase_available_startlanes[act_phase]
                ##Pega ruas da proxima fase
                nxt_phase = (act_phase + 1) % len(i.phases)
                nxt_roads = i.phase_available_startlanes[nxt_phase]
                ##Pega ruas das outras fases (excluindo as que ja foram incluidas no act_roads e nxt_roads)
                otr_roads = [road for road in i.startlanes if (road not in act_roads) and (road not in nxt_roads)]
                # print(act_roads)
                # print(nxt_roads)
                # print(otr_roads)
                vehicles = self.eng.get_lane_waiting_vehicle_count()

                act_vehicles = np.sum([vehicles[key] for key in vehicles if key in act_roads])
                nxt_vehicles = np.sum([vehicles[key] for key in vehicles if key in nxt_roads])
                otr_vehicles = np.sum([vehicles[key] for key in vehicles if key in otr_roads])

        return [act_vehicles, nxt_vehicles, otr_vehicles]

    def get_vehicle_trajectory(self):
        # lane_id and time spent on the corresponding lane that each vehicle went through
        vehicle_lane = self.get_vehicle_lane()
        vehicles = self.eng.get_vehicles(include_waiting=False)
        for vehicle in vehicles:
            if vehicle not in self.vehicle_trajectory:
                self.vehicle_trajectory[vehicle] = [[vehicle_lane[vehicle], int(self.eng.get_current_time()), 0]]
            else:
                if vehicle not in vehicle_lane.keys():
                    continue
                if vehicle_lane[vehicle] == self.vehicle_trajectory[vehicle][-1][0]:
                    self.vehicle_trajectory[vehicle][-1][2] += 1
                else:
                    self.vehicle_trajectory[vehicle].append(
                        [vehicle_lane[vehicle], int(self.eng.get_current_time()), 0])
        return self.vehicle_trajectory

    def get_history_vehicles(self):
        self.history_vehicles.update(self.eng.get_vehicles())
        return self.history_vehicles

    def _get_roadnet(self, cityflow_config):
        roadnet_file = osp.join(cityflow_config["dir"], cityflow_config["roadnetFile"])
        with open(roadnet_file) as f:
            roadnet = json.load(f)
        return roadnet

    def subscribe(self, fns):
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if not fn in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception("info function %s not exists" % fn)

    def step(self, actions=None):
        if actions is not None:
            for i, action in enumerate(actions):
                self.intersections[i].step(action, self.interval)
        self.eng.next_step()
        self._update_infos()
        if self.verbose:
            self.update_analysis()

    def reset_analysis(self):
        vis_graph = self.road_graph.vis_graph
        for n in vis_graph.nodes:
            vis_graph.nodes[n].update({'run_pass_flow_num': 0})
        for e in vis_graph.edges:
            vis_graph.edges[e].update({'run_pass_flow_num': 0})
        return

    def update_analysis(self):
        vis_graph = self.road_graph.vis_graph
        return

    def reset(self):
        self.eng.reset()
        for I in self.intersections:
            I.reset()
        self._update_infos()
        if self.verbose:
            self.reset_analysis()

    def _update_infos(self):
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_info(self, info):
        return self.info[info]

    def count_vehicles(self):
        return len(self.eng.get_vehicles(include_waiting=True))


if __name__ == "__main__":
    world = World("envs/jinan_3_4/config.json", thread_num=1)
    # print(len(world.intersections[0].startlanes))
    for n in range(200):
        world.step()

    print(world.intersections[0].phase_available_startlanes)

    print(world.get_phase_vehicles()['intersection_1_1'])

    print(world.get_pressure()['intersection_1_1'])
