import gym
import sys
import json
import numpy as np
import os.path as osp
import networkx as nx


class TLCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.

    Parameters
    ----------
    world: World object
    agents: list of agent, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric_name
    """

    def __init__(self, world, agents, metric):
        self.world = world

        self.eng = self.world.eng
        self.n_agents = len(self.world.intersection_ids)
        self.n = self.n_agents

        assert len(agents) == self.n_agents

        self.agents = agents
        action_dims = [agent.action_space.n for agent in agents]
        self.action_space = gym.spaces.MultiDiscrete(action_dims)

        if isinstance(metric, list):
            self.metric = metric
        else:
            self.metric = [metric]

    def update_metric(self):
        if self.world.eng.get_current_time() % 5 == 0:
            for metric in self.metric:
                metric.update(done=False)

    def reset_metric(self):
        for metric in self.metric:
            metric.reset()

    def step(self, actions):
        # assert len(actions) == self.n_agents

        self.world.step(actions)
        self.update_metric()

        obs = [agent.get_ob() for agent in self.agents]
        
        rewards = [agent.get_reward() for agent in self.agents]
        num_vehicles = self.world.count_vehicles()

        if num_vehicles == 0 and self.world.eng.get_current_time() > 20:
            dones = [True] * self.n_agents
        else:
            dones = [False] * self.n_agents

        infos = {"count_vehicles": num_vehicles}

        return obs, rewards, dones, infos

    def reset(self):
        self.world.reset()
        self.reset_metric()
        obs = [agent.get_ob() for agent in self.agents]
        return obs


class RoadGraph:

    def __init__(self, config):
        roadnet_file = osp.join(config["dir"], config["roadnetFile"])
        self.roadnet_dict = json.load(open(roadnet_file, "r"))
        self.net_edge_dict = {}
        self.net_node_dict = {}
        self.net_lane_dict = {}

        self.vis_graph = nx.DiGraph()
        self.node_positions = {}
        self.edge_positions = {}
        self.generate_node_dict()
        self.generate_edge_dict()
        self.generate_lane_dict()

    def generate_node_dict(self):
        '''
        node dict has key as node id, value could be the dict of input nodes and output nodes
        :return:
        '''

        for i, node_dict in enumerate(self.roadnet_dict['intersections']):
            node_id = node_dict['id']
            node_point = np.array([node_dict['point'][k] for k in ['x', 'y']])
            road_links = node_dict['roads']
            input_nodes = []
            output_nodes = []
            input_edges = []
            output_edges = []
            for road_link_id in road_links:
                road_link_dict = self._get_road_dict(road_link_id)
                if road_link_dict['startIntersection'] == node_id:
                    output_edges.append(road_link_id)
                    end_node = road_link_dict['endIntersection']
                    output_nodes.append(end_node)
                elif road_link_dict['endIntersection'] == node_id:
                    input_edges.append(road_link_id)
                    start_node = road_link_dict['startIntersection']
                    input_nodes.append(start_node)

            net_node = {
                'index': i,
                'node_id': node_id,
                'node_point': node_point,
                'input_nodes': list(set(input_nodes)),
                'input_edges': list(set(input_edges)),
                'output_nodes': list(set(output_nodes)),
                'output_edges': output_edges,
                'gen_flow_num': 0,
                'pass_flow_num': 0,
            }
            if node_id not in self.net_node_dict.keys():
                self.net_node_dict[node_id] = net_node
                self.node_positions[node_id[13:]] = node_point
                self.vis_graph.add_nodes_from([(node_id[13:], net_node)])

    def _get_road_dict(self, road_id):
        for item in self.roadnet_dict['roads']:
            if item['id'] == road_id:
                return item
        print("Cannot find the road id {0}".format(road_id))
        sys.exit(-1)
        # return None

    def generate_edge_dict(self):
        '''
        edge dict has key as edge id, value could be the dict of input edges and output edges
        :return:
        '''
        node_dict = self.net_node_dict
        for edge_dict in self.roadnet_dict['roads']:
            edge_id = edge_dict['id']
            input_node = edge_dict['startIntersection']
            output_node = edge_dict['endIntersection']
            input_point = node_dict[input_node]['node_point']
            output_point = node_dict[output_node]['node_point']

            length = np.linalg.norm(input_point - output_point)
            net_edge = {
                'edge_id': edge_id,
                'input_node': input_node,
                'output_node': output_node,
                'length': length,
                'gen_flow_num': 0,
                'pass_flow_num': 0,
            }
            if edge_id not in self.net_edge_dict.keys():
                direction = int(edge_id[-1])
                self.net_edge_dict[edge_id] = net_edge
                self.vis_graph.add_edge(input_node[13:], output_node[13:], **net_edge)
                mid_point = (input_point + output_point) / 2
                all_offsets = {0: [-70, -60], 1: [30, 0], 2: [-70, 60], 3: [-155, 0]}
                offset = np.array(all_offsets[direction])
                self.edge_positions[(input_node[13:], output_node[13:])] = mid_point + offset

    def generate_lane_dict(self):
        lane_dict = {}
        for node_dict in self.roadnet_dict['intersections']:
            for road_link in node_dict["roadLinks"]:
                lane_links = road_link["laneLinks"]
                start_road = road_link["startRoad"]
                end_road = road_link["endRoad"]
                for lane_link in lane_links:
                    start_lane = start_road + "_" + str(lane_link['startLaneIndex'])
                    end_lane = end_road + "_" + str(lane_link["endLaneIndex"])
                    if start_lane not in lane_dict:
                        lane_dict[start_lane] = {
                            "output_lanes": [end_lane],
                            "input_lanes": []
                        }
                    else:
                        lane_dict[start_lane]["output_lanes"].append(end_lane)
                    if end_lane not in lane_dict:
                        lane_dict[end_lane] = {
                            "output_lanes": [],
                            "input_lanes": [start_lane]
                        }
                    else:
                        lane_dict[end_lane]["input_lanes"].append(start_lane)

        self.net_lane_dict = lane_dict

    def hasEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return True
        else:
            return False

    def getEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return edge_id
        else:
            return None

    def getOutgoing(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return self.net_edge_dict[edge_id]['output_edges']
        else:
            return []
