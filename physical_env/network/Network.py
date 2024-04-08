import copy
import numpy as np
from scipy.spatial import distance

import optimizer
# from optimizer.utils import network_clustering
from physical_env.network.utils import network_clustering, network_cluster_id_node


class Network:
    def __init__(self, env, listNodes, baseStation, listTargets, mc_list=None, max_time=None):
        self.env = env
        self.listNodes = listNodes
        self.baseStation = baseStation
        self.listTargets = listTargets
        self.targets_active = [1 for _ in range(len(self.listTargets))]
        self.alive = 1
        # Setting BS and Node environment and network
        baseStation.env = self.env
        baseStation.net = self
        self.max_time = max_time
        self.mc_list = mc_list
        self.network_cluster = []
        self.network_cluster_id_node = []

        self.frame = np.array([self.baseStation.location[0], self.baseStation.location[0], self.baseStation.location[1],
                               self.baseStation.location[1]], np.float64)
        it = 0
        for node in self.listNodes:
            node.id = it
            node.env = self.env
            node.net = self
            it += 1
            self.frame[0] = min(self.frame[0], node.location[0])
            self.frame[1] = max(self.frame[1], node.location[0])
            self.frame[2] = min(self.frame[2], node.location[1])
            self.frame[3] = max(self.frame[3], node.location[1])
        self.nodes_density = len(self.listNodes) / ((self.frame[1] - self.frame[0]) * (self.frame[3] - self.frame[2]))
        it = 0

        # Setting name for each target
        for target in listTargets:
            target.id = it
            it += 1

    # Function is for setting nodes' level and setting all targets as covered
    def setLevels(self):
        for node in self.listNodes:
            node.level = -1
        tmp1 = []
        tmp2 = []
        for node in self.baseStation.direct_nodes:
            if node.status == 1:
                node.level = 1
                tmp1.append(node)

        for i in range(len(self.targets_active)):
            self.targets_active[i] = 0

        while True:
            if len(tmp1) == 0:
                break
            # For each node, we set value of target covered by this node as 1
            # For each node, if we have not yet reached its neighbor, then level of neighbors equal this node + 1
            for node in tmp1:
                for target in node.listTargets:
                    self.targets_active[target.id] = 1
                for neighbor in node.neighbors:
                    if neighbor.status == 1 and neighbor.level == -1:
                        tmp2.append(neighbor)
                        neighbor.level = node.level + 1

            # Once all nodes at current level have been expanded, move to the new list of next level
            tmp1 = tmp2[:]
            tmp2.clear()
        return

    def operate(self, t=1, optimizer=None):
        request_id = []
        for node in self.listNodes:
            self.env.process(node.operate(t=t))
        self.env.process(self.baseStation.operate(t=t))
        first_step = 0
        energy_warning = self.listNodes[0].threshold * 30

        while True:
            yield self.env.timeout(t / 10.0)
            self.setLevels()
            self.alive = self.check_targets()
            if not self.network_cluster:
                yield self.env.timeout(10)
                self.network_cluster = network_clustering(network=self)
                self.network_cluster_id_node = network_cluster_id_node(network=self)
                optimizer.action_list = self.network_cluster
            yield self.env.timeout(9.0 * t / 10.0)
            # optimizer.action_list = network_clustering(optimizer=optimizer, network=self, nb_cluster=83)
            for index, node in enumerate(self.listNodes):
                # if node.energy <= node.threshold * 2:
                if node.energy <= energy_warning:
                    node.request(optimizer=optimizer, t=t)
                    # print(optimizer.list_request)
                    request_id.append(index)
                else:
                    node.is_request = False
            arr_active_mc = []
            for mc in self.mc_list:
                if mc.cur_action_type == "deactive":
                    arr_active_mc.append(0)
                else:
                    arr_active_mc.append(1)
            a = 0
            b = 0
            len_list_request_before = len(optimizer.list_request)
            if optimizer and self.alive:
                for mc in self.mc_list:
                    # for other_mc in self.mc_list:
                    #     if (other_mc.id != mc.id and distance.euclidean())
                    # if distance.euclidean(mc.end, )
                    mc.runv2(network=self, time_stem=self.env.now, net=self, optimizer=optimizer)
                    # if (self.env.now % 100 == 0):
                    #     print(self.env.now, "time_stem")

                # Phương án 4
            #     if np.argmin(arr_active_mc) == 0:
            #         for mc in self.mc_list:
            #             if (not mc.is_active) and optimizer.list_request:
            #                 # new_list_request = []
            #                 # for request in optimizer.list_request:
            #                 #     if self.listNodes[request["id"]].energy < energy_warning:
            #                 #         new_list_request.append(
            #                 #             {"id": self.listNodes[request["id"]].id,
            #                 #              "energy": self.listNodes[request["id"]].energy,
            #                 #              "energyCS": self.listNodes[request["id"]].energyCS,
            #                 #              "energyRR": self.listNodes[request["id"]].energyRR, "time": t})
            #                 #     else:
            #                 #         self.listNodes[request["id"]].is_request = False
            #                 # optimizer.list_request = new_list_request
            #                 result = mc.update_q_table(net=self, optimizer=optimizer, time_stem=t)
            #                 # print(mc.q_table[mc.state])
            #                 is_same_destination = False
            #                 for other_mc in self.mc_list:
            #                     if other_mc.id != mc.id:
            #                         choose_mc_next_destination = (mc.next_phy_action[0], mc.next_phy_action[1])
            #                         other_mc_next_destination = (other_mc.next_phy_action[0], other_mc.next_phy_action[1])
            #                         if distance.euclidean(choose_mc_next_destination, other_mc_next_destination) < 1:
            #                             is_same_destination = True
            #                             break
            #                 if not is_same_destination:
            #                     phy_action = self.mc_list[mc.id].next_phy_action
            #                     self.delete_request(id_cluster=mc.state, optimizer=optimizer)
            #                     # s = "stop here"
            #                     self.env.process(mc.operate_step_v4(phy_action=phy_action))
            #
            # # Phương án 1
            #     if (len_list_request_before != len_list_request_after and np.argmin(arr_active_mc) == 0) or (len_list_request_before == len_list_request_after and np.argmin(arr_active_mc != 0)):
            #         len_list_request_after = len(optimizer.list_request)
            #         arr_q_max = []
            #         for id, mc in enumerate(self.mc_list):
            #             # temp = self.min_node()
            #             # mc.state = self.check_cluster(id_node=temp)
            #             result = mc.update_q_table(net=self, optimizer=optimizer, time_stem=t)
            #             if mc.is_active:
            #                 arr_q_max.append(0)
            #             else:
            #                 arr_q_max.append(result)
            #
            #         choose_mc = np.argmax(arr_q_max)
            #         is_same_destination = False
            #         for other_mc in self.mc_list:
            #             choose_mc_next_destination = (self.mc_list[choose_mc].next_phy_action[0], self.mc_list[choose_mc].next_phy_action[1])
            #             other_mc_next_destination = (other_mc.cur_phy_action[0], other_mc.cur_phy_action[1])
            #             if other_mc.id != choose_mc and distance.euclidean(choose_mc_next_destination, other_mc_next_destination) < 1:
            #                 is_same_destination = True
            #         if not is_same_destination:
            #             phy_action = self.mc_list[choose_mc].next_phy_action
            #             self.env.process(self.mc_list[choose_mc].operate_step_v4(phy_action=phy_action))
            # Phương án 2
            #     if np.argmin(arr_active_mc) == 0:
            #         # b = len(optimizer.list_request)
            #         # if b != a:
            #             arr_q_max = []
            #             for id, mc in enumerate(self.mc_list):
            #                 if first_step == 0:
            #                     temp = (optimizer.list_request[0])["id"]
            #                     mc.state = self.check_cluster(id_node=(optimizer.list_request[0])["id"])
            #                 result = mc.update_q_table(net=self, optimizer=optimizer, time_stem=t)
            #                 if mc.is_active:
            #                     arr_q_max.append(0)
            #                 else:
            #                     arr_q_max.append(result)
            #
            #             first_step = 10
            #             if arr_q_max[np.argmax(arr_q_max)] != 0:
            #                 choose_mc = np.argmax(arr_q_max)
            #                 phy_action = self.mc_list[choose_mc].next_phy_action
            #                 self.env.process(self.mc_list[choose_mc].operate_step_v4(phy_action=phy_action))
            #                 a = b
            # Phuowng an 3
            # arr_q_max = []
            # for id, mc in enumerate(self.mc_list):
            #     result = mc.update_q_table(net=self, optimizer=optimizer, time_stem=t)
            #     if mc.is_active:
            #         arr_q_max.append(0)
            #     else:
            #         arr_q_max.append(result)
            #
            # choose_mc = np.argmax(arr_q_max)
            # is_same_destination = False
            # for other_mc in self.mc_list:
            #     choose_mc_next_destination = (self.mc_list[choose_mc].next_phy_action[0], self.mc_list[choose_mc].next_phy_action[1])
            #     other_mc_next_destination = (other_mc.cur_phy_action[0], other_mc.cur_phy_action[1])
            #     if other_mc.id != choose_mc and distance.euclidean(choose_mc_next_destination, other_mc_next_destination) < 1:
            #         is_same_destination = True
            # if not is_same_destination:
            #     phy_action = self.mc_list[choose_mc].next_phy_action
            #     self.env.process(self.mc_list[choose_mc].operate_step_v4(phy_action=phy_action))
            # self.env.timeout(50)

            # if self.alive == 0 or self.env.now >= self.max_time:
            if self.alive == 0:
                break
        return

    def delete_request(self, id_cluster, optimizer):
        # for request in optimizer.list_request:
        #     for id_node in self.network_cluster_id_node[id_cluster]:
        #         if id_node == request['id']:
        #             optimizer.list_request.remove({'id': id_node})
        #             return
        for i, item in enumerate(optimizer.list_request):
            for id_node in self.network_cluster_id_node[id_cluster]:
                if item['id'] == id_node:
                    del optimizer.list_request[i]
                    break

    def check_cluster(self, id_node):
        for id_cluster, cluster in enumerate(self.network_cluster_id_node):
            for node_cluster in cluster:
                if node_cluster == id_node:
                    return id_cluster

    # If any target dies, value is set to 0
    def check_targets(self):
        return min(self.targets_active)

    def check_nodes(self):
        tmp = 0
        for node in self.listNodes:
            if node.status == 0:
                tmp += 1
        return tmp

    def avg_network(self):
        sum = 0
        for node in self.listNodes:
            sum += node.energy
        return sum / len(self.listNodes)

    def min_node(self):
        id_node_min = -1
        min_energy = 1000000000
        for id, node in enumerate(self.listNodes):
            if node.energy < min_energy:
                min_energy = node.energy
                id_node_min = id
        return id_node_min
