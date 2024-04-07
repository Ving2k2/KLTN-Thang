import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import copy


# from torch.testing._internal.common_device_type import ops


class MobileCharger:
    def __init__(self, location, mc_phy_spe):
        """
        The initialization for a MC.
        :param env: the time management system of this MC
        :param location: the initial coordinate of this MC, usually at the base station
        """
        self.chargingTime = 0
        self.env = None
        self.net = None
        self.id = None
        self.cur_phy_action = [500, 500, 0]
        self.location = np.array(location)
        self.energy = mc_phy_spe['capacity']
        self.capacity = mc_phy_spe['capacity']

        self.alpha = mc_phy_spe['alpha']
        self.beta = mc_phy_spe['beta']
        self.threshold = mc_phy_spe['threshold']
        self.velocity = mc_phy_spe['velocity']
        self.pm = mc_phy_spe['pm']
        self.chargingRate = 0
        self.chargingRange = mc_phy_spe['charging_range']
        self.epsilon = mc_phy_spe['epsilon']
        self.status = 1
        self.checkStatus()
        self.cur_action_type = "deactive"
        self.connected_nodes = []
        self.incentive = 0
        self.end = self.location
        self.start = self.location
        self.state = 30
        self.q_table = []
        self.next_phy_action = [500, 500, 0]
        self.save_state = []
        self.e = mc_phy_spe['e']
        self.is_active = False
        self.is_self_charge = False
        self.is_stand = False
        self.current = self.location
        self.end_time = 0
        self.moving_time = 0
        self.arrival_time = 0
        self.e_move = mc_phy_spe['velocity']

    def charge_step(self, t):
        """
        The charging process to nodes in 'nodes' within simulateTime
        :param nodes: the set of charging nodes
        :param t: the status of MC is updated every t(s)
        """
        for node in self.connected_nodes:
            node.charger_connection(self)

        # print("MC " + str(self.id) + " " + str(self.energy) + " Charging", self.location, self.energy, self.chargingRate)
        yield self.env.timeout(t)
        self.energy = self.energy - self.chargingRate * t
        self.cur_phy_action[2] = max(0, self.cur_phy_action[2] - t)
        for node in self.connected_nodes:
            node.charger_disconnection(self)  # ???
        self.chargingRate = 0
        return

    def chargev2(self, net):
        for nd in net.listNodes:
            p = nd.charge(mc=self)
            self.energy -= p

    def update_location(self):
        self.current = self.get_location()
        self.energy -= self.e_move

    def get_location(mc):
        d = distance.euclidean(mc.start, mc.end)
        time_move = d / mc.velocity
        if time_move == 0:
            return mc.current
        elif distance.euclidean(mc.current, mc.end) < 10 ** -3:
            return mc.end
        else:
            x_hat = (mc.end[0] - mc.start[0]) / time_move + mc.current[0]
            y_hat = (mc.end[1] - mc.start[1]) / time_move + mc.current[1]
            if (mc.end[0] - mc.current[0]) * (mc.end[0] - x_hat) < 0 or (
                    (mc.end[0] - mc.current[0]) * (mc.end[0] - x_hat) == 0 and (mc.end[1] - mc.current[1]) * (
                    mc.end[1] - y_hat) <= 0):
                return mc.end
            else:
                return x_hat, y_hat

    # def charge(self, chargingTime):
    #     tmp = chargingTime
    #     self.chargingTime = tmp
    #     self.connected_nodes = []
    #     for node in self.net.listNodes:
    #         if euclidean(node.location, self.location) <= self.chargingRange:
    #             self.connected_nodes.append(node)
    #     while True:
    #         if tmp == 0:
    #             break
    #         if self.status == 0:
    #             self.cur_phy_action[2] = 0
    #             yield self.env.timeout(tmp)
    #             break
    #         span = min(tmp, 1.0)
    #         if self.chargingRate != 0:
    #             span = min(span, (self.energy - self.threshold) / self.chargingRate)
    #         yield self.env.process(self.charge_step(t=span))
    #         tmp -= span
    #         self.chargingTime = tmp
    #         self.checkStatus()
    #     return

    def move_step(self, vector, t):
        yield self.env.timeout(t)
        self.location = self.location + vector
        self.energy -= self.pm * t * self.velocity

    def move(self, destination):
        moving_time = euclidean(destination, self.location) / self.velocity
        # self.end_time = moving_time
        self.arrival_time = moving_time
        moving_vector = destination - self.location
        total_moving_time = moving_time
        while True:
            if moving_time <= 0:
                break
            if self.status == 0:
                yield self.env.timeout(moving_time)
                break
            moving_time = euclidean(destination, self.location) / self.velocity
            # print("MC " + str(self.id) + " " + str(self.energy) + " Moving from", self.location, "to", destination)
            span = min(min(moving_time, 1.0), (self.energy - self.threshold) / (self.pm * self.velocity))
            # span = 1
            yield self.env.process(self.move_step(moving_vector / total_moving_time * span, t=span))
            moving_time -= span
            self.checkStatus()
        # print("energy after moving", self.energy)
        return self.arrival_time

    def move_time(self, destination):
        moving_time = euclidean(destination, self.location) / self.velocity
        self.arrival_time = moving_time
        return self.arrival_time

    def recharge(self):
        print("energy of charger is low, need to come back base station and re-charge")
        if euclidean(self.location, self.net.baseStation.location) <= self.epsilon:
            self.location = copy.deepcopy(self.net.baseStation.location)
            self.energy = self.capacity
        self.is_self_charge = True
        yield self.env.timeout(0)

    def update_q_table(self, optimizer, net, time_stem):
        result = optimizer.update_v2(self, net, time_stem)
        self.q_table = result[1]
        self.next_phy_action = []
        self.next_phy_action = [result[2][0], result[2][1], result[3]]
        return result[0]

    # def operate_step_v4(self, phy_action):
    #     # if phy_action[2] != 0:
    #     #     print("MC #", self.id, "sent to", phy_action[0], phy_action[1], "and charge in", phy_action[2])
    #     destination = np.array([phy_action[0], phy_action[1]])
    #     chargingTime = phy_action[2]
    #
    #     usedEnergy = euclidean(destination, self.location) * self.pm
    #     tmp = 0
    #     for node in self.net.listNodes:
    #         dis = euclidean(destination, node.location)
    #         if dis <= self.chargingRange and node.status == 1:
    #             tmp += self.alpha / (dis + self.beta) ** 2
    #     usedEnergy += tmp * chargingTime
    #     usedEnergy += euclidean(destination, self.net.baseStation.location) * self.pm
    #
    #     if (not self.is_active) and usedEnergy > self.energy - self.threshold - self.capacity / 200.0:
    #         self.is_active = True
    #         self.cur_phy_action = phy_action
    #         self.cur_action_type = "moving"
    #         yield self.env.process(self.move(destination=self.net.baseStation.location))
    #         yield self.env.process(self.recharge())
    #         yield self.env.process(self.move(destination=destination))
    #         self.cur_action_type = "charging"
    #         yield self.env.process(self.charge(chargingTime=chargingTime))
    #         self.cur_action_type = "deactive"
    #         self.is_active = False
    #         return
    #     if not self.is_active and (destination[0] != 500 and destination[1] != 500):
    #         self.is_active = True
    #         self.cur_phy_action = phy_action
    #         self.cur_action_type = "moving"
    #         yield self.env.process(self.move(destination=destination))
    #         self.cur_action_type = "charging"
    #         yield self.env.process(self.charge(chargingTime=chargingTime))
    #         self.cur_action_type = "deactive"
    #         self.is_active = False
    #         return
    #
    # def operate_step_v3(self, net, time_stem, optimizer):
    #     if ((not self.is_active) and self.cur_action_type == "deactive") or abs(time_stem - self.end_time) < 1:
    #         if optimizer.list_request:
    #             self.cur_action_type = "moving"
    #             new_list_request = []
    #             for request in optimizer.list_request:
    #                 if net.listNodes[request["id"]].energy <= net.listNodes[request["id"]].threshold:
    #                     new_list_request.append(request)
    #                 else:
    #                     net.listNodes[request["id"]].is_request = False
    #             optimizer.list_request = new_list_request
    #             if not optimizer.list_request:
    #                 self.status = 0
    #             self.get_next_location(network=net, time_stem=time_stem, optimizer=optimizer)
    #             print(optimizer.q_table)
    #     else:
    #         if not self.cur_action_type == "deactive":
    #             if self.cur_action_type == "moving":
    #                 # print("moving")
    #                 destination = (self.cur_phy_action[0], self.cur_phy_action[1])
    #                 yield self.env.process(self.move(destination=destination))
    #             elif self.cur_action_type == "charging":
    #                 # print("charging")
    #                 # self.charge(net)
    #                 yield self.env.process(self.charge(chargingTime=self.cur_phy_action[2]))
    #             elif self.cur_action_type == "recharging":
    #                 # print("self charging")
    #                 # self.recharge()
    #                 yield self.env.process(self.recharge())
    #
    #     if self.energy < self.threshold and not self.is_self_charge and self.end != net.baseStation:
    #         # self.start = self.current
    #         self.end = net.baseStation
    #         # self.is_stand = False
    #         # charging_time = self.capacity / self.e_self_charge
    #         moving_time = distance.euclidean(self.location, self.end) / self.velocity
    #         self.end_time = time_stem + moving_time
    #     # self.check_cur_action()

    # def operate_step(self, net, time_stem, optimizer):
    #     self.get_next_location(network=net, time_stem=time_stem, optimizer=optimizer)
    #     destination = np.array([self.cur_phy_action[0], self.cur_phy_action[1]])
    #     # destination = np.array([next_location[0], next_location[1]])
    #     charging_time = self.cur_phy_action[2]
    #     # if (charging_time != 0.0):
    #     #     print("MC " + str(self.id), "destination", destination, "charging_time", charging_time)
    #     if distance.euclidean(destination, net.baseStation.location) < 0.01:
    #         self.cur_action_type = "moving"
    #         yield self.env.process(self.move(destination=self.net.baseStation.location))
    #         yield self.env.process(self.recharge())
    #         self.recharge()
    #
    #     usedEnergy = euclidean(destination, self.location) * self.pm
    #     tmp = 0
    #     for node in self.net.listNodes:
    #         dis = euclidean(destination, node.location)
    #         if dis <= self.chargingRange and node.status == 1:
    #             tmp += self.alpha / (dis + self.beta) ** 2
    #     usedEnergy += tmp * charging_time
    #     usedEnergy += euclidean(destination, self.net.baseStation.location) * self.pm
    #
    #     if usedEnergy > self.energy - self.threshold - self.capacity / 200.0:
    #         # self.cur_phy_action = phy_action
    #         self.cur_action_type = "moving"
    #         yield self.env.process(self.move(destination=self.net.baseStation.location))
    #         yield self.env.process(self.recharge())
    #         yield self.env.process(self.move(destination=destination))
    #         self.cur_action_type = "charging"
    #         self.end_time = charging_time
    #         yield self.env.process(self.charge(chargingTime=charging_time))
    #         return
    #     # self.cur_phy_action = phy_action
    #     if charging_time != 0.0:
    #         self.cur_action_type = "moving"
    #         # self.end_time = self.move_time(self, destination=destination)
    #         yield self.env.process(self.move(destination=destination))
    #         self.cur_action_type = "charging"
    #         self.end_time = charging_time
    #         yield self.env.process(self.charge(chargingTime=charging_time))
    #     self.check_cur_action()

    # def operate_stepv2(self, phy_action, optimizer, time_stem, net):
    #     self.state = len(net.listNodes) - 1
    #     if self.cur_phy_action:
    #         destination = np.array([self.cur_phy_action[0], self.cur_phy_action[1]])
    #         charging_time = self.cur_phy_action[2]
    #     else:
    #         destination = net.baseStation.location
    #         charging_time = 0
    #     usedEnergy = euclidean(destination, self.location) * self.pm
    #     tmp = 0
    #     for node in self.net.listNodes:
    #         dis = euclidean(destination, node.location)
    #         if dis <= self.chargingRange and node.status == 1:
    #             tmp += self.alpha / (dis + self.beta) ** 2
    #     usedEnergy += tmp * charging_time
    #     usedEnergy += euclidean(destination, self.net.baseStation.location) * self.pm
    #
    #     if ((not self.status) and optimizer.list_request) or abs(time_stem - self.end_time) < 1:
    #         self.status = 1
    #         new_list_request = []
    #         for request in optimizer.list_request:
    #             if net.listNodes[request["id"]].energy < net.node[request["id"]].threshold:
    #                 new_list_request.append(request)
    #             else:
    #                 net.listNodes[request["id"]].is_request = False
    #         optimizer.list_request = new_list_request
    #         if not optimizer.list_request:
    #             self.status = 0
    #         self.get_next_location(network=net, time_stem=time_stem, optimizer=optimizer)
    #     else:
    #         if self.status:
    #             if self.cur_action_type == "moving":
    #                 # print("moving")
    #                 yield self.env.process(self.move(destination=destination))
    #             elif self.cur_action_type == "charging":
    #                 # print("charging")
    #                 # self.charge(net)
    #                 yield self.env.process(self.charge(chargingTime=charging_time))
    #             elif self.cur_action_type == "recharging":
    #                 # print("self charging")
    #                 # self.recharge()
    #                 yield self.env.process(self.recharge())
    #             else:
    #                 self.get_next_location(network=net, time_stem=time_stem, optimizer=optimizer)
    #
    #     if (usedEnergy > self.energy - self.threshold - self.capacity / 200.0
    #             and not self.is_self_charge and self.end != net.baseStation.location):
    #         self.cur_phy_action = phy_action
    #         self.cur_action_type = "moving"
    #         self.end = net.baseStation.location
    #         moving_time = distance.euclidean(self.location, net.baseStation.location) / self.velocity
    #         yield self.env.process(self.move(destination=self.net.baseStation.location))
    #         yield self.env.process(self.recharge())
    #         yield self.env.process(self.move(destination=destination))
    #         self.cur_action_type = "charging"
    #         self.end_time = charging_time
    #         yield self.env.process(self.charge(chargingTime=charging_time))
    #         # charging_time = self.capacity / self.e_self_charge
    #         self.end_time = time_stem + moving_time
    #         return
    #         # self.start = self.current
    #         # self.is_stand = False
    #     self.checkStatus()

    def check_cur_action(self):
        if not self.cur_action_type == 'moving':
            self.cur_action_type = 'charging'
        elif not self.cur_action_type == 'charging':
            self.cur_action_type = 'recharging'
        elif not self.cur_action_type == 'recharging':
            self.cur_action_type = 'deactive'

    def get_status(self):
        if not self.is_active:
            return "deactivated"
        if not self.is_stand:
            return "moving"
        if not self.is_self_charge:
            return "charging"
        return "self_charging"

    def checkStatus(self):
        """
        check the status of MC
        """
        if self.energy <= self.threshold:
            # if self.energy <= 0:
            self.status = 0
            self.energy = self.threshold

    def get_next_location(self, network, time_stem, optimizer=None):
        next_location, charging_time = optimizer.update(self, network, time_stem)
        self.start = self.current
        # self.cur_phy_action = [next_location[0], next_location[1], charging_time]
        self.end = next_location
        self.moving_time = distance.euclidean(self.location, self.end) / self.velocity
        self.end_time = time_stem + self.moving_time + charging_time
        print("[Moblie Charger] MC #{} end_time {}".format(self.id, self.end_time))
        self.chargingTime = charging_time
        self.arrival_time = time_stem + self.moving_time
        # print("[Mobile Charger] MC #{} moves to {} in {}s and charges for {}s".format(self.id, self.end, self.moving_time, charging_time))
        # with open(network.mc_log_file, "a") as mc_log_file:
        #     writer = csv.DictWriter(mc_log_file, fieldnames=['time_stamp', 'id', 'starting_point', 'destination_point', 'decision_id', 'charging_time', 'moving_time'])
        #     mc_info = {
        #         'time_stamp' : time_stem,
        #         'id' : self.id,
        #         'starting_point' : self.start,
        #         'destination_point' : self.end,
        #         'decision_id' : self.state,
        #         'charging_time' : charging_time,
        #         'moving_time' : self.moving_time
        #     }
        #     writer.writerow(mc_info)

    def runv2(self, network, time_stem, net=None, optimizer=None):
        # print(self.energy, self.start, self.end, self.current)
        # if ((((not self.is_active) or (self.is_stand and not self.is_self_charge))
        #      and optimizer.list_request)
        #         or abs(time_stem - self.end_time) < 1):
        if (((not self.is_active) and optimizer.list_request) or (np.abs(time_stem - self.end_time) < 1)):
            # temp1 = ((not self.is_active) and len(optimizer.list_request)>0)
            # temp2 = (np.abs(time_stem - self.end_time) < 1)
            # temp3 = (temp1 or temp2)
            self.is_active = True
            new_list_request = []
            for request in optimizer.list_request:
                if net.listNodes[request["id"]].energy < net.listNodes[request["id"]].threshold * 10:
                    new_list_request.append(request)
                else:
                    net.listNodes[request["id"]].is_request = False
            optimizer.list_request = new_list_request
            if not optimizer.list_request:
                self.is_active = False
            self.get_next_location(network=network, time_stem=time_stem, optimizer=optimizer)
        else:
            if self.is_active:
                if not self.is_stand:
                    # print("moving")
                    self.update_location()
                elif not self.is_self_charge:
                    # print("charging")
                    self.chargev2(net)
                else:
                    # print("self charging")
                    self.recharge()
        if self.energy < self.threshold and not self.is_self_charge and self.end != self.net.baseStation.location:
            self.start = self.current
            self.end = self.net.baseStation.location
            self.is_stand = False
            # charging_time = self.capacity / self.e_self_charge
            charging_time = 0
            moving_time = distance.euclidean(self.start, self.end) / self.velocity
            self.end_time = time_stem + moving_time + charging_time
        self.check_state()

    def __str__(self):
        return f"MobileCharger(id='{self.id}', location={self.location}, cur_action_type={self.cur_action_type})"

    def check_state(self):
        if distance.euclidean(self.current, self.end) < 1:
            self.is_stand = True
            self.current = self.end
        else:
            self.is_stand = False
        if distance.euclidean(self.net.baseStation.location, self.end) < 10 ** -3:
            self.is_self_charge = True
        else:
            self.is_self_charge = False
