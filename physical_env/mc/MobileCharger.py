import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import copy

class MobileCharger:
    def __init__(self, location, mc_phy_spe):
        """
        The initialization for a MC.
        :param env: the time management system of this MC
        :param location: the initial coordinate of this MC, usually at the base station
        """
        self.env = None
        self.net = None
        self.id = None
        self.cur_phy_action = None
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
        self.end_time = 0
        self.end = (0,0)
        self.arrival_time = -1
        self.start = self.location
        self.state = 0
        self.is_self_charge = False
        self.moving_time = 0

    def charge_step(self, t):
        """
        The charging process to nodes in 'nodes' within simulateTime
        :param nodes: the set of charging nodes
        :param t: the status of MC is updated every t(s)
        """
        for node in self.connected_nodes:
            node.charger_connection(self)

        #print("MC " + str(self.id) + " " + str(self.energy) + " Charging", self.location, self.energy, self.chargingRate)
        yield self.env.timeout(t)
        self.energy = self.energy - self.chargingRate * t
        self.cur_phy_action[2] = max(0, self.cur_phy_action[2] - t)
        for node in self.connected_nodes:
            node.charger_disconnection(self) #???
        self.chargingRate = 0
        return

    def charge(self, chargingTime):
        self.cur_action_type = "charging"
        tmp = chargingTime
        self.chargingTime = tmp
        self.connected_nodes = []
        for node in self.net.listNodes:
            if euclidean(node.location, self.location) <= self.chargingRange:
                self.connected_nodes.append(node)
        while True:
            if tmp == 0:
                break
            if self.status == 0:
                self.cur_phy_action[2] = 0
                yield self.env.timeout(tmp)
                break
            span = min(tmp, 1.0)
            if self.chargingRate != 0:
                span = min(span, (self.energy - self.threshold) / self.chargingRate)
            yield self.env.process(self.charge_step(t=span))
            tmp -= span
            self.chargingTime = tmp
            self.checkStatus()
        return

    def move_step(self, vector, t):
        yield self.env.timeout(t)
        self.location = self.location + vector
        self.energy -= self.pm * t * self.velocity
        

    def move(self, destination):
        moving_time = euclidean(destination, self.location) / self.velocity
        self.end_time = moving_time
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
            #print("MC " + str(self.id) + " " + str(self.energy) + " Moving from", self.location, "to", destination)
            # span = min(min(moving_time, 1.0), (self.energy - self.threshold) / (self.pm * self.velocity))
            span = 1
            yield self.env.process(self.move_step(moving_vector / total_moving_time * span, t=span))
            moving_time -= span
            if euclidean(self.location, destination) < 0.01:
                self.cur_action_type = "charging"
            elif euclidean(self.net.baseStation.location, self.location) < 0.01:
                self.cur_action_type = "recharging"
            self.checkStatus()
        # print("energy after moving", self.energy)
        return self.arrival_time, self.end_time

    def move_time(self, destination):
        moving_time = euclidean(destination, self.location) / self.velocity
        self.arrival_time = moving_time
        return self.arrival_time


    def recharge(self):
        if euclidean(self.location, self.net.baseStation.location) <= self.epsilon:
            self.location = copy.deepcopy(self.net.baseStation.location)
            self.energy = self.capacity
        self.is_self_charge = True
        yield self.env.timeout(0)
    
    def operate_step(self, net, time_stem, optimizer):
        self.get_next_location(network=net, time_stem=time_stem, optimizer=optimizer)
        #print("MC " + str(self.id), "phy_action", phy_action)
        destination = np.array([self.cur_phy_action[0], self.cur_phy_action[1]])
        # destination = np.array([next_location[0], next_location[1]])
        charging_time = self.cur_phy_action[2]
        # phy_action = []
        # phy_action.append(next_location[0])
        # phy_action.append(next_location[1])
        # phy_action.append(charging_time)

        usedEnergy = euclidean(destination, self.location) * self.pm
        tmp = 0
        for node in self.net.listNodes:
            dis = euclidean(destination, node.location)
            if dis <= self.chargingRange and node.status == 1:
                tmp += self.alpha / (dis + self.beta) ** 2
        usedEnergy += tmp * charging_time
        usedEnergy += euclidean(destination, self.net.baseStation.location) * self.pm
        # print("used_energy is:", usedEnergy)
        # print("energy now", self.energy)

        if usedEnergy > self.energy - self.threshold - self.capacity / 200.0:
            # self.cur_phy_action = phy_action
            self.cur_action_type = "moving"
            yield self.env.process(self.move(destination=self.net.baseStation.location))
            yield self.env.process(self.recharge())
            yield self.env.process(self.move(destination=destination))
            self.cur_action_type = "charging"
            self.end_time = charging_time
            yield self.env.process(self.charge(chargingTime=charging_time))
            return
        # self.cur_phy_action = phy_action
        self.cur_action_type = "moving"
        # self.end_time = self.move_time(self, destination=destination)
        yield self.env.process(self.move(destination=destination))
        self.cur_action_type = "charging"
        self.end_time = charging_time
        yield self.env.process(self.charge(chargingTime=charging_time))

    def operate_stepv2(self, phy_action, optimizer, time_stem, net):
        self.state = len(net.listNodes) - 1
        if self.cur_phy_action:
            destination = np.array([self.cur_phy_action[0], self.cur_phy_action[1]])
            charging_time = self.cur_phy_action[2]
        else:
            destination = net.baseStation.location
            charging_time = 0
        usedEnergy = euclidean(destination, self.location) * self.pm
        tmp = 0
        for node in self.net.listNodes:
            dis = euclidean(destination, node.location)
            if dis <= self.chargingRange and node.status == 1:
               tmp += self.alpha / (dis + self.beta) ** 2
        usedEnergy += tmp * charging_time
        usedEnergy += euclidean(destination, self.net.baseStation.location) * self.pm

        if ((not self.status) and optimizer.list_request) or abs(time_stem - self.end_time) < 1:
            self.status = 1
            new_list_request = []
            for request in optimizer.list_request:
                if net.listNodes[request["id"]].energy < net.node[request["id"]].threshold:
                    new_list_request.append(request)
                else:
                    net.listNodes[request["id"]].is_request = False
            optimizer.list_request = new_list_request
            if not optimizer.list_request:
                self.status = 0
            self.get_next_location(network=net, time_stem=time_stem, optimizer=optimizer)
        else:
            if self.status:
                if self.cur_action_type == "moving":
                    # print("moving")
                    yield self.env.process(self.move(destination=destination))
                elif self.cur_action_type == "charging":
                    # print("charging")
                    # self.charge(net)
                    yield self.env.process(self.charge(chargingTime=charging_time))
                elif self.cur_action_type == "recharging":
                    # print("self charging")
                    # self.recharge()
                    yield self.env.process(self.recharge())
                else:
                    self.get_next_location(network=net, time_stem=time_stem, optimizer=optimizer)

        if (usedEnergy > self.energy - self.threshold - self.capacity / 200.0
                and not self.is_self_charge and self.end != net.baseStation.location):
            self.cur_phy_action = phy_action
            self.cur_action_type = "moving"
            self.end = net.baseStation.location
            moving_time = distance.euclidean(self.location, net.baseStation.location) / self.velocity
            yield self.env.process(self.move(destination=self.net.baseStation.location))
            yield self.env.process(self.recharge())
            yield self.env.process(self.move(destination=destination))
            self.cur_action_type = "charging"
            self.end_time = charging_time
            yield self.env.process(self.charge(chargingTime=charging_time))
            # charging_time = self.capacity / self.e_self_charge
            self.end_time = time_stem + moving_time
            return
            # self.start = self.current
            # self.is_stand = False
        self.checkStatus()

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
        # self.location = self.current
        phy_action = [next_location[0], next_location[1], charging_time]
        self.cur_phy_action = phy_action
        self.end = next_location
        self.moving_time = distance.euclidean(self.location, self.end) / self.velocity
        self.end_time = time_stem + self.moving_time + charging_time
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
    def __str__(self):
        return f"MobileCharger(id='{self.id}', location={self.location}, cur_action_type={self.cur_action_type})"
