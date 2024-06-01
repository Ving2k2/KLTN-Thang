import random

import numpy as np
from scipy.spatial import distance

from optimizer.utils import init_function, q_max_function, reward_function
from physical_env.network import Node


class Q_learningv2:
    def __init__(self, init_func=init_function, nb_action=37, alpha=0.5, q_alpha=0.5, q_gamma=0.01,
                 load_checkpoint=False, net=None):
        self.action_list = np.zeros(nb_action + 1)
        self.nb_action = nb_action + 1
        self.q_table = init_func(nb_action=nb_action + 1)
        self.q_table_A = init_func(nb_action = nb_action + 1)
        self.q_table_B = init_func(nb_action = nb_action + 1)
        # self.state = nb_action+1
        self.charging_time = [0.0 for _ in range(nb_action + 1)]
        self.reward = np.asarray([0.0 for _ in range(nb_action + 1)])
        self.reward_max = [0.0 for _ in range(nb_action + 1)]
        self.list_request = []
        self.alpha = alpha
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma
        self.choose_Q_A = False
        self.eps_double_q = 1

    def update_v2(self, mc, network, time_stem, alpha=0.5, gamma=0.5, q_max_func=q_max_function):
        if mc.state == -1:
            return np.max(self.q_table), self.q_table, network.baseStation.location, 0

        self.set_reward(mc=mc, time_stem=time_stem, network=network)
        temp = self.q_max(mc, q_max_func)
        # self.q_table[mc.state] = (1 - self.q_alpha) * self.q_table[mc.state] + self.q_alpha * (
        #             self.reward + self.q_gamma * self.q_max(mc, q_max_func))
        self.q_table[mc.state] = (1 - self.q_alpha) * self.q_table[mc.state] + self.q_alpha * (
            self.reward)
        # print(self.q_table)
        self.choose_next_state_v2(mc, network)
        charging_time = self.charging_time[mc.state]
        # if mc.state == len(self.q_table) - 1:
        #     charging_time = 0
        # print("[Optimizer] MC #{} is sent to point {} (id={}) and charge for {:.2f}s".format(mc.id,
        #                                                                                      self.action_list[mc.state],
        #                                                                                      mc.state, charging_time))
        # print(self.charging_time)
        return np.max(self.q_table), self.q_table, self.action_list[mc.state], charging_time

    def update_double_Q(self, mc, network, time_stem, alpha=0.5, gamma=0.5, q_max_func=q_max_function):
        if not self.list_request:
            return self.action_list[mc.state], 0
        if random.random() > 0.5:
            self.choose_Q_A = True
        else:
            self.choose_Q_A = False
        if self.choose_Q_A:
            self.set_reward(mc=mc, time_stem=time_stem, network=network)
            self.choose_next_state_double_Q(mc, network, self.q_table_A)
            self.q_table_B[mc.state] = (1 - self.q_alpha) * self.q_table_B[mc.state] + self.q_alpha + (
                self.reward + self.q_gamma * self.q_max_double_Q(mc, q_max_func, q_table = self.q_table_A))
            # self.choose_next_state_double_Q(mc, network, self.q_table_B)
        else:
            self.set_reward(mc=mc, time_stem=time_stem, network=network)
            self.choose_next_state_double_Q(mc, network, self.q_table_B)
            self.q_table_A[mc.state] = (1 - self.q_alpha) * self.q_table_A[mc.state] + self.q_alpha * (
                    self.reward + self.q_gamma * self.q_max_double_Q(mc, q_max_func, q_table=self.q_table_B))
            # self.choose_next_state_double_Q(mc, network, self.q_table_A)
        if mc.state == len(self.action_list) - 1:
            charging_time = 0
            # print("chon trung tram co so - node cuoi")
        else:
            print("mc.state now MC #", mc.id, " ", mc.state)
            charging_time = self.charging_time[mc.state]
        # print("[Optimizer] MC #{} is sent to point {} (id={}) and charge for {:.2f}s".format(mc.id,
        #                                                                                      self.action_list[mc.state],
        #                                                                                      mc.state, charging_time))
        # print(self.charging_time)
        if charging_time > 500:
            charging_time = 500
        return self.action_list[mc.state], charging_time


    def update(self, mc, network, time_stem, alpha=0.5, gamma=0.5, q_max_func=q_max_function):
        if not self.list_request:
            return self.action_list[mc.state], 0
        self.set_reward(mc=mc, time_stem=time_stem, network=network)
        self.q_table[mc.state] = (1 - self.q_alpha) * self.q_table[mc.state] + self.q_alpha * (
                self.reward + self.q_gamma * self.q_max(mc, q_max_func))
        print("state have max reward is: ", self.q_table[mc.state][np.argmax(self.q_table[mc.state])], "in state ", np.argmax(self.q_table[mc.state]))
        self.choose_next_state(mc, network)
        if mc.state == len(self.action_list) - 1:
            # charging_time = (mc.capacity - mc.energy) / mc.e_self_charge
            charging_time = 0
        else:
            charging_time = self.charging_time[mc.state]
        # print("[Optimizer] MC #{} is sent to point {} (id={}) and charge for {:.2f}s".format(mc.id,
        #                                                                                      self.action_list[mc.state],
        #                                                                                      mc.state, charging_time))
        # print(self.charging_time)

        return self.action_list[mc.state], charging_time

    def q_max(self, mc, q_max_func=q_max_function):
        return q_max_function(q_table=self.q_table, state=mc.state)

    def q_max_double_Q(self, mc, q_max_func=q_max_function, q_table = None):
        return q_max_function(q_table=q_table, state=mc.state)

    def set_reward(self, mc=None, time_stem=0, network=None):
        first = np.asarray([0.0 for _ in self.action_list], dtype=float)
        second = np.asarray([0.0 for _ in self.action_list], dtype=float)
        third = np.asarray([0.0 for _ in self.action_list], dtype=float)
        for index, row in enumerate(self.q_table):
            temp = reward_function(network=network, mc=mc, q_learning=self, state=index, time_stem=time_stem)
            first[index] = temp[0]
            second[index] = temp[1]
            third[index] = temp[2]
            self.charging_time[index] = temp[3]
        first = first / np.sum(first)
        second = second / np.sum(second)
        third = third / np.sum(third)
        # self.reward = first + third
        self.reward = first * 2 + second + third
        self.reward_max = list(zip(first, second, third))

    def choose_next_state(self, mc, network):
        # next_state = np.argmax(self.q_table[mc.state])
        if mc.energy < mc.threshold:  # 10
            mc.state = len(self.q_table) - 1
            print('[Optimizer] MC #{} energy is running low ({:.2f}), and needs to rest!'.format(mc.id, mc.energy))
        else:
            mc.state = np.argmax(self.q_table[mc.state])
            if mc.state == len(self.q_table) - 1:
                mc.state = random.randrange(len(self.q_table)-1)
            # print(self.reward_max[mc.state])
            # print(self.action_list[mc.state])

    def choose_next_state_double_Q(self, mc, network, q_table):
        # next_state = np.argmax(self.q_table[mc.state])
        if mc.energy < mc.threshold:  # 10
            mc.state = len(q_table) - 1
            print('[Optimizer] MC #{} energy is running low ({:.2f}), and needs to rest!'.format(mc.id, mc.energy))
        elif random.random() > self.eps_double_q:
            mc.state = random.randint(0, self.nb_action - 2)
            self.eps_double_q += 0.01
        else:
            mc.state = np.argmax(q_table[mc.state])
            while mc.state >= len(q_table) - 1:
                mc.state = random.randrange(len(self.q_table) - 2)

    def choose_next_state_v2(self, mc, network):
        mc.state = np.nanargmax(self.q_table[mc.state])

    # def net_partition(self, net=None, net_clustering_func=network_clustering):
    #     # self.action_list = net_clustering_func(self, network=net, nb_cluster=self.nb_action)
    #     self.action_list = net_clustering_func(self, network=net)
    #     return self.action_list
