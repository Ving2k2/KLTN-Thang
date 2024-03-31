import copy
import sys
import os
from datetime import datetime

import yaml
from scipy.spatial import distance

from optimizer.q_learning_heuristic import Q_learningv2
from physical_env.mc.MobileCharger import MobileCharger

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO


def log(net, mcs):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        # print(net.env.now, net.check_nodes(), net.min_node())
        for node in net.listNodes:
            if (distance.euclidean(node.location, net.baseStation.location) < 27):
                print(node.id)
        yield net.env.timeout(1.0)

netIO = NetworkIO("C:\\Users\\HT-Com\\PycharmProjects\\multi_agent_rl_wrsn\\physical_env\\network\\network_scenarios\\hanoi1000n50.yaml")
env, net = netIO.makeNetwork()
with open("C:\\Users\\HT-Com\\PycharmProjects\\multi_agent_rl_wrsn\\physical_env\\mc\\mc_types\\default.yaml", 'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(1)]
q_learning = Q_learningv2(net=net, nb_action=len(net.listNodes))
# time_start = datetime.now().timestamp()
net.mc_list = mcs
x = env.process(net.operate(optimizer=q_learning))
env.process(log(net, None))
env.run(until=x)
# print(datetime.now().timestamp() - time_start)