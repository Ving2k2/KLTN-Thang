import yaml
from scipy.spatial import distance

from physical_env.mc.MobileCharger import MobileCharger
from physical_env.network.NetworkIO import NetworkIO
from optimizer.q_learning_heuristic import Q_learningv2
import sys
import os
import time
import copy
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def log(net, mc):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        yield net.env.timeout(100)
        # arr = []
        # print(net.env.now)
        # for node in net.listNodes:
        #     if (distance.euclidean(node.location, net.baseStation.location) < 27):
        #         arr.append(node.id)
        # if arr:
        #     print(arr)
        # else:
        #     print("nothing")
        print(net.env.now, net.min_node(), mc.energy, mc.chargingRate)
        # print("energyCS is ", print_arr_energyCS(net.listNodes))
        # print("radius: ", print_arr_radius(net.listNodes))
        # # node_distribution_plot(net)
        # print(len(arr))
        # for point in arr:
        #     print(point[0], point[1])
        # node_safety_circle_plot(net)



networkIO = NetworkIO("./physical_env/network/network_scenarios/hanoi1000n50.yaml")
env, net = networkIO.makeNetwork()
with open("C:\\Users\\HT-Com\\PycharmProjects\\multi_agent_rl_wrsn\\physical_env\\mc\\mc_types\\default.yaml", 'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(3)]
print(mc for mc in mcs)
for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id
    mc.cur_phy_action = [net.baseStation.location[0], net.baseStation.location[1], 0]
q_learning = Q_learningv2(net=net, nb_action=len(net.listNodes))

print("start program")
net.mc_list = mcs
x = env.process(net.operate(optimizer=q_learning))
for mc in net.mc_list:
    env.process(log(net, mc))
env.run(until = x)
