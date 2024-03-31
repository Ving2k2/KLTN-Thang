import sys
import os
import time
import copy
from datetime import datetime

import yaml

from optimizer.q_learning_heuristic import Q_learningv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger

def log(net, mcs):
    # If you want to print something, just put it here. Do not revise the core code.
    while True:
        print(net.env.now, mc.energy, net.min_node())
        yield net.env.timeout(1.0)

netIO = NetworkIO("C:\\Users\\HT-Com\\PycharmProjects\\multi_agent_rl_wrsn\\physical_env\\network\\network_scenarios\\hanoi1000n50.yaml")
env, net = netIO.makeNetwork()

with open("C:\\Users\\HT-Com\\PycharmProjects\\multi_agent_rl_wrsn\\physical_env\\mc\\mc_types\\default.yaml", 'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(3)]

for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id
    mc.log = [net.baseStation.location[0], net.baseStation.location[1], 0]

q_learning = Q_learningv2(net=net, nb_action=len(net.listNodes))
net.mc_list = mcs
# mc0_process = env.process(mcs[0].operate_step([0.7 * (net.frame[1] - net.frame[0]) + net.frame[0], 0.8 * (net.frame[3] - net.frame[2]) + net.frame[2], 75]))
# mc1_process = env.process(mcs[1].operate_step([0.7 * (net.frame[1] - net.frame[0]) + net.frame[0], 0.8 * (net.frame[3] - net.frame[2]) + net.frame[2], 50]))
# mc2_process = env.process(mcs[2].operate_step([0.35 * (net.frame[1] - net.frame[0]) + net.frame[0], 0.6 * (net.frame[3] - net.frame[2]) + net.frame[2], 100]))
net_process = env.process(net.operate(optimizer=q_learning))

general_process = net_process

time_start = datetime.now().timestamp()
for mc in mcs:
    env.process(log(net, mc))
env.run(until = net_process)
#env.run(until = general_process)
print(datetime.now().timestamp() - time_start)