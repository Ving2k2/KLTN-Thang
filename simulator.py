import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.spatial import distance

from physical_env.mc.MobileCharger import MobileCharger
from physical_env.network.NetworkIO import NetworkIO
from optimizer.q_learning_heuristic import Q_learningv2
import sys
import os
import time
import copy
import networkx as nx
# from networkx.drawing import draw
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def log(net, mcs, q_learning):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        yield net.env.timeout(100)
        # plot_network(net)
        print(net.listNodes[26])
        print_state_net(net, mcs)
        print(q_learning.list_request)
        print("e_RR", net.listNodes[57].energyRR)
        # for id, point in enumerate(net.network_cluster):
        #     print(id, point)
        # arr = []
        # print(net.env.now)
        # for node in net.listNodes:
        #     if (distance.euclidean(node.location, net.baseStation.location) < 27):
        #         arr.append(node.id)
        # if arr:
        #     print(arr)
        # else:
        #     print("nothing")
        # for mc in mcs:
        #     print(net.env.now, net.min_node(), mc.energy, mc.chargingRate)
        # print("energyCS is ", print_arr_energyCS(net.listNodes))
        # print("radius: ", print_arr_radius(net.listNodes))
        # # node_distribution_plot(net)
        # print(len(arr))
        # for point in arr:
        #     print(point[0], point[1])
        # node_safety_circle_plot(net)


def print_state_net(net, mcs):
    print("[Network] Simulating time: {}s, lowest energy node is: id={} energy={:.2f} at {}".format(
        net.env.now, net.min_node(), net.listNodes[net.min_node()].energy, net.listNodes[net.min_node()].location))
    print("energy of node 57 is", net.listNodes[57].energy, "energy of node 6:", net.listNodes[6].energy)
    for mc in net.mc_list:
        # if mc.state >= 0:
        #     print(mc.q_table[mc.state])
        if mc.chargingTime != 0 and mc.get_status() == "charging":
            print("\t\tMC #{} energy:{} is {} at {} state:{}".format(mc.id, mc.energy,
                                                                                         mc.get_status(),
                                                                                         mc.current,
                                                                                         mc.state))
        elif mc.moving_time != 0 and mc.get_status() == "moving":
            print("\t\tMC #{} energy:{} is {} to {} state:{}".format(mc.id, mc.energy,
                                                                                            mc.get_status(), mc.end,
                                                                                            mc.state))
        else:
            print("\t\tMC #{} energy:{} is {} at {} state:{}".format(mc.id, mc.energy, mc.get_status(), mc.current,
                                                                     mc.state))


def plot_network(net):
    G = nx.Graph()
    G.add_node("base_station", pos=tuple(net.baseStation.location))
    for node in net.network_cluster:
        G.add_node(tuple(node))
    # Thêm các đường thẳng từ base_station đến các node theo thứ tự mảng
    for i in range(len(net.network_cluster)):
        # Vẽ đường thẳng từ node trước đến node hiện tại
        if i == 0:
            # Bắt đầu từ base_station
            start_node = "base_station"
        else:
            # Bắt đầu từ node trước
            start_node = tuple(net.network_cluster[i - 1])
        end_node = tuple(net.network_cluster[i])
        G.add_edge(start_node, end_node)

    # Đặt màu cho các node
    node_colors = ["red" for _ in range(len(net.network_cluster))]

    # Tùy chỉnh các thuộc tính vẽ
    nx.draw(G, node_color='red', edge_color='blue', with_labels=False)
    # Vẽ các node
    # nx.draw_nodes(G, net.network_cluster, node_color=node_colors, with_labels=True)
    #
    # # Vẽ các đường thẳng
    # nx.draw_edges(G, with_labels=True)

    # Hiển thị đồ thị
    plt.show()


def draw_sensor_icon(ax, x, y, icon_path, icon_size):
    """
    Vẽ biểu tượng cảm biến tại tọa độ (x, y).

    Parameters:
        ax (matplotlib.axes.Axes): Đối tượng trục của đồ thị.
        x (float): Tọa độ x của trung tâm.
        y (float): Tọa độ y của trung tâm.
        icon_path (str): Đường dẫn đến hình ảnh biểu tượng cảm biến.
        icon_size (float): Kích thước của biểu tượng.

    Returns:
        None
    """
    icon = plt.imread(icon_path)
    ax.imshow(icon, extent=[x - icon_size / 2, x + icon_size / 2, y - icon_size / 2, y + icon_size / 2])


def plot_circles(nodes, arr_name_nodes, radiuses, base_station_icon_path):
    """
    Vẽ các hình tròn và các điểm giao nhau trên đồ thị.

    Parameters:
        nodes (list): Tập hợp các điểm trung tâm hình tròn.
        arr_ten_nodes (list): Tập hợp các tên hình tròn.
        radiuses (list of floats): Bán kính của từng hình tròn.
        base_station_icon_path (str): Đường dẫn đến hình ảnh biểu tượng trạm cơ sở.

    Returns:
        list: Tập hợp các điểm giao nhau.
    """
    all_intersections = []
    for centers in arr_name_nodes:
        circles = []
        for i in centers:
            circle = Point(nodes[i]).buffer(radiuses[i])
            circles.append(circle)
            draw_sensor_icon(plt.gca(), nodes[i][0], nodes[i][1], 'images/sensor.png', 20)

        # Vẽ biểu tượng trạm cơ sở
        base_station_x, base_station_y = base_station
        draw_sensor_icon(plt.gca(), base_station_x, base_station_y, base_station_icon_path, 60)

        # for circle in circles:
        #     plt.plot(circle.exterior.xy[0], circle.exterior.xy[1], color='b', linewidth=0.5)

        intersections = circles[0]
        for circle in circles[1:]:
            intersections = intersections.intersection(circle)

        if isinstance(intersections, Polygon) and not intersections.is_empty:
            centroid = intersections.centroid
            plt.plot(centroid.x, centroid.y, marker='o', color='r', markersize=3)
        else:
            plt.fill(circles[0].exterior.xy[0], circles[0].exterior.xy[1], color='orange', alpha=0.3)
            plt.plot(nodes[centers[0]][0], nodes[centers[0]][1], marker='o', color='r', markersize=3)
        all_intersections.append(intersections)
    return all_intersections


def draw_battery(ax, x, y, width, height, charge_percentage):
    ax.add_patch(Rectangle((x, y), width, height, edgecolor='black', facecolor='none'))
    charge_height = height * charge_percentage / 100

    if charge_percentage <= 20:
        color = 'red'
    elif charge_percentage > 20 and charge_percentage <= 75:
        color = 'yellow'
    else:
        color = 'lime'

    ax.add_patch(Rectangle((x, y), width, charge_height, edgecolor='none', facecolor=color))

    bolt_x = x + width / 2
    bolt_y = y + charge_height / 2
    ax.text(bolt_x, bolt_y, '⚡', fontsize=20, color='blue', va='center', ha='center')


networkIO = NetworkIO("./physical_env/network/network_scenarios/hanoi1000n50.yaml")
env, net = networkIO.makeNetwork()

with open("D:\Documents\Senior-year\multi_agent_rl_wrsn_Double_Q\physical_env\mc\mc_types\default.yaml",
          'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(3)]
print(mc for mc in mcs)
for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id
    mc.cur_phy_action = [net.baseStation.location[0], net.baseStation.location[1], 0]
q_learning = Q_learningv2(net=net, nb_action=37, alpha=0.1, q_gamma=0.1)

# Node:   50    100     150     200
# Center: 37    57      70      75
# Time:   21.9  5.5
print("start program")
net.mc_list = mcs
x = env.process(net.operate(optimizer=q_learning))
env.process(log(net, mcs, q_learning))
env.run(until=x)
