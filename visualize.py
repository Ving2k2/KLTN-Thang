import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from scipy.spatial.distance import euclidean

from physical_env.network.Package import Package
from shapely.geometry import Polygon, Point

from physical_env.network.NetworkIO import NetworkIO
from physical_env.network.Network import Network
from optimizer.utils import find_set_of_interecting_circles, remove_arr_of_set, remove_common_elements2

time_start = datetime.now().timestamp()
time = 0
def log(net):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        yield net.env.timeout(10.0)
        print(net.env.now, net.check_nodes(), net.min_node())
        # print("energyCS is ", print_arr_energyCS(net.listNodes))
        # print("radius: ", print_arr_radius(net.listNodes))
        # node_distribution_plot(net)
        arr = plot_circlesv3(net)
        print(len(arr))
        for point in arr:
            print(point[0], point[1])
        # node_safety_circle_plot(net)


def node_distribution_plot(network):
    plt.clf()
    for i, node in enumerate(network.listNodes):
        plt.plot(node.location[0], node.location[1], "o", color="blue")
        plt.annotate(int(node.energy), (node.location[0], node.location[1]), xytext=(0, -10),
                     textcoords="offset points", size=5)
    # plt.figure(figsize=(5, 5))
    name_fig = "./fig/node_energy/{}_{}.png".format("node_distribution", net.env.now)
    plt.savefig(name_fig)
    
def node_safety_circle_plot(network):
    plt.clf()
    circles = []
    for id, node in enumerate(network.listNodes):
        # Create circle
        circles.append(Point(node.location[0], node.location[1]).buffer(node.radius))
        plt.plot(node.location[0], node.location[1], "o", color="black")

        for circle in circles:
            plt.plot(circle.exterior.xy[0], circle.exterior.xy[1], color='b', linewidth=0.5)
            # plt.plot(node.location[0], node.location[1], color='black')
        # plt.figure(figsize=(4, 4))
    name_fig = "./fig/{}.png".format("node_safety_circle_radius")
    plt.savefig(name_fig)
    return

def plot_circlesv3(network):
    nodes = network.listNodes
    location_nodes = []
    radius_nodes = []
    for node in nodes:
        radius_nodes.append(node.radius)
        location_nodes.append(node.location)

    set_arr_interecting_circles = find_set_of_interecting_circles(location_nodes, radius_nodes)
    set_arr_interecting_circles = remove_arr_of_set(set_arr_interecting_circles)
    set_arr_interecting_circles = remove_common_elements2(set_arr_interecting_circles, location_nodes)
    #
    arr_name_nodes = set_arr_interecting_circles

    all_intersections = []
    charging_pos = []
    for centers in arr_name_nodes:
        circles = []
        for i in centers:
            circle = Point(location_nodes[i]).buffer(radius_nodes[i])
            circles.append(circle)
            plt.plot(location_nodes[i][0], location_nodes[i][1], marker='o', color='black', markersize=1)

        for circle in circles:
            plt.plot(circle.exterior.xy[0], circle.exterior.xy[1], color='b', linewidth=0.5)

        intersections = circles[0]
        for circle in circles[1:]:
            intersections = intersections.intersection(circle)

        if isinstance(intersections, Polygon) and not intersections.is_empty:
            centroid = intersections.centroid
            plt.plot(centroid.x, centroid.y, marker='o', color='r', markersize=3)
            plt.fill(intersections.exterior.xy[0], intersections.exterior.xy[1], color='g', alpha=0.5)
            charging_pos.append((centroid.x, centroid.y))
        else:
            plt.fill(circles[0].exterior.xy[0], circles[0].exterior.xy[1], color='purple', alpha=0.3)
            plt.plot(location_nodes[centers[0]][0], location_nodes[centers[0]][1], marker='o', color='r', markersize=3)
            charging_pos.append((location_nodes[centers[0]][0], location_nodes[centers[0]][1]))
        all_intersections.append(intersections)

    name_fig = "./fig/{}.png".format("charging_pos")
    plt.savefig(name_fig)
    return charging_pos
    # return

networkIO = NetworkIO("./physical_env/network/network_scenarios/hanoi1000n50.yaml")
env, net = networkIO.makeNetwork()
print("start program")

net.operate(t=1)
x = env.process(net.operate())
env.process(log(net))
env.run(until=x)