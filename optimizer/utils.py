#use previous code in github
# Libraries
from timeit import repeat
from scipy.spatial import distance
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry import Polygon, Point
import numpy as np
from math import sqrt
from itertools import combinations
from matplotlib.patches import Rectangle
import math
import pickle

# Modules
from optimizer import parameter as para
# from simulator.network import parameter as para
from physical_env.network.utils import find_receiver
from physical_env.network import Node

def q_max_function(q_table, state):
    temp = [max(row) if index != state else -float("inf") for index, row in enumerate(q_table)]
    return np.asarray(temp)

def reward_function(network, mc, q_learning, state, time_stem):
    alpha = q_learning.alpha
    charging_time = get_charging_time(network, mc, q_learning, time_stem=time_stem, state=state, alpha=alpha)
    w, nb_target_alive = get_weight(network, mc, q_learning, state, charging_time)
    p = get_charge_per_sec(network, q_learning, state)
    p_hat = p / np.sum(p)
    E = np.asarray([network.listNodes[request["id"]].energy for request in q_learning.list_request])
    e = np.asarray([request["energyCS"] for request in q_learning.list_request])
    second = nb_target_alive / len(network.listTargets)
    third = np.sum(w * p_hat)
    first = np.sum(e * p / E)
    return first, second, third, charging_time


def init_function(nb_action=82):
    return np.zeros((nb_action + 1, nb_action + 1), dtype=float)

def get_weight(net, mc, q_learning, action_id, charging_time):
    p = get_charge_per_sec(net, q_learning, action_id)
    all_path = get_all_path(net)
    time_move = distance.euclidean(q_learning.action_list[mc.state],
                                   q_learning.action_list[action_id]) / mc.velocity
    list_dead = []
    w = [0 for _ in q_learning.list_request]
    for request_id, request in enumerate(q_learning.list_request):
        temp = (net.listNodes[request["id"]].energy - time_move * request["energyCS"]) + (
                p[request_id] - request["energyCS"]) * charging_time
        if temp < 0:
            list_dead.append(request["id"])
    for request_id, request in enumerate(q_learning.list_request):
        nb_path = 0
        for path in all_path:
            if request["id"] in path:
                nb_path += 1
        w[request_id] = nb_path
    total_weight = sum(w) + len(w) * 10 ** -3
    w = np.asarray([(item + 10 ** -3) / total_weight for item in w])
    nb_target_alive = 0
    for path in all_path:
        if para.base in path and not (set(list_dead) & set(path)):
            nb_target_alive += 1
    return w, nb_target_alive


def get_path(net, sensor_id):
    # print("sensor ne:", sensor_id)
    path = [sensor_id]
    if distance.euclidean(net.listNodes[sensor_id].location, para.base) <= net.listNodes[sensor_id].com_range:
        path.append(para.base)
    else:
        receive_id = find_receiver(node=net.listNodes[sensor_id])
        # receive_id = net.listNodes[sensor_id].find_receiver()
        # print('receive_id ne', receive_id)
        if receive_id != -1:
            path.extend(get_path(net, receive_id))
    return path


def get_all_path(net):
    list_path = []
    for sensor_id, node in enumerate(net.listNodes):
        list_path.append(get_path(net, sensor_id))
    return list_path


def get_charge_per_sec(net, q_learning, state):
    arr = []
    for request in q_learning.list_request:
        arr.append((para.alpha) / (distance.euclidean(net.listNodes[request["id"]].location, q_learning.action_list[state]) + para.beta) ** 2)
    # return np.asarray(
    #     [para.alpha / (distance.euclidean(net.listNodes[request["id"]].location, q_learning.action_list[state]) + para.beta) ** 2 for
    #      request in q_learning.list_request])
    return arr


def get_charging_time(network=None, mc=None, q_learning=None, time_stem=0, state=None, alpha=0.1):
    # request_id = [request["id"] for request in network.mc.list_request]
    time_move = distance.euclidean(mc.location, q_learning.action_list[state]) / mc.velocity
    energy_min = network.listNodes[0].threshold + alpha * network.listNodes[0].capacity
    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    for node in network.listNodes:
        d = distance.euclidean(q_learning.action_list[state], node.location)
        p = para.alpha / (d + para.beta) ** 2
        p1 = 0
        for other_mc in network.mc_list:
            if other_mc.id != mc.id and other_mc.cur_action_type == "charging":
                d = distance.euclidean(other_mc.location, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2) * (other_mc.end_time - time_stem) #endtime wrong in this code
            elif other_mc.id != mc.id and other_mc.cur_action_type() == "moving" and other_mc.state != len(
                    q_learning.q_table) - 1:
                d = distance.euclidean(other_mc.end, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2) * (other_mc.end_time - other_mc.arrival_time)
        if node.energy - time_move * node.energyCS + p1 < energy_min and p - node.energyCS > 0:
            s1.append((node.id, p, p1))
        if node.energy - time_move * node.energyCS + p1 > energy_min and p - node.energyCS < 0:
            s2.append((node.id, p, p1))
    t = []

    for index, p, p1 in s1:
        t.append((energy_min - network.listNodes[index].energy + time_move * network.listNodes[index].energyCS - p1) / (
                p - network.listNodes[index].energyCS))
    for index, p, p1 in s2:
        t.append((energy_min - network.listNodes[index].energy + time_move * network.listNodes[index].energyCS - p1) / (
                p - network.listNodes[index].energyCS))
    dead_list = []
    for item in t:
        nb_dead = 0
        for index, p, p1 in s1:
            temp = network.listNodes[index].energy - time_move * network.listNodes[index].energyCS + p1 + (
                    p - network.listNodes[index].energyCS) * item
            if temp < energy_min:
                nb_dead += 1
        for index, p, p1 in s2:
            temp = network.listNodes[index].energy - time_move * network.listNodes[index].energyCS + p1 + (
                    p - network.listNodes[index].energyCS) * item
            if temp < energy_min:
                nb_dead += 1
        dead_list.append(nb_dead)
    if dead_list:
        arg_min = np.argmin(dead_list)
        return t[arg_min]
    return 0

def network_clustering(optimizer, network=None, nb_cluster=81):
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

        intersections = circles[0]
        for circle in circles[1:]:
            intersections = intersections.intersection(circle)

        if isinstance(intersections, Polygon) and not intersections.is_empty:
            centroid = intersections.centroid
            charging_pos.append((centroid.x, centroid.y))
        else:
            charging_pos.append((location_nodes[centers[0]][0], location_nodes[centers[0]][1]))
        all_intersections.append(intersections)

    charging_pos_each_node = []
    for index, cluster in enumerate(arr_name_nodes):
        for i in range(len(nodes)):
            for node_id in cluster:
                if node_id == i:
                    charging_pos_each_node.append(charging_pos[index])

    charging_pos_each_node.append(network.baseStation.location)
    # name_fig = "./fig/{}.png".format("charging_pos")
    # plt.savefig(name_fig)
    return charging_pos_each_node
        # arr_charge_pos.append(para.depot)
        # # print(charging_pos, file=open('log/centroid.txt', 'w'))
        # node_distribution_plot(network=network, charging_pos=arr_charge_pos)
        # network_plot(network=network, charging_pos=arr_charge_pos)

def network_clustering_v2(optimizer, network=None, nb_cluster=81):
    X = []
    Y = []
    min_node = 1000
    for node in network.listNodes:
        node.set_check_point(200)
        if node.avg_energy != 0:
            min_node = min(min_node, node.avg_energy)
    for node in network.listNodes:
        repeat = int(node.avg_energy / min_node)
        for _ in range(repeat):
            X.append(node.location)
            Y.append(node.avg_energy)
    X = np.array(X)
    Y = np.array(Y)
    # print(Y)
    d = np.linalg.norm(Y)
    Y = Y / d
    # print(d)
    # print(Y)
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(X)
    charging_pos = []
    for pos in kmeans.cluster_centers_:
        charging_pos.append((int(pos[0]), int(pos[1])))
    charging_pos.append(para.depot)
    # print(charging_pos, file=open('log/centroid.txt', 'w'))
    # node_distribution_plot(network=network, charging_pos=charging_pos)
    network_plot(network=network, charging_pos=charging_pos)
    return charging_pos


def node_distribution_plot(network, charging_pos):
    x_node = []
    y_node = []
    c_node = []
    for node in network.listNodes:
        x_node.append(node.location[0])
        y_node.append(node.location[1])
        c_node.append(node.avg_energy)
    x_centroid = []
    y_centroid = []
    plt.hist(c_node, bins=100)
    plt.savefig('fig/node_distribution.png')


def network_plot(network, charging_pos):
    x_node = []
    y_node = []
    c_node = []
    for node in network.listNodes:
        x_node.append(node.location[0])
        y_node.append(node.location[1])
        c_node.append(node.avg_energy)
    x_centroid = []
    y_centroid = []
    for centroid in charging_pos:
        x_centroid.append(centroid[0])
        y_centroid.append(centroid[1])
    c_node = np.array(c_node)
    d = np.linalg.norm(c_node)
    c_node = c_node / d * 80
    plt.scatter(x_node, y_node, s=c_node)
    plt.scatter(x_centroid, y_centroid, c='red', marker='^')
    plt.savefig('fig/network_plot.png')


#my own code
def remove_element_in_arr(arr, key):
  for i in range(len(arr)):
    if arr[i] == key:
      arr.pop(i)
      break

def find_set_of_interecting_circles(centers, radius):
  intersections = set()
  for i in range(len(centers)):
    for j in range(len(centers)):
      if (j != i):
        # Tính khoảng cách giữa hai tâm đường tròn
        d = np.sqrt((centers[j][0] - centers[i][0])**2 + (centers[j][1] - centers[i][1])**2)

        # Tính khoảng cách từ tâm đường tròn thứ nhất đến điểm giao điểm
        a = (radius[i]**2 - radius[j]**2 + d**2) / (2 * d)
        # Tính chiều cao từ điểm giao điểm đến đường thẳng nối hai tâm
        h = np.sqrt(radius[i]**2 - a**2)

        # Tính tọa độ của hai điểm giao điểm
        x_intersect1 = centers[i][0] + a * (centers[j][1] - centers[i][0]) / d
        y_intersect1 = centers[i][1] + a * (centers[j][1] - centers[i][1]) / d
        x_intersect2 = x_intersect1 + h * (centers[j][1] - centers[i][1]) / d
        y_intersect2 = y_intersect1 - h * (centers[j][1] - centers[i][0]) / d
        intersections.add((x_intersect1, y_intersect1))
        intersections.add((x_intersect2, y_intersect2))


  set_points_in_circles = []
  for point in intersections:
    point_in_circles = []
    for i in range(len(centers)):
      if np.sqrt((point[0] - centers[i][0])**2 + (point[1] - centers[i][1])**2) <= radius[i] + 0.01:
        point_in_circles.append(i)
    set_points_in_circles.append(point_in_circles)

  set_points_in_circles = sorted(set_points_in_circles, key=lambda x: len(x), reverse=True)

  return set_points_in_circles

def remove_arr_of_set(set):
  different = False
  new_arr = []
  new_arr.append(set[0])
  for i in range(1, len(set)):
    different = False
    for arr in new_arr:
      if set[i] == arr:
        different = True
    if not different:
      new_arr.append(set[i])
  return new_arr

def circle_intersection(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    # Tạo đối tượng đường tròn từ tâm và bán kính
    circle1 = Point(x1, y1).buffer(r1)
    circle2 = Point(x2, y2).buffer(r2)

    # Tìm vùng giao nhau của hai đường tròn
    intersection = circle1.intersection(circle2)

    return intersection

def find_intersecting_circles(centers, arr_radii):
  count_circle = []
  count_circle_intersection = []
  for i in range(len(centers)-1):
      count_circle = []
      for j in range(i+1, len(centers)):
          intersection = circle_intersection((centers[i][0], centers[i][1], arr_radii[i]),
                                            (centers[j][0], centers[j][1], arr_radii[j]))
          if intersection.area > 0:
              count_circle.append(j)
      count_circle.append(i)
      count_circle_intersection.append(count_circle)
  return count_circle_intersection

def find_intersecting_circles2(centers, radii):
  count_circle = []
  count_circle_intersection = []
  for i in range(len(centers)-1):
      count_circle = []
      for j in range(i+1, len(centers)):
          intersection = circle_intersection((centers[i][0], centers[i][1], radii),
                                            (centers[j][0], centers[j][1], radii))
          if intersection.area > 0:
              count_circle.append(j)
      count_circle.append(i)
      count_circle_intersection.append(count_circle)
  return count_circle_intersection

def remove_common_elements2(arr, nodes):
    for i in range(len(arr)-2):
      arr = sorted(arr, key=lambda x: len(x), reverse=True)
      first_subarray = arr[i]
      for num in first_subarray:
          for subarray in arr[(i+1):]:  # Duyệt qua các mảng còn lại
              if num in subarray:
                  subarray.remove(num)  # Xoá phần tử khỏi mảng
      arr = sorted(arr, key=lambda x: len(x), reverse=True)
    arr = [subarray for subarray in arr if subarray]

    find_and_add_alone_circle(arr, nodes)
    return arr

def find_nearest_point(d, points):
  nearest_points = []
  for point in points:
    nearest_point = None
    min_distance = float('inf')
    for other_point in points:
      if point == other_point:
        continue
      distance = sqrt(((point[0] - other_point[0]) ** 2) + ((point[1] - other_point[1]) ** 2))
      if distance < min_distance and distance <= d:
        min_distance = distance
        nearest_point = other_point
    nearest_points.append(nearest_point)
  return nearest_points

# hàm tìm và thêm các hình tròn k giao vs bất kỳ hình tròn nào khác
def find_and_add_alone_circle(arr_nodes, nodes):
  merge_arr_node = []
  for arr_node in arr_nodes:
    for node in arr_node:
      merge_arr_node.append(node)

  arr_node_alone = []
  for i in range(len(nodes)):
    arr_node_alone = []
    if i not in merge_arr_node:
      arr_node_alone.append(i)
    if (len(arr_node_alone) == 1):
      arr_nodes.append(arr_node_alone)
