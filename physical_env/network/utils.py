import numpy as np
from scipy.spatial.distance import euclidean

def find_receiver(node):
    if not (node.status == 1):
        return -1
    candidates = [other_node for other_node in node.neighbors
                  if other_node.level < node.level and other_node.status == 1]

    if len(candidates) > 0:
        distances = [euclidean(candidate.location, node.location) for candidate in candidates]
        return candidates[np.argmin(distances)].id
    else:
        return -1

def request_function(node, optimizer, t):
    """
    add a message to request list of mc.
    :param node: the node request
    :param mc: mobile charger
    :param t: time get request
    :return: None
    """
    optimizer.list_request.append(
        {"id": node.id, "energy": node.energy, "energyCS": node.energyCS, "energy_estimate": node.energy,
         "time": t})
