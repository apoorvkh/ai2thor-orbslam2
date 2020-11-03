import random
import math

import dijkstra
from ai2thor_docker.x_server import startx
from ai2thor_orbslam2 import ORBSLAM2Controller


startx()

gridSize = 0.05
rotateStepDegrees = 1.0
controller = ORBSLAM2Controller(scene='FloorPlan28', gridSize=gridSize, snapToGrid=False)

def dist(p1, p2):
    return ((p1['x'] - p2['x']) ** 2 + (p1['z'] - p2['z']) ** 2) ** 0.5

def get_rot(src, dst):
    del_x = dst['x'] - src['x']
    del_z = dst['z'] - src['z']
    if del_z > 0:
        if del_x > 0:
            return math.degrees(math.atan(del_z / del_x))
        elif del_x < 0:
            return math.degrees(math.pi - math.atan(del_z / del_x))
        return 90.0
    elif del_z < 0:
        if del_x < 0:
            return math.degrees(3/2 * math.pi - math.atan(del_x / del_z))
        elif del_x > 0:
            return math.degrees(2 * math.pi - math.atan(del_x / del_z))
        return 270.0
    return 180.0 if del_x < 0 else 0.0

def goto(src, src_rot, dst):
    rot_to_dst = src_rot - get_rot(src, dst)
    rotate_action = 'RotateRight' if rot_to_dst >= 0 else 'RotateLeft'
    rot_to_dst = abs(rot_to_dst)
    while rot_to_dst > 0:
        del_r = min(rot_to_dst, rotateStepDegrees)
        event = controller.step(action=rotate_action, degrees=del_r)
        rot_to_dst -= del_r
    dist_to_dst = dist(src, dst)
    while dist_to_dst > 0:
        d_m = min(dist_to_dst, gridSize)
        event = controller.step(action='MoveAhead', moveMagnitude=d_m)
        dist_to_dst -= d_m
    return event

def follow_path(src, src_rot, path):
    for p in path:
        event = goto(src, src_rot, p)
        src = event.metadata['agent']['position']
        src_rot = event.metadata['agent']['rotation']['y']
    return event

def random_walk(controller):
    event = controller.step(action='GetReachablePositions')
    reachable_positions = event.metadata['actionReturn']
    src = event.metadata['agent']['position']
    src_rot = event.metadata['agent']['rotation']['y']

    src_index = reachable_positions.index(src)
    src_dists = [dist(src, dst) for dst in reachable_positions]

    graph = {}
    for i, p1 in enumerate(reachable_positions):
        dists = {}
        for j, p2 in enumerate(reachable_positions):
            d = dist(p1, p2)
            if d <= gridSize and i != j:
                dists[j] = d
        graph[i] = dists

    path = None
    while path is None:
        dst = random.choice(reachable_positions)
        dst_index = reachable_positions.index(dst)
        if dist(src, dst) > 0.5 * max(src_dists):
            path = dijkstra.shortest_path(graph, src_index, dst_index)
    path = [reachable_positions[i] for i in path]

    follow_path(src, src_rot, path)

def random_shortest_path(controller):
    event = controller.step(action='GetReachablePositions')
    reachable_positions = event.metadata['actionReturn']
    src = event.metadata['agent']['position']
    src_rot = event.metadata['agent']['rotation']['y']

    src_index = reachable_positions.index(src)
    src_dists = [dist(src, dst) for dst in reachable_positions]

    dst_index = src_index
    while src_dists[dst_index] < 0.5 * max(src_dists):
        dst_index = random.choice(range(len(reachable_positions)))
    dst = reachable_positions[dst_index]

    path = controller.step(action="GetShortestPathToPoint", position=src, x=dst['x'], y=dst['y'], z=dst['z']).metadata['actionReturn']['corners']

    follow_path(src, src_rot, path)


random_walk(controller)

controller.stop()

print('FPS', controller.fps)
