import random
import math
import torch

import dijkstra
from ai2thor_docker.x_server import startx
from ai2thor_gradslam import GradslamController


def dist(src, dst):
    del_x, del_z = (dst['x'] - src['x'], dst['z'] - src['z'])
    return math.sqrt(math.pow(del_x, 2) + math.pow(del_z, 2))

def get_rot(src, dst):
    del_x, del_z = (dst['x'] - src['x'], dst['z'] - src['z'])
    if del_x == 0:
        theta = (3/2 if del_z <= 0 else 1/2) * math.pi
    elif del_z == 0:
        theta = math.pi if del_x <= 0 else 0
    else:
        theta = math.atan(del_z / del_x) + (math.pi if del_x < 0 else 0)
    return math.degrees(theta)

def goto(src, src_rot, dst):
    rot_to_dst = src_rot - get_rot(src, dst)
    rotate_action = 'Rotate' + ('Right' if rot_to_dst >= 0 else 'Left')
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

def random_walk(controller, limit=None):
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

    if limit is not None:
        path = path[:limit]

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

    follow_path(src, src_rot, path[: int(0.2 * len(path))])


if __name__ == '__main__':
    startx()

    device = torch.device('cuda')
    gridSize = 0.05
    rotateStepDegrees = 4.0
    controller = GradslamController(device=device, store_pointclouds=True,
                                    scene='FloorPlan28', gridSize=gridSize, snapToGrid=False, renderDepthImage=True)

    random_walk(controller, limit=2)

    controller.stop()
    print('FPS: %.2f' % controller.fps())

    controller.vis_gt_trajectory('gt.png')
    controller.vis_slam_trajectory('slam.png')
    controller.vis_pointcloud('pc.html')
    controller.vis_all_pointclouds('pcs.html')
