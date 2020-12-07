import random
import math

import dijkstra
import ai2thor
from ai2thor_docker.x_server import startx
from ai2thor_orbslam2 import ORBSLAM2Controller


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

def straight_line(controller, distance, step_size):
    for _ in range(int(distance // step_size)):
        event = controller.step(action='MoveAhead', moveMagnitude=step_size)
    rem_dist = distance % step_size
    if rem_dist > 0:
        event = controller.step(action='MoveAhead', moveMagnitude=rem_dist)
    return event

def rotate(controller, degrees, theta):
    rotate_action = 'Rotate' + ('Right' if degrees >= 0 else 'Left')
    for _ in range(int(degrees // theta)):
        event = controller.step(action=rotate_action, degrees=theta)
    rem_degrees = degrees % theta
    if rem_degrees > 0:
        event = controller.step(action=rotate_action, degrees=rem_degrees)
    return event

def follow_path(controller, src_pos, src_rot, path):
    cur_pos, cur_rot = src_pos, src_rot
    rotateStepDegrees = controller.initialization_parameters['rotateStepDegrees']
    gridSize = controller.initialization_parameters['gridSize']
    for p in path:
        d_rot = cur_rot - get_rot(cur_pos, p)
        d_pos = dist(cur_pos, p)
        if d_rot > 0:
            event = rotate(controller, d_rot, rotateStepDegrees)
            cur_rot = event.metadata['agent']['rotation']['y']
        if d_pos > 0:
            event = straight_line(controller, d_pos, gridSize)
            cur_pos = event.metadata['agent']['position']
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

    follow_path(controller, src, src_rot, path)

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

    follow_path(controller, src, src_rot, path)


if __name__ == '__main__':
    startx()

    gridSize = 0.05
    rotateStepDegrees = 2.0
    controller = ORBSLAM2Controller('/app/ORB_SLAM2/Vocabulary/ORBvoc.bin', scene='FloorPlan28', server_class=ai2thor.fifo_server.FifoServer, gridSize=gridSize, snapToGrid=False, renderDepthImage=True)

    # square trajectory
    for _ in range(4):
        straight_line(controller, 0.5, gridSize)
        rotate(controller, 90.0, rotateStepDegrees)

    controller.stop()
    print('FPS: %.2f' % controller.fps())
    # controller.vis_trajectory('trajectory.png')
