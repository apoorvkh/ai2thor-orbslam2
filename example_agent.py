from ai2thor_docker.x_server import startx
from ai2thor_orbslam2 import ORBSLAM2Controller
import random


if __name__ == '__main__':
    startx()
    controller = ORBSLAM2Controller(scene='FloorPlan28', gridSize=0.05)
    for _ in range(1000):
        rval = random.randint(0, 5)
        if rval == 0:
            controller.step(action='MoveAhead')
        elif rval == 1:
            controller.step(action='MoveBack')
        elif rval == 2:
            controller.step(action='MoveRight')
        elif rval == 3:
            controller.step(action='MoveLeft')
        elif rval == 4:
            controller.step(action='RotateRight', degrees=2.0)
        elif rval == 5:
            controller.step(action='RotateLeft', degrees=2.0)
    controller.stop()
