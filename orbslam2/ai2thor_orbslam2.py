import os
import tempfile
import yaml
import math
from time import time
from ai2thor.controller import Controller
import orbslam2
import matplotlib.pyplot as plt

## redo in numpy
# def init_pose(event):
#     pose = torch.eye(4)
#     pose[:3, 3] = torch.tensor([
#         event.metadata['agent']['position']['x'],
#         event.metadata['agent']['position']['y'],
#         event.metadata['agent']['position']['z'],
#     ])
#     angles = torch.tensor([
#         event.metadata['agent']['rotation']['z'],
#         event.metadata['agent']['rotation']['y'],
#         event.metadata['agent']['rotation']['x']
#     ])
#     pose[:3, :3] = euler_angles_to_matrix(torch.deg2rad(angles), 'ZYX')
#     return pose

def init_settings(event):
    w, h = event.metadata['screenWidth'], event.metadata['screenHeight']
    fov = event.metadata['fov']
    f_x = w / (2 * math.tan(math.radians(fov / 2)))
    f_y = h / (2 * math.tan(math.radians(fov / 2)))
    c_x, c_y = w / 2, h / 2
    return {
        'Camera.fx': f_x, 'Camera.fy': f_y, 'Camera.cx': c_x, 'Camera.cy': c_y,
        'Camera.k1': 0.0, 'Camera.k2': 0.0, 'Camera.p1': 0.0, 'Camera.p2': 0.0,
        'Camera.width': w, 'Camera.height': h, 'Camera.fps': 30.0, 'Camera.bf': 40.0, 'Camera.RGB': 1,
        'ThDepth': 200.0, 'DepthMapFactor': 1.0,
        'ORBextractor.nFeatures': 1000, 'ORBextractor.scaleFactor': 1.2, 'ORBextractor.nLevels': 8,
        'ORBextractor.iniThFAST': 20, 'ORBextractor.minThFAST': 1,
        'Viewer.KeyFrameSize': 0.05, 'Viewer.KeyFrameLineWidth': 1, 'Viewer.GraphLineWidth': 0.9,
        'Viewer.PointSize':2, 'Viewer.CameraSize': 0.08, 'Viewer.CameraLineWidth': 3,
        'Viewer.ViewpointX': 0, 'Viewer.ViewpointY': -0.7, 'Viewer.ViewpointZ': -1.8, 'Viewer.ViewpointF': 500
    }

def init_slam(vocab_file, event):
    _, settings_file = tempfile.mkstemp()
    with open(settings_file, 'w') as file:
        file.write('%YAML:1.0')
        file.write(yaml.dump(init_settings(event)))
    slam_system = orbslam2.SLAM(vocab_file, settings_file, 'rgbd')
    os.remove(settings_file)
    return slam_system

class ORBSLAM2Controller(Controller):
    def __init__(self, vocab_file, **kwargs):
        self.controller_init = False
        super().__init__(**kwargs)
        self.controller_init = True

        self.vocab_file = vocab_file
        self.slam_system = None
        self.gt_trajectory = {'x' : [], 'z' : []}

        self.frame_counter = 0
        self.time_elapsed = 0.0

    def step(self, **kwargs):
        start = time()
        event = super().step(**kwargs)

        if self.controller_init:
            self.gt_trajectory['x'].append(event.metadata['agent']['position']['x'])
            self.gt_trajectory['z'].append(event.metadata['agent']['position']['z'])

            if self.slam_system is None:
                self.slam_system = init_slam(self.vocab_file, event)

            pose = self.slam_system.track(event.frame, event.depth_frame)

            self.frame_counter += 1
            self.time_elapsed += time() - start

        return event

    def stop(self, **kwargs):
        super().stop(**kwargs)
        self.slam_system.shutdown()

    def vis_trajectory(self, file):
        # plt.scatter(self.gt_trajectory['x'], self.gt_trajectory['z'])

        _, keyframe_file = tempfile.mkstemp()
        self.slam_system.save_keyframe_trajectory(keyframe_file)
        with open(keyframe_file, 'r') as rf:
            keyframes = rf.readlines()
            keyframes = list(zip(*[kf.split()[1:4] for kf in keyframes]))
            kf_x = list(map(float, keyframes[0]))
            kf_z = list(map(float, keyframes[2]))
        os.remove(keyframe_file)

        plt.scatter(kf_x, kf_z)

        plt.savefig(file)

    def fps(self):
        return self.frame_counter / self.time_elapsed
