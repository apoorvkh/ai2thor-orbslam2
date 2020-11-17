import os
from time import time
from ai2thor.controller import Controller
import orbslam2
import matplotlib.pyplot as plt


class ORBSLAM2Controller(Controller):
    def __init__(self, vocab_file, settings_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slam_system = orbslam2.SLAM(vocab_file, settings_file, 'monocular')
        self.frame_counter = 0
        self.time_elapsed = 0.0
        self.gt_trajectory = []

    def step(self, *args, **kwargs):
        start = time()
        event = super().step(*args, **kwargs)
        if hasattr(self, 'slam_system'):
            pose = self.slam_system.track(event.frame)
            self.frame_counter += 1
            self.time_elapsed += time() - start
        if 'action' in kwargs and kwargs['action'].startswith('Move'):
            self.gt_trajectory.append((event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z']))
        return event

    def stop(self, keyframe_traj, traj_plot, *args, **kwargs):
        super().stop(*args, **kwargs)
        self.slam_system.shutdown()
        self.slam_system.save_keyframe_trajectory(keyframe_traj)

        with open(keyframe_traj, 'r') as rf:
            keyframes = rf.readlines()
            keyframes = list(zip(*[kf.split()[1:4] for kf in keyframes]))
            kf_x = list(map(float, keyframes[0]))
            kf_z = list(map(float, keyframes[2]))

        plt.scatter(kf_x, kf_z)
        plt.savefig(traj_plot)
        plt.clf()

        gt_camera_x, gt_camera_z = list(zip(*self.gt_trajectory))
        plt.scatter(gt_camera_x, gt_camera_z)
        plt.savefig('gt_traj.png')

    def fps(self):
        return self.frame_counter / self.time_elapsed
