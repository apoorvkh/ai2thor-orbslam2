import math
from time import time
from ai2thor.controller import Controller
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
import numpy as np
import torch
import vis
import matplotlib.pyplot as plt


def get_intrinsics(event):
    w, h = event.metadata['screenWidth'], event.metadata['screenHeight']
    focal_div = torch.tan(torch.deg2rad(torch.tensor(event.metadata['fov'] / 2)))
    intrinsics = torch.eye(4)
    intrinsics[0, 0] = w / focal_div  # f_x
    intrinsics[1, 1] = h / focal_div  # f_y
    intrinsics[0, 2] = w / 2  # c_x
    intrinsics[1, 2] = h / 2  # c_y
    return intrinsics.view(1, 1, 4, 4)

def init_pose():
    return torch.eye(4).view(1, 1, 4, 4)

def to_rgbd(frame, depth_frame, intrinsics, poses, device=None):
    frame = torch.tensor(frame.copy(), dtype=torch.float32)[None, None, ...]
    depth_frame = torch.tensor(depth_frame, dtype=torch.float32)[None, None, ..., None]
    return RGBDImages(frame, depth_frame, intrinsics, poses, device=device)


class GradslamController(Controller):

    def __init__(self, num_iters=20, device=None, store_pointclouds=False, max_pc_points=50000, **kwargs):
        self.controller_init = False
        super().__init__(**kwargs)
        self.controller_init = True

        self.device = device
        self.store_pointclouds = store_pointclouds
        self.max_pc_points = max_pc_points

        self.slam = PointFusion(numiters=num_iters, device=device)
        self.pointcloud = Pointclouds(device=device)

        self.gt_trajectory = {'x' : [], 'z' : []}
        self.intrinsics = None
        self.live_frame, self.prev_frame = None, None
        self.pointclouds = []
        self.poses = []

        self.frame_counter = 0
        self.time_elapsed = 0.0

    def step(self, **kwargs):
        start = time()
        event = super().step(**kwargs)

        if self.controller_init:
            self.gt_trajectory['x'].append(event.metadata['agent']['position']['x'])
            self.gt_trajectory['z'].append(event.metadata['agent']['position']['z'])

            if self.intrinsics is None:
                self.intrinsics = get_intrinsics(event)
            pose = init_pose() if self.prev_frame is None else None
            self.live_frame = to_rgbd(event.frame, event.depth_frame, self.intrinsics, pose, device=self.device)

            self.pointcloud, self.live_frame.poses = self.slam.step(self.pointcloud, self.live_frame, self.prev_frame)

            if self.store_pointclouds is not None:
                self.pointclouds.append(self.pointcloud[0].cpu().plotly(0, as_figure=False, max_num_points=self.max_pc_points))
            self.poses.append(self.live_frame.poses[0, 0].cpu())
            self.prev_frame, self.live_frame = self.live_frame, None

            self.frame_counter += 1
            self.time_elapsed += time() - start

        return event

    def stop(self, **kwargs):
        super().stop(**kwargs)

    def fps(self):
        return self.frame_counter / self.time_elapsed

    def vis_gt_trajectory(self, file):
        plt.scatter(self.gt_trajectory['x'], self.gt_trajectory['z'])
        plt.savefig(file)

    def vis_slam_trajectory(self, file):
        poses = torch.stack(self.poses)
        plt.scatter(poses[:, 0, 3], poses[:, 2, 3])  # x, z
        plt.savefig(file)

    def vis_pointcloud(self, file):
        with open(file, 'w') as out:
            poses = torch.stack(self.poses).numpy()
            intrinsics = self.intrinsics.view(4, 4).numpy()
            trajectory_frustum = vis.plotly_poses(poses, intrinsics)[-1]
            fig = self.pointcloud.plotly(0, max_num_points=self.max_pc_points)
            fig.add_traces(trajectory_frustum)
            out.write(fig.to_html())

    def vis_all_pointclouds(self, file):
        with open(file, 'w') as out:
            poses = torch.stack(self.poses).numpy()
            intrinsics = self.intrinsics.view(4, 4).numpy()
            fig = vis.pointcloud_updates(self.pointclouds, poses, intrinsics)
            out.write(fig.to_html())
