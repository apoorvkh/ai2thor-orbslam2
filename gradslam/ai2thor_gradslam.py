import math
from time import time
import functools
from ai2thor.controller import Controller
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
import numpy as np
import torch
import vis
import matplotlib.pyplot as plt


def get_intrinsics(event):
    w, h = event.metadata['screenWidth'], event.metadata['screenHeight']
    fov = event.metadata['fov']
    intrinsics = torch.eye(4)
    intrinsics[0, 0] = w / math.tan(math.radians(fov / 2))  # f_x
    intrinsics[1, 1] = h / math.tan(math.radians(fov / 2))  # f_y
    intrinsics[0, 2] = w / 2  # c_x
    intrinsics[1, 2] = h / 2  # c_y
    return intrinsics

# From pytorch3d.transforms (see related documentation)
def euler_angles_to_matrix(euler_angles, convention):

    def _axis_angle_rotation(axis, angle):
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        if axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        if axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)

def init_pose(event):
    pose = torch.eye(4)
    pose[0, 3] = event.metadata['agent']['position']['x']
    pose[1, 3] = event.metadata['agent']['position']['y']
    pose[2, 3] = event.metadata['agent']['position']['z']
    angles = [event.metadata['agent']['rotation']['z'],
              event.metadata['agent']['rotation']['y'],
              event.metadata['agent']['rotation']['x']]
    angles = torch.deg2rad(torch.tensor(angles))
    pose[:3, :3] = euler_angles_to_matrix(angles, 'ZYX')
    return pose

def to_rgbd(frame, depth_frame, intrinsics, poses, device=None):
    frame = torch.tensor([[frame]], dtype=torch.float32)
    depth_frame = torch.tensor([[depth_frame]], dtype=torch.float32)[..., None]
    return RGBDImages(frame, depth_frame, intrinsics, poses, device=device)


class GradslamController(Controller):

    def __init__(self, numiters=20, device=None, store_pointclouds=False, max_pc_points=50000, **kwargs):
        self.controller_init = False
        super().__init__(**kwargs)
        self.controller_init = True

        self.device = device
        self.store_pointclouds = store_pointclouds
        self.max_pc_points = max_pc_points

        self.slam = PointFusion(numiters=numiters, device=device)
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
            intrinsics = self.intrinsics.view(1, 1, 4, 4)
            pose = init_pose(event).view(1, 1, 4, 4) if self.prev_frame is None else None
            self.live_frame = to_rgbd(event.frame, event.depth_frame, intrinsics, pose, device=self.device)

            self.pointcloud, self.live_frame.poses = self.slam.step(self.pointcloud, self.live_frame, self.prev_frame)

            if self.store_pointclouds is not None:
                self.pointclouds.append(self.pointcloud[0].cpu().plotly(0, as_figure=False, max_num_points=self.max_pc_points))
            self.poses.append(self.live_frame.poses[0, 0].cpu())
            self.prev_frame, self.live_frame = self.live_frame, None

            self.frame_counter += 1
            self.time_elapsed += time() - start

        return event

    def fps(self):
        return self.frame_counter / self.time_elapsed

    def vis_trajectory(self, file):
        # ground truth trajectory
        plt.scatter(self.gt_trajectory['x'], self.gt_trajectory['z'])
        # slam trajectory
        poses = torch.stack(self.poses)
        plt.scatter(poses[:, 0, 3], poses[:, 2, 3])  # x, z

        plt.savefig(file)

    def vis_pointcloud(self, file):
        with open(file, 'w') as out:
            # poses = torch.stack(self.poses).numpy()
            # intrinsics = self.intrinsics.numpy()
            # trajectory_frustum = vis.plotly_poses(poses, intrinsics)[-1]
            fig = self.pointcloud.plotly(0, max_num_points=self.max_pc_points)
            # fig.add_traces(trajectory_frustum)
            out.write(fig.to_html())

    def vis_all_pointclouds(self, file):
        with open(file, 'w') as out:
            poses = torch.stack(self.poses).numpy()
            intrinsics = self.intrinsics.numpy()
            fig = vis.pointcloud_updates(self.pointclouds, poses, intrinsics)
            out.write(fig.to_html())
