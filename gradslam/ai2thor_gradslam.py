import os
import math
from time import time
from ai2thor.controller import Controller
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
import numpy as np
import torch


def get_intrinsics(event):
    fov = event.metadata['fov']
    w, h = event.metadata['screenWidth'], event.metadata['screenHeight']
    intrinsics = np.eye(4)
    intrinsics[0, 0] = w / (2 * math.tan(math.radians(fov / 2)))  # f_x
    intrinsics[1, 1] = h / (2 * math.tan(math.radians(fov / 2)))  # f_y
    intrinsics[0, 2] = w / 2  # c_x
    intrinsics[1, 2] = h / 2  # c_y
    return intrinsics

def to_rgbd(frame, depth_frame, intrinsics, poses, device=None):
    frame = torch.tensor([[frame]], dtype=torch.float32, device=device)
    depth_frame = torch.tensor([[depth_frame]], dtype=torch.float32, device=device).unsqueeze(-1)
    intrinsics = torch.tensor([[intrinsics]], dtype=torch.float32, device=device)
    rgbd_images = RGBDImages(frame, depth_frame, intrinsics, poses, device=device)
    return rgbd_images

class GradslamController(Controller):
    def __init__(self, device, *args, **kwargs):
        self.super_init = False
        super().__init__(*args, **kwargs)
        self.super_init = True
        #
        self.device = device
        self.slam = PointFusion(device=device)
        self.pointclouds = Pointclouds(device=device)
        self.live_frame, self.prev_frame = None, None
        #
        self.frame_counter = 0
        self.time_elapsed = 0.0

    def step(self, *args, **kwargs):
        if self.super_init:
            start = time()

        event = super().step(*args, **kwargs)

        if self.super_init:
            if not hasattr(self, 'intrinsics'):
                self.intrinsics = get_intrinsics(event)
            if self.prev_frame is None:
                poses = torch.eye(4, device=self.device).view(1, 1, 4, 4)
            else:
                poses = self.prev_frame.poses
            self.live_frame = to_rgbd(event.frame, event.depth_frame, self.intrinsics, poses, device=self.device)
            self.pointclouds, self.live_frame.poses = self.slam.step(self.pointclouds, self.live_frame, self.prev_frame)
            self.prev_frame = self.live_frame
            #
            self.frame_counter += 1
            self.time_elapsed += time() - start

        return event

    def stop(self, *args, **kwargs):
        super().stop(*args, **kwargs)
        with open('pointclouds.html', 'w') as out:
            out.write(self.pointclouds.plotly(0).to_html())

    def fps(self):
        return self.frame_counter / self.time_elapsed
