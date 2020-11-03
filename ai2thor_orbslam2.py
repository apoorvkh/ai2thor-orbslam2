import os
from os.path import join
from ai2thor.controller import Controller
import cv2
import subprocess
import orbslam2
from time import time

class ORBSLAM2Controller(Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slam_system = orbslam2.SLAM(b'/app/ORB_SLAM2/Vocabulary/ORBvoc.txt', b'/app/ORB_SLAM2/Examples/Monocular/TUM3.yaml')
        self.start_time = None
        self.frame_counter = 0

    def step(self, *args, **kwargs):
        event = super().step(*args, **kwargs)
        try:
            self.slam_system.track_monocular(event.frame, 0)
            if self.start_time is None:
                self.start_time = time()
            self.frame_counter += 1
        except AttributeError:
            pass
        return event

    def stop(self, *args, **kwargs):
        super().stop(*args, **kwargs)
        self.slam_system.shutdown()
        self.fps = self.frame_counter / (time() - self.start_time)
