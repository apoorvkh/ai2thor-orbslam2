import os
from os.path import join
from ai2thor.controller import Controller
import cv2
import subprocess

class ORBSLAM2Controller(Controller):
    def __init__(self, orbslam2_dir, frame_dir, fps, *args, **kwargs):
        self.orbslam2_dir = orbslam2_dir
        self.frame_dir = frame_dir
        self.frame_counter = 0
        self.timestamp_incr = 1 / fps
        self.write_file = open(join(self.frame_dir, 'rgb.txt'), 'w')
        print('#\n#\n#', file=self.write_file)
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        event = super().step(*args, **kwargs)
        out_file = '%06d.jpg' % self.frame_counter
        timestamp = self.frame_counter * self.timestamp_incr
        cv2.imwrite(join(self.frame_dir, out_file), event.cv2image())
        print('%.6f %s' % (timestamp, out_file), file=self.write_file)
        self.frame_counter += 1
        return event

    def stop(self, *args, **kwargs):
        super().stop(*args, **kwargs)
        self.write_file.close()
        exec_file = join(self.orbslam2_dir, 'Examples', 'Monocular', 'mono_tum')
        vocab_file = join(self.orbslam2_dir, 'Vocabulary', 'ORBvoc.txt')
        camera_file = join(self.orbslam2_dir, 'Examples', 'Monocular', 'TUM1.yaml')
        os.system('%s %s %s %s' % (exec_file, vocab_file, camera_file, self.frame_dir))
