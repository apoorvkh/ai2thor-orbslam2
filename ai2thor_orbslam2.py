from ai2thor.controller import Controller
import cv2
import subprocess

class ORBSLAM2Controller(Controller):
    def __init__(self, orbslam2_dir, frame_dir, fps, *args, **kwargs):
        self.orbslam2_dir = orbslam2_dir
        self.frame_dir = frame_dir
        self.frame_counter = 0
        self.timestamp_incr = 1 / fps
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        event = super().step(*args, **kwargs)
        cv2.imwrite(join(self.frame_dir, '%.6f.jpg' % (self.frame_counter * self.timestamp_incr)), event.cv2image())
        self.frame_counter += 1
        return event

    def stop(self, *args, **kwargs):
        super().stop(*args, **kwargs)
        exec_file = join(self.orbslam2_dir, 'mono_ai2thor')
        vocab_file = join(self.orbslam2_dir, 'Vocabulary', 'ORBvoc.txt')
        rc = subprocess.call('%s %s %s' % (exec_file, vocab_file, self.frame_dir))
