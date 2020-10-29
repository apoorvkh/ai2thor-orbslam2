from ai2thor.controller import Controller
import orbslam2

class ORBSLAM2Controller(Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slam_system = orbslam2.SLAM(b'/app/ORB_SLAM2/Vocabulary/ORBvoc.txt', b'/app/ORB_SLAM2/Examples/Monocular/TUM3.yaml')

    def step(self, *args, **kwargs):
        event = super().step(*args, **kwargs)
        if hasattr(self, 'slam_system'): # since step is called by ai2thor before __init__
            self.slam_system.track_monocular(event.frame, 0)
        return event

    def stop(self, *args, **kwargs):
        super().stop(*args, **kwargs)
        self.slam_system.shutdown()
