from pyrep.robots.arms.arm import Arm

class Gen3(Arm):

    def __init__(self, count: int = 0):
        super().__init__(count, 'Gen3', num_joints=7)
