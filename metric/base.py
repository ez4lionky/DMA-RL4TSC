class BaseMetric(object):
    def __init__(self, world):
        self.world = world
        self.name = "Base Metric"

    def update(self, done=False):
        raise NotImplementedError()
        return self.eval()

    def eval(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()