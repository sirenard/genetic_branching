import ecole
import my_module
from objproxies import LazyProxy


class ObservationWrapper:
    def __init__(self, constructor, *args, lazy_access=True):
        self.observation = constructor(*args)
        self.lazy_access = lazy_access

    def __getitem__(self, i):
        def f():
            return self.observation[i]

        if self.lazy_access:
            return LazyProxy(f)
        else:
            return f()

    def reset(self):
        self.observation.reset()

    def __len__(self):
        return self.observation.size()

class Observation:
    def __init__(self, *observations):
        self.observations = observations

    def __getitem__(self, i):
        for observation in self.observations:
            if i < len(observation):
                return observation[i]

            i -= len(observation)

        raise IndexError

    def __len__(self):
        return sum([len(observation) for observation in self.observations])
