from .base import DataProvider

class MassSpecGymProvider(DataProvider):
    def __init__(self):
        super(MassSpecGymProvider, self).__init__("MassSpecGym")
