from .base import DataProvider
from SpecEmbedding.config import config

class MassSpecGymProvider(DataProvider):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = config.data.data_path
        super(MassSpecGymProvider, self).__init__("MassSpecGym", data_dir)
