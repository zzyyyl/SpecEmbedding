from .base import DataProvider
from SpecEmbedding.config import config

class MassBankProvider(DataProvider):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = config.data.data_path
        super(MassBankProvider, self).__init__("MassBank", data_dir)
