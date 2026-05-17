from .base import DataProvider
from SpecEmbedding.config import config

class NPLIB1Provider(DataProvider):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = config.data.data_path
        super(NPLIB1Provider, self).__init__("NPLIB1", data_dir)
