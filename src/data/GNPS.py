from SpecEmbedding.config import config

from .base import DataProvider


class GNPSProvider(DataProvider):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = config.data.data_path
        super(GNPSProvider, self).__init__("GNPS", data_dir)
