from SpecEmbedding.config import config

from .base import DataProvider


class MoNAProvider(DataProvider):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = config.data.data_path
        super(MoNAProvider, self).__init__("MoNA", data_dir)
