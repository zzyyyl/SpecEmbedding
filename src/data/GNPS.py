from .base import DataProvider

class GNPSProvider(DataProvider):
    def __init__(self):
        super(GNPSProvider, self).__init__("GNPS")
