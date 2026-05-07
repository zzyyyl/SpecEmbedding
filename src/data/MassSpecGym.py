from .utils import DataProvider

class MassSpecGymProvider(DataProvider):
    def __init__(self):
        super(MassSpecGymProvider, self).__init__("MassSpecGym")

    def load_candidates(self, type='mass'):
        '''
        type: 'mass', 'formula' (bonus)
        '''
        return super(MassSpecGymProvider, self).load_candidates(type=type)
