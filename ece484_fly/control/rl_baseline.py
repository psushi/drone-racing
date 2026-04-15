

class RLBaseLineController(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)

    def compute_control(self, obs, info=None) -> NDArray[np.floating]:

