from toy_envs.cts_cart_pole import CtsCartPoleEnv
from toy_envs.pilco_cart_pole import PILCOCartPoleEnv

all_envs = {'CtsCartPoleEnv': CtsCartPoleEnv,
            'PILCOCartPoleEnv': PILCOCartPoleEnv}

__all__ = ['CtsCartPoleEnv', 'PILCOCartPoleEnv']
