"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint

class PILCOCartPoleEnv(gym.Env):
    """
    Description:
        Adaptated from gym's cart pole environment be the same as the one
        presented in PILCO.

    Source:
        This environment corresponds to the cart pole problem in
        Marc Deisenroth's thesis "Efficient Reinforcement Learning using
        Gaussian Processes."

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -Inf            Inf
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Continuous value where negative values are forces to the left.

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.82
        self.masscart = 0.5
        self.masspole = 0.5
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.25 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.friction = 0.1
        self.tau = 0.025  # seconds between state updates

        # Angle at which to fail the episode
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        max_force = np.asarray([10.0])

        self.action_space = spaces.Box(-max_force, max_force, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # Calculate the rates of change.
        pml = self.polemass_length * 2
        pl = self.length * 2
        def get_rate(curr_state, t):
            x, x_vel, theta, theta_vel = curr_state
            sintheta = np.sin(theta)
            costheta = np.cos(theta)
            x_acc = 2 * pml * theta_vel ** 2 * sintheta
            x_acc += 3 * self.masspole * self.gravity * sintheta * costheta
            x_acc += 4 * (action - 4 * self.friction * x_vel)
            x_acc /= 4 * self.total_mass - 3 * self.masspole * costheta ** 2
            theta_acc = -3 * pml * theta_vel ** 2 * sintheta * costheta
            theta_acc -= 6 * self.total_mass * self.gravity * sintheta
            theta_acc -= 6 * (action - self.friction * x_vel) * costheta
            theta_acc /= 4 * pl * self.total_mass - 3 * pml * costheta ** 2
            return np.asarray([x_vel, x_acc, theta_vel, theta_acc])
        # Integrate rates and find new state.
        init_state = self.state
        self.state = odeint(get_rate, init_state, [0, self.tau])[-1, :]
        # Calculate the reward.
        x = self.state[0]
        done =  x < -self.x_threshold \
                or x > self.x_threshold
        done = bool(done)

        theta = self.state[2]
        pend_tip = np.asarray([pl * np.sin(theta), -1 * pl * np.cos(theta)])
        pend_tip_diff = np.linalg.norm(np.asarray([0, pl]) - pend_tip)
        reward = -1 * (1 - np.exp(-0.5 * pend_tip_diff ** 2))

        if done:
            if self.steps_beyond_done is None:
                # Pole just fell!
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
            reward = -1 * (1 - np.exp(-0.5 * pl ** 2))

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.zeros(4)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2] + np.pi)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
