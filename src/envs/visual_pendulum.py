__credits__ = ["Carlos Luis, Pavel Osinenko"]

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from os import path
from typing import Optional
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium import spaces  # Import spaces to define the observation space
from typing import Optional


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class VisualPendulum(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        g: float = 10.0,
        costs_coefs: tuple[float, float, float] = (1.0, 0.1, 0.001),
    ):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.render_mode = render_mode
        self.costs_coefs = costs_coefs

        self.screen_dim = 512
        self.screen = None
        self.clock = None
        self.isopen = True

        # Define action space
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        # Define observation space as image
        image_shape = (self.screen_dim, self.screen_dim, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=image_shape, dtype=np.uint8
        )

        self.state = None
        self.last_u = None

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # Ensure the action is a scalar or a 1D array
        u = np.clip(u, -self.max_torque, self.max_torque)

        # If the action is a scalar, do not index it
        if np.isscalar(u):
            torque = u
        else:
            torque = u[0]  # Index only if it's a vector

        # Update state
        costs = (
            self.costs_coefs[0] * angle_normalize(th) ** 2
            + self.costs_coefs[1] * thdot**2
            + self.costs_coefs[2] * (u**2)
        )
        newthdot = (
            thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * torque) * dt
        )
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        self.last_u = u

        # Get observation (image)
        obs = self._render_without_arrow()

        # Ensure costs is a scalar
        if isinstance(costs, np.ndarray):
            costs = float(costs)

        return obs, -costs, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])

        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        # Get observation (image)
        obs = self._render_without_arrow()

        # Ensure observation is properly shaped
        if obs is None:
            raise ValueError("Observation from _render_without_arrow is None")

        # Check if observation has the correct shape
        expected_shape = (self.screen_dim, self.screen_dim, 3)
        if obs.shape != expected_shape:
            raise ValueError(
                f"Observation has shape {obs.shape}, expected {expected_shape}"
            )

        return obs, {}

    def _get_raw_obs(self):
        """Get the raw state observation (not used for agent, only for debugging)"""
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def _render_without_arrow(self):
        """Render the environment without the torque arrow"""
        if self.render_mode is None:
            return None

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array" or "rgb_array_with_arrow"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None
        else:  # mode == "rgb_array" or "rgb_array_with_arrow"
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _render_with_arrow(self):
        """Render the environment with the torque arrow"""
        if self.render_mode is None:
            return None

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array" or "rgb_array_with_arrow"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        # Add the torque arrow
        if self.last_u is not None:
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            img = pygame.image.load(fname)

            # Fix: use conversion to integer for mitigating pygame error
            scale_factor = max(
                0.2, min(1.0, np.abs(self.last_u) / self.max_torque)
            )  # Ensure scale is between 0.2 and 1.0
            arrow_size = (
                int(scale_factor * 100),
                int(scale_factor * 100),
            )  # Define max arrow size
            scale_img = pygame.transform.smoothscale(img, arrow_size)

            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)

            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None
        else:  # mode == "rgb_array" or "rgb_array_with_arrow"
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def render(self):
        return self._render_with_arrow()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.screen = None
            self.clock = None

    def get_rng_state(self):
        """Retrieve the RNG state."""
        if hasattr(self.np_random, "bit_generator"):  # For Generator class
            return self.np_random.bit_generator.state
        raise AttributeError("The current RNG does not support retrieving state.")
