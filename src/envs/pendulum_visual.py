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


class PendulumRenderFix(gym.Env):
    """
    ## Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)

    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |


    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ## Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
      The default value is g = 10.0 .

    ```python
    import gymnasium as gym
    gym.make('Pendulum-v1', g=9.81)
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.

    ## Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        g=10.0,
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

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the gymnasium api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # Ensure the action is a scalar or a 1D array
        u = np.clip(u, -self.max_torque, self.max_torque)
        # print(f"We are in the environment. u is {u}")

        # If the action is a scalar, do not index it
        if np.isscalar(u):
            torque = u
        else:
            torque = u[0]  # Index only if it's a vector

        # Debug the current state before applying updates
        # print(f"Before update: th={th}, thdot={thdot}, torque={torque}")

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

        # Debug the updated state
        # print(f"After update: th={newth}, thdot={newthdot}")

        # Render if needed
        if self.render_mode == "human":
            self.render()

        assert np.all(np.isfinite(self.state)), f"Invalid state: {self.state}"

        # Ensure that self.last_u is updated with the most recent action
        self.last_u = u

        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        # print(f"DEBUG: Base reset called with seed={seed}")
        super().reset(seed=seed)

        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        # obs = self._get_obs()
        # print(f"DEBUG: Base reset returned obs={obs}")

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def get_rng_state(self):
        """Retrieve the RNG state."""
        if hasattr(self.np_random, "bit_generator"):  # For Generator class
            return self.np_random.bit_generator.state
        raise AttributeError("The current RNG does not support retrieving state.")

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

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
            else:  # mode in "rgb_array"
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

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)

        if self.last_u is not None:

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

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumRenderFixNoArrow(PendulumRenderFix):
    """
    A variant of PendulumRenderFix that removes the torque arrows during rendering.
    """

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

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
            else:  # mode in "rgb_array"
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

        # Remove the section that draws the torque arrows

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


class PendulumRenderFixNoArrowParallelizable(PendulumRenderFixNoArrow):
    """
    A variant of PendulumRenderFixNoArrow that ensures compatibility with parallel environments.
    """

    def __init__(
        self,
        render_mode=None,
        costs_coefs: tuple[float, float, float] = (1.0, 0.1, 0.001),
        **kwargs,
    ):
        # Initialize parent class with stateless behavior
        super().__init__(render_mode=render_mode, costs_coefs=costs_coefs, **kwargs)

    def reset(self, *, seed: int = None, options: dict = None):
        # Reset the environment
        obs, info = super().reset(seed=seed, options=options)

        if self.render_mode == "rgb_array":
            # Render for image-based observations
            image = self.render()
            return image, info
        else:
            # Return the raw observation for non-image modes
            return obs, info

    def step(self, action):
        # Step in the environment
        obs, reward, done, truncated, info = super().step(action)

        if self.render_mode == "rgb_array":
            # Render for image-based observations
            image = self.render()
            return image, reward, done, truncated, info
        else:
            # Return the raw observation for non-image modes
            return obs, reward, done, truncated, info

    def render(self):
        # Avoid creating rendering resources during initialization
        if self.render_mode is None:
            return None
        return super().render()

    def close(self):
        # Ensure resources are properly cleaned up
        if hasattr(self, "screen") and self.screen is not None:
            self.screen = None
        super().close()


class PendulumVisualNoArrowParallelizable(PendulumRenderFixNoArrowParallelizable):
    """
    Gym's Pendulum environment modified to provide image-based observations instead of direct state measurements.
    Inherits from PendulumRenderFixNoArrowParallelizable to fix rendering issues, remove torque arrow and allowing usage with SubprocVecEnv
    """

    def __init__(
        self,
        render_mode="rgb_array",
        costs_coefs: tuple[float, float, float] = (1.0, 0.1, 0.001),
        render_during_training=False,
    ):
        super(PendulumVisualNoArrowParallelizable, self).__init__(
            render_mode=render_mode, costs_coefs=costs_coefs
        )  # Pass render_mode to the parent class

        self.render_mode = render_mode  # Set the render mode

        # Update the observation space to use image dimensions
        image_shape = (500, 500, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=image_shape, dtype=np.uint8
        )

    def reset(self, *, seed: int = None, options: dict = None):
        # Reset using the custom environment's method
        obs, info = super().reset(seed=seed, options=options)

        # print(f"Raw observation from PendulumVisual: Min={obs.min()}, Max={obs.max()}, Shape={obs.shape}")

        image = self.render()  # Get the image-based observation

        # Debug: Plot raw image
        # self._plot_image(image)

        # Render image for the agent if in "rgb_array" mode
        if self.render_mode == "rgb_array":
            image = self.render()
            # Check if the image is valid
            if image is None or image.size == 0:
                raise ValueError("Rendered image in reset() is empty or None.")
            return image, info
        else:
            # If in "human" mode, just return a placeholder observation
            return obs, info

    def step(self, action):
        # Step using the custom environment's method
        obs, reward, done, truncated, info = super().step(action)
        # print(f"Render mode is {self.render_mode}")

        # Ensure reward is a scalar for single environments or 1D for vectorized
        if isinstance(reward, np.ndarray):
            reward = np.squeeze(reward)  # Remove singleton dimensions

        # Render image for the agent if in "rgb_array" mode
        if self.render_mode == "rgb_array":
            image = self.render()

            # print(f"Raw observation from PendulumVisual: Min={image.min()}, Max={image.max()}, Shape={image.shape}")

            # print(f"Final preprocessed observation before CNN: Min={image.min()}, Max={image.max()}")

            # Debug: Plot raw image
            # self._plot_image(image)

            # Check if the image is valid
            if image is None or image.size == 0:
                raise ValueError("Rendered image in step() is empty or None.")
            return image, float(reward), done, truncated, info
        else:
            # If in "human" mode, just return a placeholder observation
            return obs, float(reward), done, truncated, info

    def render(self):
        return super().render()  # Call the render method from PendulumRenderFix

    def close(self):
        super().close()  # Call the close method from PendulumRenderFix

    def _plot_image(self, image):
        """Helper function to plot an image and wait for user to close it before proceeding."""
        plt.imshow(image)
        plt.title("Raw Observation from PendulumVisual")
        plt.axis("off")  # Hide axes
        plt.show()  # Block execution until the plot is closed


class PendulumVisual(PendulumRenderFix):
    """
    Gym's Pendulum environment modified to provide image-based observations instead of direct state measurements.
    Inherits from PendulumRenderFix to fix rendering issues.
    """

    def __init__(
        self,
        render_mode="rgb_array",
        costs_coefs: tuple[float, float, float] = (1.0, 0.1, 0.001),
        render_during_training=False,
    ):
        super(PendulumVisual, self).__init__(
            render_mode=render_mode, costs_coefs=costs_coefs
        )  # Pass render_mode to the parent class

        self.render_mode = render_mode  # Set the render mode

        # Update the observation space to use image dimensions
        image_shape = (500, 500, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=image_shape, dtype=np.uint8
        )

    def reset(self, *, seed: int = None, options: dict = None):
        # Reset using the custom environment's method
        obs, info = super().reset(seed=seed, options=options)

        # print(f"Raw observation from PendulumVisual: Min={obs.min()}, Max={obs.max()}, Shape={obs.shape}")

        image = self.render()  # Get the image-based observation

        # Debug: Plot raw image
        # self._plot_image(image)

        # Render image for the agent if in "rgb_array" mode
        if self.render_mode == "rgb_array":
            image = self.render()
            # Check if the image is valid
            if image is None or image.size == 0:
                raise ValueError("Rendered image in reset() is empty or None.")
            return image, info
        else:
            # If in "human" mode, just return a placeholder observation
            return obs, info

    def step(self, action):
        # Step using the custom environment's method
        obs, reward, done, truncated, info = super().step(action)
        # print(f"Render mode is {self.render_mode}")

        # Ensure reward is a scalar for single environments or 1D for vectorized
        if isinstance(reward, np.ndarray):
            reward = np.squeeze(reward)  # Remove singleton dimensions

        # Render image for the agent if in "rgb_array" mode
        if self.render_mode == "rgb_array":
            image = self.render()

            # print(f"Raw observation from PendulumVisual: Min={image.min()}, Max={image.max()}, Shape={image.shape}")

            # print(f"Final preprocessed observation before CNN: Min={image.min()}, Max={image.max()}")

            # Debug: Plot raw image
            # self._plot_image(image)

            # Check if the image is valid
            if image is None or image.size == 0:
                raise ValueError("Rendered image in step() is empty or None.")
            return image, reward, done, truncated, info
        else:
            # If in "human" mode, just return a placeholder observation
            return obs, reward, done, truncated, info

    def render(self):
        return super().render()  # Call the render method from PendulumRenderFix

    def close(self):
        super().close()  # Call the close method from PendulumRenderFix

    def _plot_image(self, image):
        """Helper function to plot an image and wait for user to close it before proceeding."""
        plt.imshow(image)
        plt.title("Raw Observation from PendulumVisual")
        plt.axis("off")  # Hide axes
        plt.show()  # Block execution until the plot is closed
