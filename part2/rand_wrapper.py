from collections import deque

import gymnasium as gym
import numpy as np


class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies domain randomization to the cube mass.

    Modes:
        none : fixed mass, no randomization
        udr  : Uniform Domain Randomization — sample mass uniformly from [mass_min, mass_max]
        adr  : Automatic Domain Randomization — adaptively expand [mass_min, mass_max]
                based on agent performance at the boundaries
    """

    def __init__(
        self,
        env,
        mass_range=(0.5, 6.0),
        mode="none",
        # ADR hyperparameters
        adr_step=0.2,
        adr_threshold=0.5,
        adr_buffer_size=50,
        adr_p_boundary=0.1,
        mass_min_limit=0.1,
        mass_max_limit=10.0,
    ):
        super().__init__(env)

        self.mode = mode

        # UDR and ADR share these boundaries
        self.mass_min = float(mass_range[0])
        self.mass_max = float(mass_range[1])

        # Global hard limits — boundaries never go outside these
        self.mass_min_limit = mass_min_limit
        self.mass_max_limit = mass_max_limit

        # ADR-specific parameters
        self.adr_step = adr_step            # how much to expand boundaries per update
        self.adr_threshold = adr_threshold  # performance threshold to trigger expansion
        self.adr_p_boundary = adr_p_boundary  # probability of sampling from boundary

        # Buffers tracking episode returns at lower/upper boundaries
        self.lower_buffer = deque(maxlen=adr_buffer_size)
        self.upper_buffer = deque(maxlen=adr_buffer_size)

        # Tracks what type of sample was used this episode
        # Values: "lower", "upper", "middle", or None (for none/udr)
        self.last_sample_type = None

        # Tracks cumulative reward for the current episode (used by ADR)
        self._episode_return = 0.0

    # -----------------------------------------------------------------
    # Mass Sampling
    # -----------------------------------------------------------------

    def _sample_mass(self):
        """Sample a new mass according to the current mode."""

        if self.mode == "none":
            return None

        elif self.mode == "udr":
            # Uniform Domain Randomization: sample uniformly from [mass_min, mass_max]
            self.last_sample_type = "middle"
            return float(np.random.uniform(self.mass_min, self.mass_max))

        elif self.mode == "adr":
            # With probability p_boundary, sample from one of the two boundaries
            # Otherwise sample uniformly from the interior
            r = np.random.random()
            if r < self.adr_p_boundary / 2:
                self.last_sample_type = "lower"
                return float(self.mass_min)
            elif r < self.adr_p_boundary:
                self.last_sample_type = "upper"
                return float(self.mass_max)
            else:
                self.last_sample_type = "middle"
                return float(np.random.uniform(self.mass_min, self.mass_max))

        else:
            raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented.")

    # -----------------------------------------------------------------
    # ADR boundary update
    # -----------------------------------------------------------------

    def _update_adr_boundaries(self, episode_return):
        """
        Update ADR boundaries based on agent performance at the boundaries.
        Called at the end of each episode.
        """
        if self.mode != "adr":
            return

        # Record the episode return in the appropriate buffer
        if self.last_sample_type == "lower":
            self.lower_buffer.append(episode_return)
        elif self.last_sample_type == "upper":
            self.upper_buffer.append(episode_return)

        # Only update when buffers are full
        if len(self.lower_buffer) == self.lower_buffer.maxlen:
            mean_lower = float(np.mean(self.lower_buffer))
            if mean_lower > self.adr_threshold:
                # Agent performs well at lower boundary — expand downward
                self.mass_min = max(
                    self.mass_min_limit,
                    self.mass_min - self.adr_step
                )
                self.lower_buffer.clear()

        if len(self.upper_buffer) == self.upper_buffer.maxlen:
            mean_upper = float(np.mean(self.upper_buffer))
            if mean_upper > self.adr_threshold:
                # Agent performs well at upper boundary — expand upward
                self.mass_max = min(
                    self.mass_max_limit,
                    self.mass_max + self.adr_step
                )
                self.upper_buffer.clear()

    # -----------------------------------------------------------------
    # Gymnasium interface
    # -----------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Accumulate episode return for ADR boundary updates
        self._episode_return += float(reward)

        done = terminated or truncated
        if done and self.mode == "adr":
            self._update_adr_boundaries(self._episode_return)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset episode return tracker
        self._episode_return = 0.0

        # Sample new mass
        new_mass = self._sample_mass()

        if new_mass is not None:
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            print(
                f"[{self.mode}] mass={new_mass:.3f}kg "
                f"range=[{self.mass_min:.2f}, {self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
            )

        return super().reset(**kwargs)
