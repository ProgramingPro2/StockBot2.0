import numpy as np

class OrnsteinUhlenbeckActionNoise:
    """A class for generating Ornstein-Uhlenbeck process noise."""

    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        """
        Initializes the OrnsteinUhlenbeckActionNoise.

        Args:
            mu: The mean of the noise.
            sigma: The volatility of the noise.
            theta: The rate of mean reversion.
            dt: The time step.
            x0: The initial value of the noise.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Generates the next noise value.

        Returns:
            The next noise value.
        """
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        """Resets the noise to its initial state."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
