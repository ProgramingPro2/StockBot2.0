import numpy as np
import tensorflow as tf
import os
import logging

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """A class for storing and retrieving experiences for training a reinforcement learning agent."""

    def __init__(self, capacity, state_dim, action_dim):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity: The maximum number of experiences to store.
            state_dim: The dimension of the state space.
            action_dim: The dimension of the action space.
        """
        self.capacity = capacity
        self.position = 0
        self.size = 0

        # Pre-allocate memory for better performance
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode is done.
        """
        # Direct assignment to pre-allocated arrays
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def as_dataset(self, batch_size):
        """
        Creates a tf.data.Dataset from the replay buffer.

        Args:
            batch_size: The batch size for the dataset.

        Returns:
            A tf.data.Dataset object or None if the buffer is empty.
        """
        if self.size == 0:
            return None

        # Create a dataset from the stored transitions
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.states[: self.size],
                self.actions[: self.size],
                self.rewards[: self.size],
                self.next_states[: self.size],
                self.dones[: self.size],
            )
        )

        # Shuffle, repeat, and batch the data
        dataset = (
            dataset.shuffle(buffer_size=self.size, reshuffle_each_iteration=True)
            .repeat()
            .batch(batch_size)
        )

        # Prefetch data to the GPU
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def __len__(self):
        """Returns the current size of the replay buffer."""
        return self.size

    def save(self, path):
        """Save replay buffer to disk"""
        buffer_data = {
            "states": self.states[: self.size],
            "actions": self.actions[: self.size],
            "rewards": self.rewards[: self.size],
            "next_states": self.next_states[: self.size],
            "dones": self.dones[: self.size],
            "position": self.position,
            "size": self.size,
        }
        np.savez_compressed(path, **buffer_data)

    def load(self, path):
        """Load replay buffer from disk with improved error handling"""
        if not os.path.exists(path):
            logger.info(f"Replay buffer file not found: {path}")
            return False

        try:
            data = np.load(path, allow_pickle=True)

            # Check if all expected keys are present
            expected_keys = [
                "states",
                "actions",
                "rewards",
                "next_states",
                "dones",
                "position",
                "size",
            ]
            missing_keys = [key for key in expected_keys if key not in data]
            if missing_keys:
                logger.error(
                    f"Corrupted replay buffer file: {path}. Missing keys: {missing_keys}"
                )
                self._reset_buffer()
                return False

            size = min(int(data["size"]), self.capacity)
            if size <= 0:
                logger.warning(f"Replay buffer file {path} was empty or had size 0.")
                self._reset_buffer()
                return False

            # Validate shapes before loading
            try:
                self._validate_shapes(data, size)
            except ValueError as e:
                logger.error(f"Shape validation failed: {e}")
                self._reset_buffer()
                return False

            # Load data with shape validation
            self._load_validated_data(data, size)
            
            logger.info(f"Replay buffer loaded {self.size} transitions from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading replay buffer from {path}: {e}")
            self._reset_buffer()
            return False

    def _validate_shapes(self, data, size):
        """Validate that loaded data has compatible shapes"""
        expected_state_shape = (size, self.states.shape[1])
        expected_action_shape = (size, self.actions.shape[1])
        
        if data["states"].shape[:2] != expected_state_shape:
            raise ValueError(
                f"State shape mismatch: expected {expected_state_shape}, got {data['states'].shape[:2]}"
            )
        
        if data["actions"].shape[:2] != expected_action_shape:
            raise ValueError(
                f"Action shape mismatch: expected {expected_action_shape}, got {data['actions'].shape[:2]}"
            )
        
        # Check other arrays have correct length
        for key in ["rewards", "next_states", "dones"]:
            if len(data[key]) != size:
                raise ValueError(f"{key} length mismatch: expected {size}, got {len(data[key])}")

    def _load_validated_data(self, data, size):
        """Load data after validation has passed"""
        self.states[:size] = data["states"][:size]
        self.actions[:size] = data["actions"][:size]
        self.rewards[:size] = data["rewards"][:size]
        self.next_states[:size] = data["next_states"][:size]
        self.dones[:size] = data["dones"][:size]
        
        self.position = int(data["position"]) % self.capacity
        self.size = size

    def _reset_buffer(self):
        """Reset buffer to empty state"""
        self.position = 0
        self.size = 0
