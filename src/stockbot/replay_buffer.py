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

        # GPU acceleration setup - only use if memory allows
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        self.use_gpu_tensors = False
        
        # Calculate memory requirements
        total_memory_needed = capacity * (state_dim + action_dim + 2) * 4  # 4 bytes per float32
        gpu_memory_available = 8 * 1024 * 1024 * 1024  # 8GB in bytes
        
        if self.device == '/GPU:0' and total_memory_needed < gpu_memory_available * 0.5:  # Use 50% of GPU memory max
            try:
                # Pre-allocate GPU tensors for faster operations
                with tf.device(self.device):
                    self.states_gpu = tf.Variable(tf.zeros((capacity, state_dim), dtype=tf.float32))
                    self.actions_gpu = tf.Variable(tf.zeros((capacity, action_dim), dtype=tf.float32))
                    self.rewards_gpu = tf.Variable(tf.zeros(capacity, dtype=tf.float32))
                    self.next_states_gpu = tf.Variable(tf.zeros((capacity, state_dim), dtype=tf.float32))
                    self.dones_gpu = tf.Variable(tf.zeros(capacity, dtype=tf.float32))
                self.use_gpu_tensors = True
                print(f"Using GPU tensors for replay buffer. Memory usage: {total_memory_needed / (1024**3):.2f} GB")
            except Exception as e:
                print(f"Failed to allocate GPU tensors, falling back to CPU: {e}")
                self.use_gpu_tensors = False
        else:
            print(f"Using CPU tensors for replay buffer. Memory needed: {total_memory_needed / (1024**3):.2f} GB")
            self.use_gpu_tensors = False

    def push(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer with GPU acceleration.

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

        # GPU acceleration: Update GPU tensors only if available
        if self.use_gpu_tensors:
            try:
                with tf.device(self.device):
                    self.states_gpu[self.position].assign(state)
                    self.actions_gpu[self.position].assign(action)
                    self.rewards_gpu[self.position].assign(reward)
                    self.next_states_gpu[self.position].assign(next_state)
                    self.dones_gpu[self.position].assign(float(done))
            except Exception as e:
                # If GPU operations fail, disable GPU tensors
                print(f"GPU tensor operation failed, disabling GPU tensors: {e}")
                self.use_gpu_tensors = False

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def as_dataset(self, batch_size):
        """
        Creates a tf.data.Dataset from the replay buffer with optimized performance.

        Args:
            batch_size: The batch size for the dataset.

        Returns:
            A tf.data.Dataset iterator for training.
        """
        if self.size < batch_size:
            return None

        # Use GPU tensors if available, otherwise use CPU
        if self.use_gpu_tensors:
            try:
                with tf.device(self.device):
                    # Create dataset from GPU tensors
                    dataset = tf.data.Dataset.from_tensor_slices((
                        self.states_gpu[:self.size],
                        self.actions_gpu[:self.size],
                        self.rewards_gpu[:self.size],
                        self.next_states_gpu[:self.size],
                        self.dones_gpu[:self.size]
                    ))
            except Exception as e:
                print(f"GPU dataset creation failed, falling back to CPU: {e}")
                self.use_gpu_tensors = False
                # Fall through to CPU implementation
                
        if not self.use_gpu_tensors:
            # Create dataset from CPU tensors
            dataset = tf.data.Dataset.from_tensor_slices((
                self.states[:self.size],
                self.actions[:self.size],
                self.rewards[:self.size],
                self.next_states[:self.size],
                self.dones[:self.size]
            ))

        # Optimize dataset performance
        dataset = dataset.shuffle(buffer_size=min(10000, self.size))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Create iterator and clear any previous iterators to prevent memory leaks
        if hasattr(self, '_current_iterator'):
            del self._current_iterator
        
        self._current_iterator = iter(dataset)
        return self._current_iterator

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
