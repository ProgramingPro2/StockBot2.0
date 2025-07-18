import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Dense,
    concatenate,
    BatchNormalization,
)
import numpy as np
from pathlib import Path
import logging

from noise import OrnsteinUhlenbeckActionNoise
from replay_buffer import ReplayBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a"
)
logger = logging.getLogger(__name__)

class DRLAgent:
    """A Deep Reinforcement Learning Agent using Actor-Critic architecture."""

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate,
        gamma,
        memory_size,
        noise_sigma,
        noise_theta,
    ):
        """
        Initializes the DRLAgent.

        Args:
            state_dim: The dimension of the state space.
            action_dim: The dimension of the action space.
            learning_rate: The learning rate for the optimizers.
            gamma: The discount factor for future rewards.
            memory_size: The capacity of the replay buffer.
            noise_sigma: The sigma parameter for the Ornstein-Uhlenbeck noise.
            noise_theta: The theta parameter for the Ornstein-Uhlenbeck noise.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Use the optimized replay buffer
        self.memory = ReplayBuffer(memory_size, state_dim, action_dim)

        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(action_dim), sigma=noise_sigma, theta=noise_theta
        )

        # Set default TensorFlow dtype to float32 explicitly
        tf.keras.backend.set_floatx("float32")

        # Create separate optimizers for actor and critic
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Build models
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # Create target networks for more stable learning
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()

        # Initialize target networks with same weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # TensorFlow checkpoint for saving models
        self.checkpoint_dir = "./checkpoints"
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        self.actor_checkpoint = tf.train.Checkpoint(
            model=self.actor, optimizer=self.actor_optimizer
        )
        self.critic_checkpoint = tf.train.Checkpoint(
            model=self.critic, optimizer=self.critic_optimizer
        )
        self.actor_manager = tf.train.CheckpointManager(
            self.actor_checkpoint,
            directory=f"{self.checkpoint_dir}/actor",
            max_to_keep=3,
        )
        self.critic_manager = tf.train.CheckpointManager(
            self.critic_checkpoint,
            directory=f"{self.checkpoint_dir}/critic",
            max_to_keep=3,
        )

    def _build_actor(self):
        """Build policy network (actor) that selects actions with GPU optimizations."""
        with tf.device('/GPU:0'):
            state_input = Input(shape=(self.state_dim,))
            x = Dense(
                512,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(state_input)
            x = BatchNormalization(dtype="float32")(x)
            x = Dense(
                256,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(x)
            x = BatchNormalization(dtype="float32")(x)
            x = Dense(
                128,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(x)
            output = Dense(
                self.action_dim,
                activation="tanh",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
            )(x)
            model = Model(inputs=state_input, outputs=output)
            model.compile(loss="mse", optimizer=self.actor_optimizer)
            return model

    def _build_critic(self):
        """Build value network (critic) that estimates Q-values with GPU optimizations."""
        with tf.device('/GPU:0'):
            state_input = Input(shape=(self.state_dim,))
            action_input = Input(shape=(self.action_dim,))

            # State branch with batch normalization
            state_h1 = Dense(
                512, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
            )(state_input)
            state_bn1 = BatchNormalization(dtype="float32")(state_h1)
            state_h2 = Dense(
                256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
            )(state_bn1)
            state_bn2 = BatchNormalization(dtype="float32")(state_h2)

            # Action branch
            action_h1 = Dense(
                256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
            )(action_input)
            action_bn1 = BatchNormalization(dtype="float32")(action_h1)

            # Merge
            merged = concatenate([state_bn2, action_bn1])
            merged_h1 = Dense(
                256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
            )(merged)
            merged_bn1 = BatchNormalization(dtype="float32")(merged_h1)
            merged_h2 = Dense(
                128, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
            )(merged_bn1)
            output = Dense(
                1,
                activation="linear",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
            )(merged_h2)

            model = Model(inputs=[state_input, action_input], outputs=output)
            model.compile(loss="mse", optimizer=self.critic_optimizer)
            return model

    @tf.function
    def _actor_train_step(self, states):
        """
        Performs a single training step for the actor network.

        Args:
            states: A TensorFlow tensor representing the current states.

        Returns:
            The actor loss as a TensorFlow scalar.
        """
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states, training=True)
            actor_loss = -tf.reduce_mean(
                self.critic([states, actions_pred], training=True)
            )

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # Apply gradient clipping
        actor_grads = [(tf.clip_by_norm(grad, 1.0)) for grad in actor_grads]
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )
        return actor_loss

    @tf.function
    def _critic_train_step(self, states, actions, q_targets):
        """
        Performs a single training step for the critic network.

        Args:
            states: A TensorFlow tensor representing the current states.
            actions: A TensorFlow tensor representing the actions taken.
            q_targets: A TensorFlow tensor representing the target Q-values.

        Returns:
            The critic loss as a TensorFlow scalar.
        """
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            # Ensure q_values is float32 for calculation with q_targets
            q_values_float32 = tf.cast(q_values, tf.float32)
            critic_loss = tf.reduce_mean(tf.square(q_targets - q_values_float32))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # Apply gradient clipping
        critic_grads = [(tf.clip_by_norm(grad, 1.0)) for grad in critic_grads]
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables)
        )
        return critic_loss

    def act(self, state, explore=True):
        """
        Chooses an action based on the current state.

        Args:
            state: The current state of the environment.
            explore: A boolean indicating whether to add exploration noise.

        Returns:
            An array representing the chosen action.
        """
        if explore and np.random.rand() <= self.epsilon:
            # Exploration: random actions
            return np.random.uniform(-1, 1, self.action_dim)

        # Exploitation: use the actor to predict actions
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action_output = self.actor(state_tensor).numpy()[0]

        # Add exploration noise
        if explore:
            action_output += self.noise()

        return action_output

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def train(self, dataset):
        """
        Trains the agent using a batch from the dataset.

        Args:
            dataset: A tf.data.Dataset iterator providing batches of experiences.

        Returns:
            A dictionary containing the critic and actor losses.
        """
        if dataset is None:
            return

        # Get the next batch from the dataset iterator
        states, actions, rewards, next_states, dones = next(dataset)

        # Ensure tensors have the correct data type
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        next_states = tf.cast(next_states, tf.float32)
        dones = tf.cast(dones, tf.float32)

        # Calculate target Q-values
        next_actions = self.target_actor(next_states, training=False)
        q_next = self.target_critic([next_states, next_actions], training=False)
        q_next = tf.cast(q_next, tf.float32)
        q_targets = rewards + (1 - dones) * self.gamma * q_next[:, 0]

        # Train critic and actor
        critic_loss = self._critic_train_step(
            states, actions, tf.expand_dims(q_targets, axis=1)
        )
        actor_loss = self._actor_train_step(states)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Soft update target networks
        self._update_target_networks()

        actor_loss_val = actor_loss.numpy()
        critic_loss_val = critic_loss.numpy()

        if np.isnan(actor_loss_val) or np.isnan(critic_loss_val):
            logger.warning(f"NaN loss detected: actor_loss={actor_loss_val}, critic_loss={critic_loss_val}")

        return {"critic_loss": critic_loss_val, "actor_loss": actor_loss_val}

    def _update_target_networks(self, tau=0.005):
        """Soft updates the target actor and critic networks."""
        for target_param, param in zip(
            self.target_actor.variables, self.actor.variables
        ):
            target_param.assign(param * tau + target_param * (1 - tau))

        for target_param, param in zip(
            self.target_critic.variables, self.critic.variables
        ):
            target_param.assign(param * tau + target_param * (1 - tau))

    def save_models(self, path=None):
        """
        Saves the actor and critic models using TensorFlow checkpoints and .keras format.

        Args:
            path: The directory to save the .keras models. If None, only checkpoints are saved.
        """
        # Use .keras format for path-based saving (recommended)
        if path:
            Path(path).mkdir(exist_ok=True)  # Ensure directory exists
            try:
                self.actor.save(f"{path}/actor_model.keras")
                self.critic.save(f"{path}/critic_model.keras")
                logger.info(f"Models saved in .keras format to {path}")
            except Exception as e:
                logger.error(f"Failed to save models in .keras format: {e}")

        # Save in checkpoint format (always do this)
        try:
            actor_path = self.actor_manager.save()
            critic_path = self.critic_manager.save()
            logger.info(
                f"Checkpoint saved: Actor to {actor_path}, Critic to {critic_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoints: {e}")
            actor_path, critic_path = None, None  # Indicate failure

        # Save replay buffer using the correct extension (.npz)
        buffer_path = f"{self.checkpoint_dir}/replay_buffer.npz"
        try:
            self.memory.save(buffer_path)
            logger.info(f"Replay buffer saved to {buffer_path}")
        except Exception as e:
            logger.error(f"Failed to save replay buffer: {e}")

        return actor_path, critic_path

    def load_models(self, path=None):
        """
        Loads the actor and critic models, prioritizing checkpoints, then .keras files.

        Args:
            path: The directory to load the .keras models from. If None, only checkpoints are attempted.

        Returns:
            True if models were successfully loaded, False otherwise.
        """
        checkpoint_loaded = False
        # --- Attempt 1: Load from Checkpoints ---
        if (
            self.actor_manager.latest_checkpoint
            and self.critic_manager.latest_checkpoint
        ):
            try:
                logger.info(
                    f"Attempting to restore models from checkpoints: {self.actor_manager.latest_checkpoint}"
                )
                # Restore model weights using the checkpoint objects
                actor_status = self.actor_checkpoint.restore(
                    self.actor_manager.latest_checkpoint
                )
                critic_status = self.critic_checkpoint.restore(
                    self.critic_manager.latest_checkpoint
                )

                # Verify restoration
                actor_status.assert_existing_objects_matched()
                critic_status.assert_existing_objects_matched()

                logger.info("Models successfully restored from checkpoints.")

                # CRITICAL: Re-initialize optimizers *after* restoring model weights
                # The optimizer state is also restored by the checkpoint if it was saved.
                # If optimizers were part of the checkpoint (they are), restoration handles them.
                # We just need to make sure the agent's optimizer attributes point correctly.
                self.actor_optimizer = self.actor_checkpoint.optimizer
                self.critic_optimizer = self.critic_checkpoint.optimizer

                checkpoint_loaded = True

            except Exception as e:
                logger.warning(
                    f"Checkpoint restoration failed: {e}. Trying other methods..."
                )
                # Fall through to try path-based loading

        # Load replay buffer - always try this after attempting checkpoint load
        buffer_path = f"{self.checkpoint_dir}/replay_buffer.npz"
        if self.memory.load(buffer_path):
            logger.info(f"Replay buffer loaded from {buffer_path}")
        else:
            logger.warning(
                f"Replay buffer not found or failed to load from {buffer_path}"
            )

        if checkpoint_loaded:
            return True  # Successfully loaded from checkpoint

        # --- Attempt 2: Load from .keras files (if path provided) ---
        if path:
            keras_actor_path = f"{path}/actor_model.keras"
            keras_critic_path = f"{path}/critic_model.keras"

            if tf.io.gfile.exists(keras_actor_path) and tf.io.gfile.exists(keras_critic_path):
                logger.info(f"Attempting to load models from .keras files in {path}")
                try:
                    # No custom_objects usually needed for .keras format with standard layers/ops
                    self.actor = load_model(
                        keras_actor_path, compile=False
                    )  # Re-compile after loading
                    self.critic = load_model(keras_critic_path, compile=False)

                    # Re-compile the models with their respective optimizers
                    self.actor.compile(
                        optimizer=self.actor_optimizer, loss="mse"
                    )  # Loss might not matter if only using for inference/gradients
                    self.critic.compile(optimizer=self.critic_optimizer, loss="mse")

                    logger.info(
                        f"Models successfully loaded from .keras files in {path}"
                    )
                    return True
                except Exception as e:
                    logger.error(
                        f"Failed to load models from .keras files in {path}: {e}. Trying H5 fallback..."
                    )
            else:
                logger.info(f".keras models not found in {path}.")

        # --- Failed to load ---
        logger.error("Failed to load models from checkpoints or specified path.")
        return False
