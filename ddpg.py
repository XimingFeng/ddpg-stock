import gym
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from replay_buffer import Buffer
from tensorflow.keras import activations
from scipy.special import softmax
import pandas as pd


class ActorCritic:

    def __init__(self, state_dim, action_dim, gamma=1.0,
                 actor_hp_param=None, critic_hp_param=None, verbose=True):
        self.gamma = tf.convert_to_tensor(gamma)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_assets = state_dim[0] + 1  # Non-cash asset + cash. m in paper
        self.window_len = state_dim[1]  # n in paper
        self.num_features = state_dim[2]  # f in paper
        self.main_actor = self.build_actor()
        self.main_critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.main_actor.set_weights(self.target_actor.get_weights())
        self.main_critic.set_weights(self.target_critic.get_weights())
        if verbose:
            print("Actor Network Summary: ")
            self.main_actor.summary()
            print("Critic Network Summary: ")
            self.main_critic.summary()

    def train(self, env, num_eps, actor_lr, critic_lr, train_every_step=1, batch_size=16, rho=0.999, verbose=True):
        replay_buffer = Buffer(self.state_dim, self.action_dim)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        rewards_history = []  # include total rewards in each episode
        detail_reward_history = []  # include rewards for each step in each episode
        loss_history = []
        detail_loss_history = []
        for eps in range(num_eps):
            current_state = env.reset()
            step_count = 0
            done = False
            replay_buffer.reset_buffer()
            episodic_reward = 0
            single_eps_rewards = []
            episodic_loss = 0
            single_eps_loss = []
            while done is False:
                # print(f"current state: {current_state}")
                current_state_tf = tf.convert_to_tensor(current_state)
                actor_out = np.squeeze(self.main_actor(current_state_tf))
                actor_out = self.add_noise(actor_out)
                reward, done, nxt_state = env.step(actor_out)
                episodic_reward += reward
                replay_buffer.insert(current_state, actor_out, reward, nxt_state, done)
                single_eps_rewards.append(episodic_reward)
                if verbose:
                    print(f"step {step_count}, take action {actor_out}, receive reward {reward}")
                if step_count % train_every_step == 0:
                    train_batch = replay_buffer.sample_batch(batch_size=batch_size)
                    critic_loss = self.one_step_train(train_batch, actor_optimizer, critic_optimizer, rho)
                    single_eps_loss.append(critic_loss)
                    episodic_loss += critic_loss
                current_state = nxt_state
                step_count += 1
            rewards_history.append(episodic_reward)
            detail_reward_history.append(single_eps_rewards)
            loss_history.append(episodic_loss)
            detail_loss_history.append(single_eps_loss)
            print(f"Episode {eps} >>>>>>>>>>>>>>>>>>> reward: {episodic_reward}")
        return rewards_history, detail_reward_history, loss_history, detail_loss_history

    def add_noise(self, action):
        noise = np.random.normal(0, 0.5, self.action_dim)
        # print(noise)
        action_noisy = softmax(action + noise)
        return action_noisy

    def one_step_train(self, batch_replay, actor_optimizer, critic_optimizer, rho):
        states = tf.cast(tf.convert_to_tensor(batch_replay[0]), tf.float32)
        actions = tf.cast(tf.convert_to_tensor(batch_replay[1]), tf.float32)
        rewards = tf.cast(tf.convert_to_tensor(batch_replay[2]), tf.float32)
        next_states = tf.cast(tf.convert_to_tensor(batch_replay[3]), tf.float32)
        is_term = tf.cast(tf.convert_to_tensor(batch_replay[4]), tf.float32)
        # print(f'State: {states}')
        # print(f'State shape: {states.shape}')
        # print(f"Action: {actions}")
        # print(f"Rewards:  {rewards}")
        # print(f"Next States: {next_states}")
        # print(f"Is terminal states: {is_term}")
        # Update the parameters of main critic to minimize Mean Squared Bellman Error
        with tf.GradientTape() as tape:
            target_nxt_actions = self.target_actor(next_states)
            target_nxt_value = self.target_critic([next_states, target_nxt_actions])
            # print(f"target critic: {target_nxt_value}")
            target_value = rewards + self.gamma * (1 - is_term) * target_nxt_value
            main_value = self.main_critic([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(main_value - target_value))
        # print(f'target value: {target_value}, value from the main critic: {main_value}')
        # print(f"critic loss {critic_loss}")
        critic_gradient = tape.gradient(critic_loss, self.main_critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_gradient, self.main_critic.trainable_variables))

        # Update the parameters of main actor to maximize action-value function
        with tf.GradientTape() as tape:
            main_action = self.main_actor(states)
            actor_loss = tf.math.reduce_mean(-self.main_critic([states, main_action]))
        # print(f'actor loss {actor_loss}')
        actor_gradient = tape.gradient(actor_loss, self.main_actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_gradient, self.main_actor.trainable_variables))

        # Update target network weights
        self.update_target_weights(rho)
        return critic_loss.numpy()

    def update_target_weights(self, rho):
        # Update target critic weights a little bit towards main critic's weights
        new_weights = []
        target_critic_weights = self.target_critic.weights
        for i, weight in enumerate(self.main_critic.weights):
            new_weights.append(rho * target_critic_weights[i] + (1 - rho) * weight)
        self.target_critic.set_weights(new_weights)

        # Update target actor weights a little bit towards main actor's weights
        new_weights = []
        target_actor_weights = self.target_actor.weights
        for i, weight in enumerate(self.main_actor.weights):
            new_weights.append(rho * target_actor_weights[i] + (1 - rho) * weight)
        self.target_actor.set_weights(new_weights)

    def build_actor(self):
        # input shape (1, m, n, f)
        inputs = layers.Input(shape=self.state_dim, dtype=tf.float32)

        out = self.build_residual_block(inputs)
        out = layers.Flatten()(out)
        out = layers.Dense(256, activation=activations.relu)(out)
        out = layers.Dropout(0.5)(out)
        out = layers.Dense(self.num_assets, dtype=tf.float32, activation=activations.softmax)(out)
        # out = layers.Flatten()(inputs)
        # out = layers.Dense(self.num_assets)(out)
        # out = layers.Softmax()(out)
        return Model(inputs, out)

    def build_critic(self):
        # state as input
        state_input = layers.Input(shape=self.state_dim, dtype=tf.float32)
        state_out = self.build_residual_block(state_input)
        state_out = layers.Flatten()(state_out)
        state_out = layers.Dense(256, activation=activations.relu)(state_out)

        # action as input
        action_input = layers.Input(shape=self.action_dim, dtype=tf.float32)
        action_out = layers.Dense(256, activation=activations.relu)(action_input)

        critic_out = layers.Add()([state_out, action_out])

        critic_out = layers.Dense(512)(critic_out)
        critic_out = layers.BatchNormalization()(critic_out)
        critic_out = layers.ReLU()(critic_out)
        critic_out = layers.Dropout(0.5)(critic_out)
        critic_out = layers.Dense(1)(critic_out)
        return Model([state_input, action_input], critic_out)

    def build_residual_block(self, block_input):
        shortcut_input = block_input

        # main path
        out = layers.Conv2D(filters=32,
                            kernel_size=1,
                            strides=1,
                            padding="SAME")(block_input)
        out = layers.BatchNormalization()(out)
        out = layers.ReLU()(out)

        out = layers.Conv2D(filters=32,
                            kernel_size=1,
                            strides=1,
                            padding="SAME")(out)
        out = layers.BatchNormalization()(out)
        out = layers.ReLU()(out)

        out = layers.Conv2D(filters=32,
                            kernel_size=1,
                            strides=1,
                            padding="SAME")(out)
        out = layers.BatchNormalization()(out)
        out = layers.ReLU()(out)

        # shortcut path
        shortcut_out = layers.Conv2D(filters=32,
                            kernel_size=1,
                            strides=1,
                            padding="SAME")(shortcut_input)
        shortcut_out = layers.BatchNormalization()(shortcut_out)

        out = layers.Add()([out, shortcut_out])
        out = layers.ReLU()(out)
        return out