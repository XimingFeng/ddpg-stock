import gym
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from replay_buffer import Buffer
from tensorflow.keras import activations
import pandas as pd


class ActorCritic:

    def __init__(self, state_dim, action_dim, gamma=0.99,
                 actor_hp_param=None, critic_hp_param=None):
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
        print("Actor Network Summary: ")
        self.main_actor.summary()
        print("Critic Network Summary: ")
        self.main_critic.summary()

    def train(self, env, num_eps, actor_lr, critic_lr, train_every_step=1, batch_size=1, verbose=True):
        replay_buffer = Buffer(self.state_dim, self.action_dim, batch_size=batch_size)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        rewards = []
        for eps in range(num_eps):
            current_state = env.reset()
            step_count = 0
            done = False
            episodic_reward = 0
            while done is False:
                # print(f"current state: {current_state}")
                current_state_tf = tf.convert_to_tensor(current_state)
                actor_out = np.squeeze(self.main_actor(current_state_tf))
                reward, done, nxt_state = env.step(actor_out)
                episodic_reward += reward
                replay_buffer.insert(current_state, actor_out, reward, nxt_state, done)
                if verbose:
                    print(f"step {step_count}, take action {actor_out}, receive reward {reward}")
                if step_count % train_every_step == 0:
                    train_batch = replay_buffer.sample_batch()
                    self.one_step_train(train_batch, actor_optimizer, critic_optimizer)
                current_state = nxt_state
                step_count += 1
            replay_buffer.reset_buffer()
            rewards.append(episodic_reward)
            print(f"Episode {eps}, reward: {episodic_reward}")
        return rewards

    def test(self, env):
        current_state = env.reset()
        done = False
        current_state_tf = tf.convert_to_tensor(current_state)
        action = self.main_actor(current_state_tf)
        reward, done, nxt_state = env.step(action)
        env.render()

        # while done is False:
        #     # print(f"-------------- step {step_count} --------------------")
        #     current_state_tf = tf.convert_to_tensor(current_state)
        #     nxt_state, reward, done, info = env.step(current_state_tf)
        #     current_state = nxt_state
        #     env.render()
        # env.close()

    def one_step_train(self, batch_replay, actor_optimizer, critic_optimizer):
        states = tf.cast(tf.convert_to_tensor(batch_replay[0]), tf.float32)
        actions = tf.cast(tf.convert_to_tensor(batch_replay[1]), tf.float32)
        rewards = tf.cast(tf.convert_to_tensor(batch_replay[2]), tf.float32)
        next_states = tf.cast(tf.convert_to_tensor(batch_replay[3]), tf.float32)
        is_term = tf.cast(tf.convert_to_tensor(batch_replay[4]), tf.float32)
        # print(f'State: {states}')
        # print(f"Action: {actions}")
        # print(f"Rewards:  {rewards}")
        # print(f"Nest States: {next_states}")
        # print(f"Is terminal states: {is_term}")
        # Update the parameters of main critic to minimize Mean Squared Bellman Error
        with tf.GradientTape() as tape:
            target_nxt_actions = self.target_actor(next_states)
            target_nxt_value = self.target_critic([next_states, target_nxt_actions])
            # print(f"target critic: {target_nxt_value}")
            target_value = rewards + self.gamma * (1 - is_term) * target_nxt_value
            main_value = self.main_critic([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(main_value - target_value))
        critic_gradient = tape.gradient(critic_loss, self.main_critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_gradient, self.main_critic.trainable_variables))

        # Update the parameters of main actor to maximize action-value function
        with tf.GradientTape() as tape:
            main_action = self.main_actor(states)
            main_value = -tf.math.reduce_mean(self.main_critic([states, main_action]))
        actor_gradient = tape.gradient(main_value, self.main_actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_gradient, self.main_actor.trainable_variables))

        # Update target network weights
        self.update_target_weights()

    def update_target_weights(self, rho=0.995):
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
        out = layers.Softmax()(out)
        return Model(inputs, out)

    def build_critic(self):
        # state as input
        state_input = layers.Input(shape=self.state_dim, dtype=tf.float32)
        state_out = self.build_residual_block(state_input)

        # action as input
        action_input = layers.Input(shape=self.action_dim, dtype=tf.float32)

        state_action_sum = state_out + action_input

        out = layers.Dense(512, activation="relu")(state_action_sum)
        # out = layers.BatchNormalization()(out)
        outputs = layers.Dense(1)(out)
        return Model([state_input, action_input], outputs)


    def build_residual_block(self, block_input):
        # conv1 out: (m, n - 2, 2)
        out = layers.Conv2D(filters=2,
                            kernel_size=(1, 3),
                            strides=1,
                            padding="VALID",
                            activation=activations.relu)(block_input)

        # conv2 out: (m, 1, 21)
        out = layers.Conv2D(filters=21,
                            kernel_size=(1, 48),
                            strides=1,
                            padding="VALID",
                            activation=activations.relu)(out)

        # con3 out: (m, 1, 1)
        out = layers.Conv2D(filters=1,
                            kernel_size=1,
                            strides=1,
                            padding="VALID", activation=activations.relu)(out)

        out = layers.Flatten()(out)
        out = layers.Dense(self.num_assets)(out)
        return out