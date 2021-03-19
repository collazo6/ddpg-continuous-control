import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

plt.style.use('dark_background')


class DDPGTrainer:
    """
    A class for the implementation and utilization of the training process
    steps for the Deep Deterministic Policy Gradient algorithm.

    Attributes:
        env: A UnityEnvironment used for Agent evaluation and training.
        agent: Agent object being trained in env.
        update_frequency: An integer designating the step frequency of
            updating target network parameters.
        num_updates: Integer number of updates desired for every
            update_frequency steps.
        max_epsiode_length: An integer for maximum number of timesteps per
            episode.
        save_dir: Path designating directory to save resulting files.
        score_window_size: Integer window size used in order to gather
            mean score to evaluate environment solution.
    """

    def __init__(self, env, agent, update_frequency, num_updates,
                 max_episode_length, save_dir, score_window_size):

        # Initialize relevant variables for training
        self.env = env
        self.brain_name = env.brain_names[0]
        self.agent = agent
        self.update_frequency = update_frequency
        self.num_updates = num_updates
        self.max_episode_length = max_episode_length
        self.save_dir = save_dir
        self.score_window_size = score_window_size

        # Initialize episode number and scoring array.
        self.i_episode = 0
        self.scores = []
        self.max_score = -np.inf

    def reset_env(self):
        """Resets environement and returns original state."""

        env_info = self.env.reset()[self.brain_name]
        return env_info.vector_observations

    def step_env(self, actions):
        """
        Realizes actions in environment and returns relevant attributes.

        Parameters:
            actions: Actions array to be realized in the environment.

        Returns:
            states: Array with next state information.
            rewards: Array with rewards information.
            dones: Array with boolean values with 'true' designating the
                episode has finished.
            env_info: BrainInfo object with current environment data.
        """

        # From environment information, extract states and rewards.
        env_info = self.env.step(actions)[self.brain_name]
        states = env_info.vector_observations
        rewards = env_info.rewards

        # Evaluate if episode has finished.
        dones = env_info.local_done

        return states, rewards, dones, env_info

    def run_episode(self, max_episode_length):
        """
        Runs a single episode in the training process for max_episode_length
        timesteps.

        Parameters:
            max_episode_length: Integer number of timesteps in one episode.

        Returns:
            scores: Array with rewards aquired from episode.
        """

        # Restart the environment and gather original states.
        states = self.reset_env()

        # Initialize scores array to hold reward values for each episode.
        scores = np.zeros(states.shape[0])

        # Act and evaluate results and networks for each timestep.
        for t in range(max_episode_length):

            # Act and evaluate results of action.
            actions = self.agent.act(states)
            next_states, rewards, dones, _ = self.step_env(actions)
            dones = np.array(dones).astype(int)

            # Add experiences to memory
            self.agent.add_experience(states, actions, rewards,
                                      next_states, dones)

            # Update networks num_update times each update_frequency timesteps.
            if (t + 1) % self.update_frequency == 0:
                for _ in range(self.num_updates):
                    self.agent.step()

            # Save states and scores and break if training is complete.
            states = next_states
            scores += np.array(rewards)
            if any(dones):
                break

        return scores

    def train_step(self):
        """Steps through each episode and stores mean scores output."""

        self.i_episode += 1
        scores = self.run_episode(self.max_episode_length)
        self.scores.append(scores.mean())

    def get_running_mean_score(self):
        """
        Returns the mean score for the last score_window_size episodes or
        for as many episodes that have been evaluated.
        """

        # If less than score_window_size episodes evaluated, return mean
        # up until that point.
        if len(self.scores) < self.score_window_size:
            return np.mean(self.scores).item()

        # Return mean score for the last score_window_size episodes.
        return np.mean(self.scores[-self.score_window_size:]).item()

    def print_status(self, put_new_line):
        """Displays current episode and average score to terminal."""

        if put_new_line:
            print('\rEpisode {0}\tAverage Score: {1:.2f}'.format(
                self.i_episode, self.get_running_mean_score()))
        else:
            print('\rEpisode {0}\tAverage Score: {1:.2f}'.format(
                self.i_episode, self.get_running_mean_score()), end='')

    def save(self):
        """Saves local network parameters for successful Actor and Critic."""

        torch.save(
            self.agent.actor_local.state_dict(),
            f'{self.save_dir}/checkpoint_actor_{self.i_episode}.pth'
        )

        torch.save(
            self.agent.critic_local.state_dict(),
            f'{self.save_dir}/checkpoint_critic_{self.i_episode}.pth'
        )

    def plt_mavg(self, window_size):
        """Plots moving average score for the last window_size episodes."""

        # Calculate rolling averages based on the last window_size episodes.
        rolling_avgs = pd.DataFrame(self.scores).rolling(window_size).mean()

        # Force index to start at 1 for 1st episode.
        rolling_avgs.index += 1

        # Set coordinates (episode, score) when agent solved env.
        x = self.i_episode
        y = rolling_avgs[0].iloc[-1]

        # Plot rolling averages and save resulting plot
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(rolling_avgs, color='paleturquoise', linewidth=1.5)
        ax.grid(color='w', linewidth=0.2)
        ax.set_title(
            f'Learning Curve: Deep Deterministic Policy Gradient',
            fontsize=30
        )
        ax.set_xlabel('Episode', fontsize=20)
        ax.set_ylabel('Score', fontsize=20)
        plt.tight_layout()
        plt.savefig(rf'{self.save_dir}/scores_mavg_{self.i_episode}')
        plt.show()

        return fig
