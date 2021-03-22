from ddpg.ddpg_agent import Agent, ReplayBuffer, GaussianNoise
from unityagents.exception import UnityEnvironmentException
from ddpg.ddpg_model import Actor, Critic
from ddpg.ddpg_trainer import DDPGTrainer
from unityagents import UnityEnvironment
from torch import optim
import torch
import time
import sys
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_env(env_loc):
    """
    Initializes the UnityEnviornment based on the running operating system.

    Arguments:
        env_loc: A string designating unity environment directory.

    Returns:
        env: A UnityEnvironment used for Agent evaluation and training.
    """

    # Set path for unity environment based on operating system.
    if sys.platform == 'linux':
        p = os.path.join(env_loc, 'Reacher_Linux/Reacher.x86_64')

    elif sys.platform == 'darwin':
        p = os.path.join(env_loc, 'Reacher')
    else:
        p = os.path.join(env_loc, 'Reacher_Windows_x86_64/Reacher.exe')

    # Initialize unity environment, return message if error thrown.
    try:
        env = UnityEnvironment(file_name=p)

    except UnityEnvironmentException:
        print('\nEnvironment not found or Operating System not supported.\n')
        sys.exit()

    # Extract state dimensionality from env.
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = env_info.vector_observations.shape[-1]

    # Extract action dimensionality and number of agents from env.
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)

    # Display relevant environment information.
    print('Number of agents: {}, state size: {}, action size: {}'.format(
        num_agents, state_size, action_size))

    return env, state_size, action_size


def create_agent(state_size, action_size, seed=0, actor_fc1_units=400,
                 actor_fc2_units=300, actor_lr=1e-4, critic_fc1_units=400,
                 critic_fc2_units=300, critic_lr=1e-4, weight_decay=0,
                 buffer_size=int(1e5), batch_size=128, gamma=0.99, tau=0.1,
                 noise_dev=0.3):
    """
    This function creates an agent with specified parameters for training.

    Arguments:
        state_size: An integer count of dimensions for each state.
        action_size: An integer count of dimensions for each action.
        seed: Random seed specified to diversify results.
        actor_fc1_units: An integer number of units used in the first FC
            layer for the Actor object.
        actor_fc2_units: An integer number of units used in the second FC
            layer for the Actor object.
        actor_lr: A float designating the learning rate of the Actor's
            optimizer.
        critic_fc1_units: An integer number of units used in the first FC
            layer for the Critic object.
        critic_fc2_units: An integer number of units used in the second FC
            layer for the Critic object.
        critic_lr: A float designating the learning rate of the Critic's
            optimizer.
        weight_decay: Float multiplicative factor to stabilize complexity
            penalization.
        buffer_size: An integer for replay buffer size.
        batch_size: An integer for minibatch size.
        gamma: A float designating the discount factor.
        tau: A float designating multiplication factor for soft update of
            target parameters.
        noise_dev: Float designating the noise to be added to action decisions.

    Returns:
        agent: An Agent object used for training.
    """

    # Initialize the replay buffer from which experiences are gathered for
    # training the agent.
    replay_buffer = ReplayBuffer(
        seed=seed,
        buffer_size=buffer_size,
        batch_size=batch_size
    )

    # Initialize local and target Actor Networks and optimizer.
    actor_local = Actor(state_size, action_size, seed,
                        actor_fc1_units, actor_fc2_units).to(device)
    actor_target = Actor(state_size, action_size, seed,
                         actor_fc1_units, actor_fc2_units).to(device)
    actor_optimizer = optim.Adam(actor_local.parameters(), lr=actor_lr)

    # Initialize local and target Critic Networks and optimizer.
    critic_local = Critic(state_size, action_size, seed,
                          critic_fc1_units, critic_fc2_units).to(device)
    critic_target = Critic(state_size, action_size, seed,
                           critic_fc1_units, critic_fc2_units).to(device)
    critic_optimizer = optim.Adam(critic_local.parameters(), lr=critic_lr,
                                  weight_decay=weight_decay)

    # Initialize Gaussian noise to reduce generalization error.
    noise = GaussianNoise(action_size, seed, mu=0.0, sigma=noise_dev)

    # Create agent object used for training.
    agent = Agent(
        seed=seed,
        memory=replay_buffer,
        batch_size=batch_size,
        actor_local=actor_local,
        actor_target=actor_target,
        actor_optimizer=actor_optimizer,
        critic_local=critic_local,
        critic_target=critic_target,
        critic_optimizer=critic_optimizer,
        noise=noise,
        gamma=gamma,
        tau=tau
    )

    return agent


def create_trainer(env, agent, update_frequency=1, num_updates=5,
                   max_episode_length=1000, save_dir=r'./saved_files',
                   score_window_size=100):
    """
    Initializes trainer to train agents in specified environment.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        agent: An Agent object used for training.
        update_frequency: An integer designating the step frequency of
            updating target network parameters.
        num_updates: Integer number of updates desired for every
            update_frequency steps.
        max_epsiode_length: An integer for maximum number of timesteps per
            episode.
        save_dir: Path designating directory to save resulting files.
        score_window_size: Integer window size used in order to gather
            mean score to evaluate environment solution.

    Returns:
        trainer: A DDPGTrainer object used to train agents in environment.
    """

    # Initialize DDPGTrainer object with relevant arguments.
    trainer = DDPGTrainer(
        env=env,
        agent=agent,
        update_frequency=update_frequency,
        num_updates=num_updates,
        max_episode_length=max_episode_length,
        save_dir=save_dir,
        score_window_size=score_window_size
    )

    return trainer


def train_agents(env, trainer, n_episodes=300, target_score=30,
                 score_window_size=100):
    """
    This function carries out the training process with specified trainer.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        trainer: A DDPGTrainer object used to train agent in environment.
        n_episodes: An integer for maximum number of training episodes.
        target_score: An integer mean target score to be achieved over
            the last score_window_size episodes.
        score_window_size: The integer number of past episode scores
            utilized in order to calculate the current mean score.
    """

    # Initialize timer and notify of training start.
    t_start = time.time()
    print('Starting training...')

    # Train the agent for n_episodes.
    for i in range(1, n_episodes + 1):

        # Step through the training process and notify progress
        trainer.train_step()
        trainer.print_status(put_new_line=False)

        # After every 10 episodes, initiate new line of notification.
        if trainer.i_episode % 10 == 0:
            trainer.print_status(put_new_line=True)

        # If mean score over last score_window_size episodes exceeds
        # target score, plot training progress and save trainer
        if len(trainer.scores) > score_window_size and \
                trainer.get_running_mean_score() > target_score:
            trainer.plt_mavg()
            print('\nEnvironment is solved.')
            print('Saving trainer...')
            trainer.save()
            print('Done.')
            break

    # Close environment and notify of training time.
    print('\nFinished training, closing env')
    env.close()
    t_end = time.time()
    delta = t_end - t_start
    minutes = delta / 60
    print(f'Training took {minutes:.1f} minutes')


if __name__ == '__main__':

    # Initialize environment and extract action and state dimensions.
    env, state_size, action_size = load_env(env_loc=os.getcwd())

    # Create agent used for training.
    agent = create_agent(state_size=state_size, action_size=action_size)

    # Create DDPGTrainer object to train agent.
    trainer = create_trainer(env=env, agent=agent)

    # Train agent in specified environment!
    train_agents(env=env, trainer=trainer)
