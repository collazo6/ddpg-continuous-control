# Deep Deterministic Policy Gradient: Training Agents in Continuous Action Space

Among the most exciting applications for Deep Reinforcement Learning is the implementation of intelligent robotics to carry out the execution of processes normally meant for humans.  One of the largest difficulties in this pursuit remains the complexity of real world environments and training the agent's appropriate interaction with it in order to solve intricate problems.  The Deep Deterministic Policy Gradient (DDPG) algorithm provides a highly effective solution to real world reinforcement learning problems dealing with continuous action spaces.  Built on a deterministic policy gradient algorithm which optimizes a policy based on cumulative reward, DDPG also utilizes experience replay as well as local and target networks for stabilized learning from Deep Q Network methodology.


## Details

The aim of the above implementation is to convey how Deep Reinforcement Learning may be utilized in training multiagent behavior.  Utilizing simulations of robotic arms, the agents are trained on their ability to remain inside of a desired location for as long as possible.


<br />

<div align="center">
  <img width="400" height="200" src="saved_files/untrained_agents.gif">
  &nbsp;&nbsp;&nbsp;
  <img width="400" height="200" src="saved_files/trained_agents.gif">
</div>

<br />

<div align="center">
  <img width="550" height="423" img src="saved_files/scores_mavg_101.png">
</div>

<br />
