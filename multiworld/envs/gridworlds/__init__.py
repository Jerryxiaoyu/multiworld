from gym.envs.registration import register
register(
    id='GoalGridworld-v0',
    entry_point='multiworld.envs.gridworlds.goal_gridworld:GoalGridworld',
    max_episode_steps=50,
)
register(
    id='GoalGridworld-Concatenated-v0',
    entry_point='multiworld.envs.gridworlds.goal_gridworld:GoalGridworld',
     kwargs={'concatenated':True}
)