from gymnasium.envs.registration import register
from .multi_goal import MultiGoal

register(
    id='MultiGoal-v0',
    entry_point='toy_envs:MultiGoal',
)