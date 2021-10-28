import gym

from .soccer_1v1 import SoccerEnv as Soccer

env_registry = dict()
env_registry["soccer_1v1"] = Soccer
env_registry["cart_pole"] = gym.make("CartPole-v0")
