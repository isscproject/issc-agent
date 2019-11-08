"""Implementation of a simple deterministic agent using Docker."""

import pommerman
from issc_agent import *
from pommerman.runner import DockerAgentRunner

class ISSC_Agent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        super(ISSC_Agent, self).__init__()
        self._agent = ISSCAgent("model.pth")

    def init_agent(self, id, game_type):
        self._agent.init_agent(id, game_type)

    def episode_end(self, reward):
        self._agent.episode_end(reward)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)


def main():
    agent = ISSC_Agent()
    agent.run()


if __name__ == "__main__":
    main()
