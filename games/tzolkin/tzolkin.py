from games import AbstractGame, AbstractConfig
import numpy as np

class Game(AbstractGame):
    def __init__(self, seed=None):
        self.env = Tzolkin(seed)
    pass

    def step(self, action):
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        return self.env.to_play()

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        return

    def expert_agent(self):
        return self.env.expert_action()

    def action_to_string(self, action_number):
        return


class Tzolkin:
    def __init__(self, **kwargs):
        self.players = kwargs.get('players', 4)
        self.player = 0
        self.board = self.make_board()

    def to_play(self):
        return self.player + 1 if self.player < self.players else 0

    def reset(self):
        self.board = self.make_board()
        return self.get_observation()

    def step(self, action):
        pass

    def legal_actions(self):
        pass

    def render(self):
        pass

    def expert_action(self):
        pass

    def make_board(self):
        return np.zeros(1,1)

    def get_observation(self):
        pass