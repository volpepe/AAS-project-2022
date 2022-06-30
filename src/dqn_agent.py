from action import Action
from agent import Agent
from state import State

class DQN_Agent(Agent):
    def __init__(self) -> None:
        # TODO
        super().__init__()

    def choose_action(self, state: State) -> Action:
        # TODO
        return super().choose_action(state)

    def train_step(self, train_sequence) -> None:
        # TODO
        return super().train_step(train_sequence)
