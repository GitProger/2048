import torch
from model import get_state

class Agent:
    def __init__(self, model):
        self.model = model.eval()

    def act(self, board):
        state = torch.tensor(get_state(board), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.model(state)
            act = 1 + q.argmax().item()
        return act

