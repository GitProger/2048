import torch
import torch.nn as nn
from model import get_state
import board

class Agent:
    def __init__(self, model: nn.Module):
        self.model = model.eval()

    def act(self, board: board.Board):
        state = torch.tensor(get_state(board), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.model(state).squeeze(0)
        mask = torch.tensor(board.legal_step_mask())
        q[~mask] = float('-inf')
        act = 1 + q.argmax().item()
        return act

