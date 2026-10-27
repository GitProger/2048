import numpy as np 
import torch
import torch.nn as nn

# from importlib import reload

# def get_state(b: Board):
#     state = []
#     for i in range(b.sz):
#         for j in range(b.sz):
#             val = b[i][j]
#             state.append(math.log2(val) if val > 0 else 0)
#     return np.array(state, type=np.float32)
def get_state(b):
    b = np.array(b.board)
    out = np.zeros_like(b, dtype=np.float32)
    mask = b > 0
    out[mask] = np.log2(b[mask])
    return out.flatten()

class QNetwork(nn.Module):
    def __init__(self, s1=128, s2=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, s1), nn.ReLU(),
            nn.Linear(s1, s2), nn.ReLU(),
            nn.Linear(s2, 4)
        )

    def forward(self, x):
        return self.model(x)

class TrainParameters:
    pass

def save_basic(model: nn.Module, file_prefix: str):
    torch.save(model.state_dict(), file_prefix + '_trained.zip')

def load_basic(model: nn.Module, file_prefix: str):
    state = torch.load(file_prefix + '_trained.zip')
    model.load_state_dict(state)

def save_model(model: nn.Module, file_prefix: str, s1, s2):
    torch.save({
        "s1": s1,
        "s2": s2,
        "state_dict": model.state_dict()
    }, file_prefix + '_trained.pt')

def load_model(file_prefix: str) -> nn.Module:
    data = torch.load(file_prefix + '_trained.pt')
    model = QNetwork(data["s1"], data["s2"])
    model.load_state_dict(data["state_dict"])
    return model
