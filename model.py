import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from board import Board

N_RANKS = 16 # 12

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
    out[mask] = np.log2(b[mask]) / 11 # norm log2(2048) # / 16
    # exps = np.zeros(16, dtype=int)
    # mask = b > 0
    # exps[mask] = np.log2(b[mask]).astype(int)
    # out = np.zeros((16, 16), dtype=np.float32)
    # out[np.arange(16), exps] = 1.0
    # ## for cell in range(16): out[cell][exps[cell]] = 1.0

    # n_levels = 16
    # ranks = np.zeros(b.shape, dtype=np.int64)
    # mask = b > 0
    # ranks[mask] = np.log2(b[mask]).astype(np.int64)  # 0 = пусто, 1 = 2, 2 = 4, ...
    # out = np.zeros((n_levels, 4, 4), dtype=np.float32)
    # for r in range(n_levels): out[r] = (ranks == r)
    # return out.reshape(-1) 
    return out.flatten()

def get_emb_state(b: Board):
    b = np.array(b.board)
    mask = b > 0
    ranks = np.zeros(b.shape, dtype=np.int64)
    ranks[mask] = np.log2(b[mask]).astype(np.int64)
    return ranks.flatten()
    # return F.one_hot(torch.tensor(ranks), 16)


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


class QNetworkEmbed(nn.Module):
    def __init__(self, d=16, h=256):
        super().__init__()
        self.emb = nn.Embedding(N_RANKS, d)
        self.mlp = nn.Sequential(
            nn.Linear(16 * d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 4),
        )

    def forward(self, x: torch.Tensor):
        x = self.emb(x).flatten(1)
        return self.mlp(x)


class QNetworkLookForward(nn.Module):
    pass

class ConvNetwork(nn.Module):
    def __init__(self, ds=[64, 128], h=128, kernels=[(1,2),(2,1)]):
        layers = []
        dims = [1] + ds
        for i, k in enumerate(kernels):
            layers.append(nn.Conv2d(in_channels=dims[i], out_channels=dims[i + 1], kernel_size=k))
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            t: torch.Tensor = self.net(torch.zeros(1, 1, 4, 4))
            n = t.shape[1]
        self.head = nn.Sequential(nn.Linear(n, h), nn.ReLU(), nn.Linear(h, 4))

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1, 4, 4)
        return self.head(self.net(x))

class TrainParameters:
    pass

def save_basic(model: nn.Module, file_prefix: str):
    torch.save(model.state_dict(), file_prefix + '_trained.zip')

def load_basic(model: nn.Module, file_prefix: str):
    state = torch.load(file_prefix + '_trained.zip')
    model.load_state_dict(state)

def save_model(model: nn.Module, file_name: str, s1, s2):
    torch.save({
        "s1": s1,
        "s2": s2,
        "state_dict": model.state_dict()
    }, file_name + '.pt')

def load_model(file_name: str) -> nn.Module:
    data = torch.load(file_name + '.pt')
    model = QNetwork(data["s1"], data["s2"])
    model.load_state_dict(data["state_dict"])
    return model
