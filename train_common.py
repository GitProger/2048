import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from pathlib import Path
from typing import *
from board import Board


def log_episode(metrics: List, episode: int, board: Board, total_reward: int, steps: int):
    max_tile = board.max
    empty = sum(1 for i in range(board.sz) for j in range(board.sz) if board.board[i][j] == 0)
    print(f"=== episode #{episode}")
    print(f"    max tile:      {max_tile}")
    print(f"    empty cells:   {empty}")
    print(f"    total reward:  {total_reward}")
    print(f"    total steps:   {steps}")
    print(f"    board:")
    for i in range(board.sz):
        print(' ' * 6, ' '.join(["%5d" % board[i][j] for j in range(board.sz)]))
    metrics.append((total_reward, max_tile, steps))

def load_checkpoint(path: str, model: nn.Module, target: nn.Module, optimizer: optim.Optimizer):
    checkpoint: dict = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    target.load_state_dict(checkpoint["target_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if checkpoint.get("random_rng") is not None:
        random.setstate(checkpoint["random_rng"])
    if checkpoint.get("numpy_rng") is not None:
        np.random.set_state(checkpoint["numpy_rng"])
    if checkpoint.get("torch_rng") is not None:
        torch.set_rng_state(checkpoint["torch_rng"])
    return checkpoint

def save_checkpoint(path: str, model: nn.Module, target: nn.Module, optimizer: optim.Optimizer,
                    episode, global_step,
                    eps, s1, s2, metrics,
                    eval_history, best_metric):
    path: Path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "target_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "eps": eps,
        "s1": s1,
        "s2": s2,
        "metrics": list(metrics),
        "eval_history": list(eval_history),
        "best_metric": best_metric,
        "random_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)

def evaluate_model(state_load, model, games=20, max_steps=3000, seed=12345):
    games = max(1, int(games))
    device = next(model.parameters()).device
    random_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    was_training = model.training

    scores = []
    max_tiles = []
    wins = []
    lengths = []

    model.eval()
    try:
        with torch.no_grad():
            for game in range(games):
                random.seed(seed + game)
                np.random.seed(seed + game)
                board = Board(4)
                score = 0
                step = 0

                while board.running and step < max_steps:
                    state = torch.tensor(state_load(board), dtype=torch.float32, device=device).unsqueeze(0)
                    mask = torch.tensor(board.legal_step_mask(), dtype=torch.bool, device=device)
                    if not mask.any().item():
                        break

                    q = model(state).squeeze(0)
                    q[~mask] = float("-inf")
                    act = 1 + q.argmax().item()

                    moved, merged, _ = board.apply(act)
                    score += merged
                    step += 1
                    if not moved:
                        break

                scores.append(float(score))
                max_tiles.append(int(board.max))
                wins.append(bool(board.has_won()))
                lengths.append(int(step))
    finally:
        random.setstate(random_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        model.train(was_training)

    result = {
        "games": games,
        "avg_score": float(np.mean(scores)),
        "avg_max_tile": float(np.mean(max_tiles)),
        "best_max_tile": int(max(max_tiles)),
        "win_rate": float(np.mean(wins)),
        "avg_steps": float(np.mean(lengths)),
        "scores": scores,
        "max_tiles": max_tiles,
        "wins": wins,
        "steps": lengths,
    }
    print(
        f"[eval] games={games} avg_score={result['avg_score']:.1f} "
        f"avg_max_tile={result['avg_max_tile']:.1f} "
        f"best_max_tile={result['best_max_tile']} "
        f"win_rate={result['win_rate']:.1%} avg_steps={result['avg_steps']:.1f}"
    )
    return result

