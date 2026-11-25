import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from typing import *
from board import Board
from model import QNetwork, get_state
from rewards import reward_simple, reward_medium, reward_advanced
from collections import deque
from train_common import *

def train(s1=128, s2=64, NUM_EPISODES=1024, reward_fn=reward_medium,
          checkpoint_prefix="model", eval_every=50, save_every=50,
          eval_games=20, eval_max_steps=3000, eval_seed=12345,
          resume_from=None):

    metrics = []
    MEMORY_LIMIT = 50000

    model = QNetwork(s1, s2)
    target = QNetwork(s1, s2)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    eps = 1.0
    # 0.995, 0.1 - rewards were quite random, even for advanced function, max was in [64, 128, 256 (rare)]
    # p.s. actually there was a bug, _move returned last merged_score not merge_sum
    eps_decay = 0.99 # 0.995
    eps_min = 0.02
    gamma = 0.985

    batch_size = 64
    memory = deque(maxlen=MEMORY_LIMIT+10)

    tstate = lambda board: torch.tensor(get_state(board), dtype=torch.float32).unsqueeze(0)

    global_step = 0
    eval_history = []
    best_metric = None
    start_episode = 0

    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, target, optimizer)
        start_episode = int(checkpoint.get("episode", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        eps = float(checkpoint.get("eps", eps))
        metrics = checkpoint.get("metrics", metrics)
        eval_history = checkpoint.get("eval_history", eval_history)
        best_metric = checkpoint.get("best_metric", best_metric)
        if best_metric is not None:
            best_metric = tuple(best_metric)

    last_episode = start_episode - 1
    last_eval_episode = None


    def save_latest(episode, tag="checkpoint"):
        if checkpoint_prefix:
            save_checkpoint(
                f"{checkpoint_prefix}_{tag}.pt",
                model, target, optimizer,
                episode, global_step, eps, s1, s2,
                metrics, eval_history, best_metric)

    def run_eval_and_save(episode):
        nonlocal best_metric, last_eval_episode
        last_eval_episode = episode
        evaluation = evaluate_model(get_state,
            model,
            games=eval_games,
            max_steps=eval_max_steps,
            seed=eval_seed,
        )
        eval_history.append({
            "episode": episode,
            "global_step": global_step,
            **evaluation,
        })
        metric = (
            evaluation["win_rate"],
            evaluation["avg_max_tile"],
            evaluation["avg_score"],
        )
        if best_metric is None or metric > best_metric:
            best_metric = metric
            save_latest(episode, tag="best")

    for episode in range(start_episode, NUM_EPISODES):
        last_episode = episode
        board = Board(4)
        done = False
        state = tstate(board)
        total_reward = 0
        step = 0

        while not done:
            step += 1
            if step > 3000:
                done = True # way too long
                break
            if done:
                break
            # if done and board.max != 2048:
            #     total_reward -= 1000 # lost
            #     memory.append((state.detach(), None, -1000, state.detach(), True, torch.tensor([False] * 4)))
            #     if len(memory) > MEMORY_LIMIT:
            #         memory.popleft()
            #     break

            mask = torch.tensor(board.legal_step_mask())
            if np.random.rand() < eps:
                pos = mask.nonzero().flatten().tolist()
                if not pos:
                    raise ValueError('zero step possibilities in a non-empty loop')
                act = 1 + np.random.choice(pos)
            else:
                with torch.no_grad():
                    qv = model(state).squeeze(0)
                qv[~mask] = float('-inf')
                act = 1 + qv.argmax().item() 

            prev_board_raw = [row[:] for row in board.board]
            moved, merged_value, _ = board.apply(act, generate=False)
            after_board_raw = [row[:] for row in board.board]

            if moved:
                board.gen()
            done = not board.running
            next_state = tstate(board)
            next_mask = torch.tensor(board.legal_step_mask()).unsqueeze(0)                

            #reward = next_state.sum() - state.sum()
            reward = reward_fn(prev_board_raw, after_board_raw, np.log2(1 + merged_value / 100), moved)

            if board.status() == -1:
                reward -= 1 #1000
            elif board.status() == 1:
                reward += 1 #20
                done = True
            total_reward += reward

            # using detach to unnessesery clear graph bindings from memory
            memory.append((state.detach(), act, reward, next_state.detach(), done, next_mask))
            if len(memory) > MEMORY_LIMIT:
                memory.popleft()
            state = next_state
            global_step += 1

            assert torch.equal(state, tstate(board))

            if step % 1000 == 0:
                print(f"    episode #{episode}: step #{step}, running: {board.running}")


            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)

                states, acts, rewards, next_states, dones, next_masks = zip(*batch)
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                next_masks = torch.cat(next_masks)

                acts    = torch.tensor(acts).long().unsqueeze(1) - 1
                rewards = torch.tensor(rewards).float().unsqueeze(1)
                dones   = torch.tensor(dones).float().unsqueeze(1)

                # q_values = model(state).gather(1, acts)
                q_values = model(states)
                q_values = q_values.gather(1, acts)
    
                with torch.no_grad():
                    # next_q = target(next_states).max(1, keepdim=True)[0]
                    next_q = target(next_states)                         # (32, 4)
                    next_q = next_q.masked_fill(~next_masks, -1e9)       # illigal steps -> very small number
                    next_q = next_q.max(1, keepdim=True)[0]              # (32, 1)
                    target_vals = rewards + gamma * next_q * (1 - dones)

                # loss = nn.MSELoss()(q_values, target_vals)
                loss = nn.SmoothL1Loss()(q_values, target_vals)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if eps > eps_min:
            eps *= eps_decay
        if episode % 10 == 0:
            target.load_state_dict(model.state_dict())

        log_episode(metrics, episode, board, total_reward, step)

        if eval_every and (episode + 1) % eval_every == 0:
            run_eval_and_save(episode)
        if save_every and (episode + 1) % save_every == 0:
            save_latest(last_episode)

    target.load_state_dict(model.state_dict()) # sync

    if eval_every and last_episode >= start_episode and last_eval_episode != last_episode:
        run_eval_and_save(last_episode)
    save_latest(last_episode)
    return target, metrics


# def make_tiny():
#     tiny = train(16, 8, 64)
#     save(tiny, 'tiny')
