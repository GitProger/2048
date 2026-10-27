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

def load_checkpoint(path, model, target, optimizer):
    pass
        
def save_checkpoint(path, model, target, optimizer, 
                    episode, global_step, 
                    eps, s1, s2, metrics, 
                    eval_history, bast_mertic):
    pass

def evaluate_model(model, games=20, max_steps=3000, seed=12345):
    pass


def train(s1=128, s2=64, NUM_EPISODES=1024, reward_fn=reward_medium, 
          checkpoint_prefix=prefix, eval_every=50, save_every=50):
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
    eps_min = 0.1
    gamma = 0.985

    batch_size = 32
    memory = deque(maxlen=MEMORY_LIMIT+10)

    tstate = lambda board: torch.tensor(get_state(board), dtype=torch.float32).unsqueeze(0)

    for episode in range(NUM_EPISODES):
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

    target.load_state_dict(model.state_dict()) # sync
    return target, metrics


# def make_tiny():
#     tiny = train(16, 8, 64)
#     save(tiny, 'tiny')
