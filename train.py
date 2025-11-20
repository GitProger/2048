import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from board import Board
from model import QNetwork, get_state
from rewards import reward_simple, reward_medium, reward_advanced

def log_episode(metrics, episode, board, total_reward, steps):
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


def train(s1=128, s2=64, NUM_EPISODES=1024, reward_fn=reward_medium):
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
    gamma = 0.95

    batch_size = 32
    memory = []

    tstate = lambda board: torch.tensor(get_state(board), dtype=torch.float32).unsqueeze(0)

    for episode in range(NUM_EPISODES):
        board = Board(4)
        done = False
        state = tstate(board)
        total_reward = 0
        step = 0

        while not done:
            step += 1
            if step > 5000:
                done = True # way too long

            rnd_choose = False
            if np.random.rand() < eps:
                act = 1 + np.random.randint(4)
                rnd_choose = True
            else:
                with torch.no_grad():
                    qv = model(state)
                    act = 1 + qv.argmax().item() 

            prev_board_raw = [row[:] for row in board.board]
            moved, merged_value, _ = board.apply(act)
            if not moved and rnd_choose:
            	continue

            next_state = tstate(board)
            next_board_raw = [row[:] for row in board.board]
            
            #reward = next_state.sum() - state.sum()
            reward = reward_fn(prev_board_raw, next_board_raw, merged_value, moved)
            done = not board.running
            total_reward += reward

            # using detach to unnessesery clear graph bindings from memory
            memory.append((state.detach(), act, reward, next_state.detach(), done))
            if len(memory) > MEMORY_LIMIT:
                memory.pop(0)
            state = next_state

            if step % 1000 == 0:
                print(f"    episode #{episode}: step #{step}, running: {board.running}")


            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)

                states, acts, rewards, next_states, dones = zip(*batch)
                states = torch.cat(states)
                next_states = torch.cat(next_states)

                acts    = torch.tensor(acts).long().unsqueeze(1) - 1
                rewards = torch.tensor(rewards).float().unsqueeze(1)
                dones   = torch.tensor(dones).float().unsqueeze(1)

                # q_values = model(state).gather(1, acts)
                q_values = model(states)
                q_values = q_values.gather(1, acts)
    
                with torch.no_grad():
                    next_q = target(next_states).max(1, keepdim=True)[0]
                    target_vals = rewards + gamma * next_q * (1 - dones)

                loss = nn.MSELoss()(q_values, target_vals)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if eps > eps_min:
            eps *= eps_decay
        if episode % 10 == 0:
            target.load_state_dict(model.state_dict())

        log_episode(metrics, episode, board, total_reward, step)
    return target, metrics


# def make_tiny():
#     tiny = train(16, 8, 64)
#     save(tiny, 'tiny')
