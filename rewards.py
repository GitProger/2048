import numpy as np

NOT_MOVING_FINE = 100

def reward_simple(prev_board, next_board, merged_value, moved):
    reward = merged_value
    if moved: 
        reward += 0.1
    else:
        reward -= NOT_MOVING_FINE
    return reward

def reward_medium(prev_board, next_board, merged_value, moved):
    empty_prev = sum(1 for row in prev_board for v in row if v == 0)
    empty_next = sum(1 for row in next_board for v in row if v == 0)
    empty_bonus = empty_next - empty_prev
    reward = merged_value + 0.1 * empty_bonus - 0.01
    if not moved:
        reward -= NOT_MOVING_FINE
    return reward

# from https://arxiv.org/abs/1604.05085
def reward_advanced(prev_board, next_board, merged_value, moved):
    def monotonicity(b):
        score = 0
        for row in b:
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]: score += 1
        for col in zip(*b):
            for i in range(len(col) - 1):
                if col[i] >= col[i + 1]: score += 1
        return score

    def smoothness(b):
        score = 0
        for i in range(4):
            for j in range(4):
                if i + 1 < 4 and b[i][j] and b[i + 1][j]:
                    score -= abs(np.log2(b[i][j]) - np.log2(b[i + 1][j]))
                if j + 1 < 4 and b[i][j] and b[i][j + 1]:
                    score -= abs(np.log2(b[i][j]) - np.log2(b[i][j + 1]))
        return score

    empty_prev = sum(1 for row in prev_board for v in row if v == 0)
    empty_next = sum(1 for row in next_board for v in row if v == 0)
    empty_bonus = empty_next - empty_prev

    basic_reward = merged_value + 0.1 * empty_bonus
    if not moved:
    	basic_reward -= NOT_MOVING_FINE
    return basic_reward + 0.1 * monotonicity(next_board) + 0.1 * smoothness(next_board)

