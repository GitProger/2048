import os

import pygame
import random
from typing import Optional, Tuple, List

# is always square for simplicity
class Board:
    DIRS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}

    def __init__(self, sz):
        self.sz = sz
        self.board = [[0] * sz for _ in range(sz)]
        self.running = True
        self.max = 0
        self.ok = lambda x, y: 0 <= x < self.sz and 0 <= y < self.sz
        self.gen()
        self.gen()

    def _next(self) -> int:
        r = 2 if random.random() < 0.9 else 4
        self.max = max(self.max, r)
        return r

    def _rndpos(self) -> Optional[Tuple[int, int]]:
        free = [(i, j) for i in range(self.sz) for j in range(self.sz) if self.board[i][j] == 0]
        return random.choice(free) if free else None

    def can_move(self, d):
        dx, dy = self.DIRS[d]
        b, sz = self.board, self.sz
        for i in range(sz):
            for j in range(sz):
                if b[i][j] == 0:
                    continue
                i2, j2 = i + dx, j + dy 
                if self.ok(i2, j2) and b[i2][j2] in (0, b[i][j]):
                    return True
        return False

    def legal_step_mask(self):
        return [self.can_move(d) for d in range(1, 5)]

    def has_moves(self):
        b, sz = self.board, self.sz
        for i in range(sz):
            for j in range(sz):
                if b[i][j] == 0 or (i + 1 < sz and b[i][j] == b[i + 1][j]) or (j + 1 < sz and b[i][j] == b[i][j + 1]):
                    return True
        return False

    def check_running(self):
        return self.has_moves() and not self.has_won()

    def gen(self) -> Optional[Tuple[int, int]]:
        pos = self._rndpos()
        if pos:
            i, j = pos
            self.board[i][j] = self._next()
        self.running = self.check_running()
        return pos

    def collapse(self, a: List[int], dir: int) -> Tuple[bool, List[int], int]:
        if dir == 0 or not a:
            return False, a, 0
        if dir not in (-1, 1):
                raise ValueError("dir must be -1, 0 or 1")

        step = -dir
        moved = False
        score = 0
        n = len(a)

        i = 0 if step == 1 else n - 1
        l = i + step
        end = n if step == 1 else -1

        while i != end:
            l = max(l, i + 1) if step == 1 else min(l, i - 1)
            while l != end and a[l] == 0:
                l += step

            if l == end:
                break
            if a[i] == 0: # empty
                moved = True
                a[i] = a[l]
                a[l] = 0
                continue
            elif a[i] == a[l]: # merge
                moved = True
                a[i] *= 2
                score += a[i]
                self.max = max(self.max, a[i])
                a[l] = 0
                i += step
                l += step
            else:
                i += step
        return moved, a, score

    def _move(self, dirs, generate=True) -> Tuple[bool, int, Optional[Tuple[int, int]]]:
        if not self.running:
            return False, 0, None
        dx, dy = dirs
        moved = False
        merge_sum = 0
        if dx == 0:
            for i in range(self.sz):
                mv, self.board[i], sc = self.collapse(self.board[i], dy)
                moved |= mv
                merge_sum += sc
        elif dy == 0:
            for i in range(self.sz):
                mv, a, sc = self.collapse([self.board[j][i] for j in range(self.sz)], dx)
                for j in range(self.sz):
                    self.board[j][i] = a[j]
                moved |= mv
                merge_sum += sc

        new = None
        if generate and moved:
            new = self.gen()
        return moved, merge_sum, new


    def left(self, g=True):  return self._move(self.DIRS[1], g)
    def up(self, g=True):    return self._move(self.DIRS[2], g)
    def right(self, g=True): return self._move(self.DIRS[3], g)
    def down(self, g=True):  return self._move(self.DIRS[4], g)
    def apply(self, d, generate=True) -> Optional[Tuple[int, int]]:
        dirs = {1: self.left, 2: self.up, 3: self.right, 4: self.down}
        return dirs[d](generate)
    def stop(self):
        self.running = False
        return False, 0, None

    def has_won(self):
        return self.max >= 2048
    
    def status(self):
        if self.running:
            return 0
        return 1 if self.has_won() else -1

    def __getitem__(self, i):
        return self.board[i]

class Pygame:
    AGENT_DELAY = 500
    COLORS = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46)
    }
    KEY_MAP = {
        pygame.K_LEFT:  1,
        pygame.K_UP:    2,
        pygame.K_RIGHT: 3,
        pygame.K_DOWN:  4
    }

    def __init__(self, board: Board, cell_size=100):
        self.board = board
        self.cell_size = cell_size
        self.margin = 10
        self.size = board.sz * cell_size + (board.sz + 1) * self.margin
        pygame.init()
        self.screen = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption("2048")
        self.font = pygame.font.SysFont("arial", 40, bold=True)

    def draw(self):
        self.screen.fill((187, 173, 160))
        for i in range(self.board.sz):
            for j in range(self.board.sz):
                val = self.board.board[i][j]
                rect = pygame.Rect(
                    self.margin + j * (self.cell_size + self.margin),
                    self.margin + i * (self.cell_size + self.margin),
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.COLORS.get(val, (60, 58, 50)), rect)
                if val != 0:
                    text = self.font.render(str(val), True, (0, 0, 0) if val <= 4 else (255, 255, 255))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
        pygame.display.flip()

    def run(self, agent=None):
        agent_timer = 0

        clock = pygame.time.Clock()
        running = True
        while running:
            now = pygame.time.get_ticks()
            self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if agent is None and event.type == pygame.KEYDOWN:
                    if event.key in self.KEY_MAP:
                        # print(self.board.running)
                        moved, merged, new = self.board.apply(self.KEY_MAP[event.key])
                        if new:
                            i, j = new
                            self.animate_spawn(i, j, self.board.board[i][j])

            if agent is not None and self.board.running and now - agent_timer >= self.AGENT_DELAY:
                act = agent.act(self.board)
                act_name = ["left", "up", "right", "down"][act - 1]
                moved, merged, new = self.board.apply(act)
                print(f"Agent makes step {act_name}: moved: {moved}, merged value:{merged}")
                agent_timer = now
                if new:
                    i, j = new
                    self.animate_spawn(i, j, self.board.board[i][j])


            clock.tick(30)
        pygame.quit()

    def animate_spawn(self, i, j, value, duration=120):
        x = self.margin + j * (self.cell_size + self.margin)
        y = self.margin + i * (self.cell_size + self.margin)

        start = pygame.time.get_ticks()
        font = pygame.font.SysFont("arial", self.cell_size // 3, bold=True)
        clock = pygame.time.Clock()

        while True:
            t = (pygame.time.get_ticks() - start) / duration
            if t >= 1:
                break
            s = max(0.05, t)
            w = int(self.cell_size * s)
            h = int(self.cell_size * s)
            r = pygame.Rect(x + (self.cell_size - w) // 2,
                            y + (self.cell_size - h) // 2,
                            w, h)

            self.draw()  
            pygame.draw.rect(self.screen, self.COLORS[value], r, border_radius=8)

            txt = font.render(str(value), True, (119,110,101) if value <= 4 else (255,255,255))
            self.screen.blit(txt, txt.get_rect(center=r.center))

            pygame.display.update()
            clock.tick(60)

        self.draw()
        pygame.display.update()


if __name__ == "__main__":
    b = Board(4)
    ui = Pygame(b)
    ui.run()
