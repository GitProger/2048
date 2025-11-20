import pygame
import random
from typing import Optional, Tuple
from abc import abstractmethod

# is always square for simplicity
class Board:
    def __init__(self, sz):
        self.sz = sz
        self.board = [[0] * sz for _ in range(sz)]
        self.running = True
        self.max = 0
        self.gen()
        self.ok = lambda x: 0 <= x < self.sz

    def _next(self) -> int:
        r = 2 if random.random() < 0.9 else 4
        self.max = max(self.max, r)
        return r

    def _rndpos(self) -> Optional[Tuple[int, int]]:
        free = [(i, j) for i in range(self.sz) for j in range(self.sz) if self.board[i][j] == 0]
        return random.choice(free) if free else None

    def check(self):
        b, sz = self.board, self.sz
        for i in range(sz):
            for j in range(sz):
                if b[i][j] == 2048:
                    return False
                if b[i][j] == 0 or (i + 1 < sz and b[i][j] == b[i + 1][j]) or (j + 1 < sz and b[i][j] == b[i][j + 1]):
                    return True
        return False

    def gen(self) -> Optional[Tuple[int, int]]:
        pos = self._rndpos()
        if pos:
            i, j = pos
            self.board[i][j] = self._next()
        self.running = self.check()
        return pos

    def _merge(self, x, y, dx, dy) -> bool:
        cx, cy = x + dx, y + dy
        merged = 0
        merged_score = 0
        moved = False
        while self.ok(cx) and self.ok(cy):
            if self.board[cx][cy] == 0:
                if self.board[x][y] > 0:
                    self.board[cx][cy] = self.board[x][y]
                    self.board[x][y] = 0
                    moved = True
            elif merged < 1 and self.board[cx][cy] == self.board[x][y]:
                self.board[cx][cy] *= 2
                merged_value = self.board[cx][cy]
                merged_score += merged_value
                self.max = max(self.max, merged_value)
                self.board[x][y] = 0
                merged = 1
                moved = True
            x, y = cx, cy
            cx, cy = x + dx, y + dy
        return moved, merged_score

    def _move(self, dirs) -> Optional[Tuple[int, int]]:
        if not self.running:
            return False, 0, None
        dx, dy = dirs
        ns, ne, nd = (0, self.sz, 1) if dx != 1 else (self.sz - 1, -1, -1)
        ms, me, md = (0, self.sz, 1) if dy != 1 else (self.sz - 1, -1, -1)
        moved = False
        merge_sum = 0
        for i in range(ns, ne, nd):
            for j in range(ms, me, md):
                was, merged_score = self._merge(i, j, dx, dy)
                moved |= was
                merge_sum += merged_score
        new = self.gen() if moved else None
        return moved, merge_sum, new

    #row, col
    def left(self):  return self._move((0, -1))
    def up(self):    return self._move((-1, 0))
    def right(self): return self._move((0, 1))
    def down(self):  return self._move((1, 0))
    def apply(self, d) -> Optional[Tuple[int, int]]:
        if d == 0:
            self.running = False
            return False, 0, None
        dirs = {1: self.left, 2: self.up, 3: self.right, 4: self.down}
        return dirs[d]()

    def __getitem__(self, i):
        return self.board[i]

class TerminalGame:
    def __init__(self, sz):
        self.board = Board(sz)
    def iteration(self):
        step = self.wait()
        self.board.apply(step)
        self.render()
    def render(self):
        os.system('clear')
        for i in range(self.board.sz):
            for j in range(self.board.sz):
                print("%5d" % self.board[i][j], end=" ")
            print()
    def wait(self):
        self.board.apply(self.supply())
    @abstractmethod
    def supply(self):
        pass

class UserInterface(TerminalGame):
    def __init__(self, model, sz):
        super().__init__(sz)
        
    def supply(self):
        tr = {'a': 1, 'w': 2, 'd': 3, 's': 4}
        x = input()[0]
        return tr[x]

class NeuralInterface(TerminalGame):
    def supply(self):
        pass

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
