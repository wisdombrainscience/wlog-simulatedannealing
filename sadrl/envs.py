"""DynamicMaze: grid world with moving obstacles (extends the maze of the original draft).

Observation (18-dim, matches the 18-input network of the original manuscript):
  [0:2]   agent position (row, col) normalised to [0,1]
  [2:4]   goal displacement (drow, dcol) normalised to [-1,1]
  [4:8]   wall flags in the 4 neighbouring cells (up, down, left, right)
  [8:12]  obstacle-occupancy flags of the 4 neighbouring cells *after* obstacles move
  [12:18] displacement of the 3 obstacles relative to the agent, normalised
Actions: 0 up, 1 down, 2 left, 3 right.
Reward: +1 on reaching the goal, -1 on collision with an obstacle (terminal),
        -0.002 per step otherwise (a time-out therefore costs -0.3 > -1); episodes are truncated after `max_steps` steps.
"""
import numpy as np

LAYOUT = [
    "############",
    "#S...#.....#",
    "#.##.#.###.#",
    "#.#..#...#.#",
    "#.#.####.#.#",
    "#.#......#.#",
    "#.######.#.#",
    "#......#.#.#",
    "#.####.#.#.#",
    "#....#.....#",
    "####.#.###G#",
    "############",
]
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


class DynamicMaze:
    n_actions = 4
    obs_dim = 18

    def __init__(self, max_steps=150, seed=0, obstacle_cols=(4, 6, 8)):
        g = np.array([list(r) for r in LAYOUT])
        self.walls = g == "#"
        self.H, self.W = g.shape
        self.start = tuple(np.argwhere(g == "S")[0])
        self.goal = tuple(np.argwhere(g == "G")[0])
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.obstacle_cols = obstacle_cols
        # each obstacle patrols the free cells of its column, moving up and wrapping around
        self.tracks = []
        for c in obstacle_cols:
            rows = [r for r in range(self.H) if not self.walls[r, c]]
            self.tracks.append([(r, c) for r in rows])

    def seed(self, s):
        self.rng = np.random.default_rng(s)

    def _obs_cells(self):
        return [tr[(i) % len(tr)] for tr, i in zip(self.tracks, self.idx)]

    def _obs(self):
        r, c = self.pos
        o = np.zeros(18, dtype=np.float32)
        o[0], o[1] = r / (self.H - 1), c / (self.W - 1)
        o[2], o[3] = (self.goal[0] - r) / self.H, (self.goal[1] - c) / self.W
        nxt = [tr[(i - 1) % len(tr)] for tr, i in zip(self.tracks, self.idx)]  # obstacles move "up" = index-1
        for k, (dr, dc) in enumerate(MOVES):
            rr, cc = r + dr, c + dc
            o[4 + k] = float(self.walls[rr, cc])
            o[8 + k] = float((rr, cc) in nxt)
        for j, (orr, occ) in enumerate(self._obs_cells()):
            o[12 + 2 * j] = (orr - r) / self.H
            o[13 + 2 * j] = (occ - c) / self.W
        return o

    def reset(self):
        self.pos = self.start
        self.t = 0
        self.idx = [int(self.rng.integers(len(tr))) for tr in self.tracks]
        return self._obs()

    def step(self, a):
        self.t += 1
        r, c = self.pos
        dr, dc = MOVES[a]
        if not self.walls[r + dr, c + dc]:
            self.pos = (r + dr, c + dc)
        self.idx = [(i - 1) % len(tr) for tr, i in zip(self.tracks, self.idx)]
        if self.pos == self.goal:
            return self._obs(), 1.0, True, False
        if self.pos in self._obs_cells():
            return self._obs(), -1.0, True, False
        trunc = self.t >= self.max_steps
        return self._obs(), -0.002, False, trunc


class GymWrap:
    """Thin wrapper giving gymnasium envs the same (obs, r, term, trunc) API."""

    def __init__(self, name, seed=0):
        import gymnasium as gym
        self.env = gym.make(name)
        self.n_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self._seed = seed
        self._first = True

    def seed(self, s):
        self._seed = s
        self._first = True

    def reset(self):
        if self._first:
            o, _ = self.env.reset(seed=self._seed)
            self._first = False
        else:
            o, _ = self.env.reset()
        return o.astype(np.float32)

    def step(self, a):
        o, r, term, trunc, _ = self.env.step(int(a))
        return o.astype(np.float32), float(r), term, trunc


def make_env(name, seed=0):
    if name == "DynamicMaze":
        e = DynamicMaze(seed=seed)
        return e
    return GymWrap(name, seed=seed)
