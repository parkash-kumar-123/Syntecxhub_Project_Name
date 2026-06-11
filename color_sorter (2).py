"""
╔══════════════════════════════════════════════════════╗
║       COLOR BLOCKS SORTER AGENT                      ║
║       AI Project — Sukkur IBA University             ║
║       Uses: BFS · IDDFS · UCS · A* · Greedy BFS      ║
║                                                      ║
║  HOW TO RUN:                                         ║
║    • Jupyter:  just run this cell                    ║
║    • Terminal: python color_sorter.py                ║
╚══════════════════════════════════════════════════════╝
"""

import copy, random, time, heapq, threading
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, RadioButtons
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────
ROWS, COLS = 5, 3
COLORS = ['R', 'G', 'B', 'Y']
TARGET_ROW = {'R': 0, 'G': 1, 'B': 2, 'Y': 3}
MAX_NODES = 100_000

COLOR_FACE = {
    'R': '#E84040', 'G': '#2ECC71',
    'B': '#4A90D9', 'Y': '#F5C542', None: '#1E2030'
}
COLOR_TEXT = {
    'R': 'white', 'G': 'white',
    'B': 'white', 'Y': '#1A1A1A', None: '#3A3D52'
}
COLOR_LABEL = {
    'R': 'Red', 'G': 'Green', 'B': 'Blue', 'Y': 'Yellow'
}

# ─────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────
class State:
    def __init__(self, grid):
        self.grid = grid

    def __eq__(self, o):
        return self.grid == o.grid

    def __hash__(self):
        return hash(tuple(tuple(r) for r in self.grid))

    def __lt__(self, o):
        return False

    def is_goal(self):
        for r in range(4):
            row = self.grid[r]
            if not all(c == row[0] and c is not None for c in row):
                return False
        return all(self.grid[4][c] is None for c in range(COLS))

    def get_empty_cells(self):
        return [(r, c) for r in range(ROWS)
                for c in range(COLS) if self.grid[r][c] is None]

    def get_neighbors(self):
        nbrs = []
        for (er, ec) in self.get_empty_cells():
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = er+dr, ec+dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr][nc]:
                    g = copy.deepcopy(self.grid)
                    g[er][ec], g[nr][nc] = g[nr][nc], None
                    nbrs.append(State(g))
        return nbrs

    def clone(self):
        return State(copy.deepcopy(self.grid))

    @staticmethod
    def make_goal():
        grid = [[c]*COLS for c in COLORS]
        grid.append([None]*COLS)
        return State(grid)

    @staticmethod
    def generate_random(moves=150):
        state = State.make_goal()
        visited = {hash(state)}
        for _ in range(moves):
            nbrs = state.get_neighbors()
            random.shuffle(nbrs)
            for n in nbrs:
                if hash(n) not in visited:
                    state = n
                    visited.add(hash(n))
                    break
        return state


# ─────────────────────────────────────────────────────
#  HEURISTICS
# ─────────────────────────────────────────────────────
def iddfs(initial):
    """Iterative Deepening Depth-First Search (IDDFS).
    Repeatedly runs depth-limited DFS with increasing depth limits (0, 1, 2, …)
    until the goal is found. Combines the space efficiency of DFS (O(depth))
    with the optimality/completeness of BFS — finds the shallowest solution.
    Uses a visited set per iteration to avoid cycles within each depth limit."""

    def dls(state, path, depth_limit, visited):
        """Depth-Limited Search — returns (path, nodes, found) or (None, nodes, False)."""
        nonlocal total_nodes
        total_nodes += 1
        if total_nodes > 500_000:
            return None, True   # cutoff signal: hit node limit

        if state.is_goal():
            return path, False  # False = not a node-limit cutoff

        if len(path) - 1 >= depth_limit:
            return None, False  # depth limit reached, not a goal

        for nbr in state.get_neighbors():
            hn = hash(nbr)
            if hn not in visited:
                visited.add(hn)
                result, cutoff = dls(nbr, path + [nbr], depth_limit, visited)
                visited.discard(hn)   # backtrack — allow revisit on other branches
                if result is not None:
                    return result, False
                if cutoff:
                    return None, True  # propagate node-limit cutoff upward
        return None, False

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    IDDFS_MAX = 500_000
    total_nodes = 0
    for depth_limit in range(0, IDDFS_MAX):
        visited = {hash(initial)}
        result, cutoff = dls(initial, [initial], depth_limit, visited)
        if result is not None:
            return result, total_nodes, "Solved"
        if cutoff:
            return None, total_nodes, "Node limit reached"
        # No solution at this depth — increase limit and retry
    return None, total_nodes, "No solution found"

def h_row_distance(state):
    return sum(abs(TARGET_ROW[state.grid[r][c]] - r)
               for r in range(ROWS) for c in range(COLS)
               if state.grid[r][c])

def h_row_conflict(state):
    base = h_row_distance(state)
    penalty = sum(2 for r in range(4)
                  if len(set(state.grid[r][c] for c in range(COLS)
                             if state.grid[r][c])) > 1)
    return base + penalty


# ─────────────────────────────────────────────────────
#  ALGORITHMS
# ─────────────────────────────────────────────────────
def bfs(initial):
    queue = deque([(initial, [initial])])
    visited = {hash(initial)}
    nodes = 0
    while queue:
        state, path = queue.popleft()
        nodes += 1
        if nodes > MAX_NODES:
            return None, nodes, "Node limit reached"
        if state.is_goal():
            return path, nodes, "Solved"
        for nbr in state.get_neighbors():
            h = hash(nbr)
            if h not in visited:
                visited.add(h)
                queue.append((nbr, path + [nbr]))
    return None, nodes, "No solution found"


def astar(initial, heuristic):
    cnt = 0
    heap = [(heuristic(initial), 0, cnt, initial, [initial])]
    best_g = {hash(initial): 0}
    nodes = 0
    while heap:
        f, g, _, state, path = heapq.heappop(heap)
        nodes += 1
        if nodes > MAX_NODES:
            return None, nodes, "Node limit reached"
        if state.is_goal():
            return path, nodes, "Solved"
        hs = hash(state)
        if g > best_g.get(hs, float('inf')):
            continue
        for nbr in state.get_neighbors():
            ng = g + 1
            hn = hash(nbr)
            if ng < best_g.get(hn, float('inf')):
                best_g[hn] = ng
                cnt += 1
                heapq.heappush(heap,
                    (ng + heuristic(nbr), ng, cnt, nbr, path + [nbr]))
    return None, nodes, "No solution found"


def gbfs(initial, heuristic):
    cnt = 0
    heap = [(heuristic(initial), cnt, initial, [initial])]
    visited = {hash(initial)}
    nodes = 0
    while heap:
        _, _, state, path = heapq.heappop(heap)
        nodes += 1
        if nodes > MAX_NODES:
            return None, nodes, "Node limit reached"
        if state.is_goal():
            return path, nodes, "Solved"
        for nbr in state.get_neighbors():
            h = hash(nbr)
            if h not in visited:
                visited.add(h)
                cnt += 1
                heapq.heappush(heap, (heuristic(nbr), cnt, nbr, path + [nbr]))
    return None, nodes, "No solution found"


UCS_MAX_NODES = 300_000   # UCS explores far more states than informed search
UCS_TIMEOUT   = 30.0      # seconds — uninformed search can be slow

def ucs(initial):
    """Uniform Cost Search — expands nodes by cumulative path cost g(n).
    Every move costs 1, so UCS finds the minimum-moves solution optimally.
    Since all edge costs are equal, the first time a state is popped it is
    guaranteed to be via the cheapest path — we use a closed set to skip
    re-expansion. Path reconstructed via came_from for memory efficiency.
    NOTE: UCS explores far more nodes than A* because it has no heuristic;
    this is intentional — the comparison demonstrates why heuristics matter."""
    cnt = 0
    heap = [(0, cnt, initial)]
    visited   = set()
    best_cost = {hash(initial): 0}
    came_from = {hash(initial): None}
    state_map = {hash(initial): initial}
    nodes = 0
    t_start = time.time()

    while heap:
        cost, _, state = heapq.heappop(heap)
        nodes += 1

        if nodes > UCS_MAX_NODES:
            return None, nodes, "Node limit (use A* for large puzzles)"
        if time.time() - t_start > UCS_TIMEOUT:
            return None, nodes, f"Timeout after {UCS_TIMEOUT}s"

        hs = hash(state)
        if hs in visited:
            continue
        visited.add(hs)

        if state.is_goal():
            path = []
            cur = hs
            while cur is not None:
                path.append(state_map[cur])
                cur = came_from[cur]
            path.reverse()
            return path, nodes, "Solved"

        for nbr in state.get_neighbors():
            hn = hash(nbr)
            if hn in visited:
                continue
            new_cost = cost + 1
            if new_cost < best_cost.get(hn, float('inf')):
                best_cost[hn] = new_cost
                came_from[hn] = hs
                state_map[hn] = nbr
                cnt += 1
                heapq.heappush(heap, (new_cost, cnt, nbr))

    return None, nodes, "No solution found"


# ─────────────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────────────
class ColorSorterGUI:

    BG       = '#0D0F1A'
    PANEL    = '#13162A'
    ACCENT   = '#4A90D9'
    GREEN    = '#2ECC71'
    YELLOW   = '#F5C542'
    RED      = '#E84040'
    PURPLE   = '#9B59B6'
    TEXT     = '#C8CEDE'
    MUTED    = '#5A6080'
    BORDER   = '#252840'

    ANIM_INTERVAL = 0.38

    def __init__(self):
        self.state     = None
        self.initial   = None
        self.solution  = []
        self.step_idx  = 0
        self.animating = False
        self.solving   = False

        self._build_figure()
        self.new_puzzle(None)
        plt.show()

    # ── FIGURE LAYOUT ────────────────────────────
    def _build_figure(self):
        plt.rcParams.update({
            'figure.facecolor': self.BG,
            'axes.facecolor':   self.BG,
            'text.color':       self.TEXT,
            'font.family':      'monospace',
        })

        self.fig = plt.figure(figsize=(14, 8.5))
        self.fig.patch.set_facecolor(self.BG)

        try:
            self.fig.canvas.manager.set_window_title(
                'Color Blocks Sorter Agent — AI Project')
        except Exception:
            pass

        self.fig.text(0.03, 0.965, 'COLOR BLOCKS SORTER AGENT',
                      fontsize=15, fontweight='bold',
                      color=self.ACCENT, fontfamily='monospace')
        self.fig.text(0.03, 0.942,
                      'Sukkur IBA University  ·  Artificial Intelligence',
                      fontsize=8.5, color=self.MUTED, fontfamily='monospace')

        gs = gridspec.GridSpec(
            4, 4,
            left=0.03, right=0.97,
            top=0.92,  bottom=0.06,
            hspace=0.6, wspace=0.35
        )

        self.ax_grid    = self.fig.add_subplot(gs[:, 0])
        self.ax_metrics = self.fig.add_subplot(gs[0, 1])
        self.ax_radio   = self.fig.add_subplot(gs[1:3, 1])
        self.ax_legend  = self.fig.add_subplot(gs[3, 1])

        self.ax_btn_new   = self.fig.add_subplot(gs[0, 2])
        self.ax_btn_solve = self.fig.add_subplot(gs[0, 3])
        self.ax_btn_back  = self.fig.add_subplot(gs[1, 2])
        self.ax_btn_fwd   = self.fig.add_subplot(gs[1, 3])
        self.ax_btn_anim  = self.fig.add_subplot(gs[2, 2])
        self.ax_btn_stop  = self.fig.add_subplot(gs[2, 3])
        self.ax_btn_reset = self.fig.add_subplot(gs[3, 2])
        self.ax_btn_cmp   = self.fig.add_subplot(gs[3, 3])

        # Step label strip
        self.ax_step = self.fig.add_axes([0.03, 0.01, 0.3, 0.03])
        self.ax_step.set_facecolor(self.BG)
        self.ax_step.axis('off')
        self.step_text = self.ax_step.text(
            0.0, 0.5, '', ha='left', va='center',
            fontsize=9, color=self.MUTED, fontfamily='monospace'
        )

        self._build_buttons()
        self._build_radio()
        self._build_legend()
        self._build_metrics()

    def _style_ax(self, ax):
        ax.set_facecolor(self.PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(self.BORDER)
            spine.set_linewidth(0.8)
        ax.set_xticks([]); ax.set_yticks([])

    # ── WIDGETS ──────────────────────────────────
    def _lighten(self, hex_color):
        r = min(255, int(hex_color[1:3], 16) + 30)
        g = min(255, int(hex_color[3:5], 16) + 30)
        b = min(255, int(hex_color[5:7], 16) + 30)
        return f'#{r:02X}{g:02X}{b:02X}'

    def _make_btn(self, ax, label, color, callback, text_color=None):
        ax.set_facecolor(self.BG)
        ax.axis('off')
        btn = Button(ax, label, color=color, hovercolor=self._lighten(color))
        btn.label.set_fontfamily('monospace')
        btn.label.set_fontsize(9)
        btn.label.set_fontweight('bold')
        btn.label.set_color(text_color or ('#1A1A1A' if color == self.YELLOW else 'white'))
        btn.on_clicked(callback)
        return btn

    def _build_buttons(self):
        self.btn_new   = self._make_btn(self.ax_btn_new,   'NEW PUZZLE',  self.ACCENT,  self.new_puzzle)
        self.btn_solve = self._make_btn(self.ax_btn_solve, 'SOLVE',       self.GREEN,   self.solve)
        self.btn_back  = self._make_btn(self.ax_btn_back,  'STEP BACK',   self.YELLOW,  self.step_back)
        self.btn_fwd   = self._make_btn(self.ax_btn_fwd,   'STEP FWD',    self.YELLOW,  self.step_fwd)
        self.btn_anim  = self._make_btn(self.ax_btn_anim,  'ANIMATE',     self.PURPLE,  self.animate)
        self.btn_stop  = self._make_btn(self.ax_btn_stop,  'STOP ANIM',   self.RED,     self.stop_anim)
        self.btn_reset = self._make_btn(self.ax_btn_reset, 'RESET',       self.BORDER,  self.reset, self.TEXT)
        self.btn_cmp   = self._make_btn(self.ax_btn_cmp,   'COMPARE ALL', '#6C3483',    self.compare_all)

    def _build_radio(self):
        self.ax_radio.set_facecolor(self.PANEL)
        for spine in self.ax_radio.spines.values():
            spine.set_edgecolor(self.BORDER)
        self.ax_radio.set_title('ALGORITHM', color=self.MUTED,
                                 fontsize=8, fontfamily='monospace', pad=4)

        algo_options = (
            'BFS',
            'IDDFS',
            'A* - Row Distance',
            'A* - Row Conflict',
            'Greedy BFS',
            'Uniform Cost Search',
        )
        self.radio = RadioButtons(self.ax_radio, algo_options,
                                  activecolor=self.ACCENT)
        for lbl in self.radio.labels:
            lbl.set_fontfamily('monospace')
            lbl.set_fontsize(8.5)
            lbl.set_color(self.TEXT)
        self.radio.set_active(3)

    def _build_legend(self):
        self.ax_legend.set_facecolor(self.PANEL)
        for spine in self.ax_legend.spines.values():
            spine.set_edgecolor(self.BORDER)
        self.ax_legend.set_title('LEGEND', color=self.MUTED,
                                  fontsize=8, fontfamily='monospace', pad=4)
        self.ax_legend.axis('off')
        items = [
            mpatches.Patch(color=COLOR_FACE['R'], label='Red    -> row 1'),
            mpatches.Patch(color=COLOR_FACE['G'], label='Green  -> row 2'),
            mpatches.Patch(color=COLOR_FACE['B'], label='Blue   -> row 3'),
            mpatches.Patch(color=COLOR_FACE['Y'], label='Yellow -> row 4'),
            mpatches.Patch(color=COLOR_FACE[None], label='Empty  (row 5)'),
        ]
        self.ax_legend.legend(
            handles=items, loc='center',
            fontsize=8, frameon=False,
            labelcolor=self.TEXT,
            prop={'family': 'monospace', 'size': 8}
        )

    def _build_metrics(self):
        self._style_ax(self.ax_metrics)
        self.ax_metrics.set_title('METRICS', color=self.MUTED,
                                   fontsize=8, fontfamily='monospace', pad=4)
        self.metric_text = self.ax_metrics.text(
            0.05, 0.5,
            'Status   : --\nSteps    : --\nNodes    : --\nTime(ms) : --',
            transform=self.ax_metrics.transAxes,
            fontsize=8.5, fontfamily='monospace',
            color=self.TEXT, va='center', linespacing=2.0
        )

    # ── GRID DRAWING ─────────────────────────────
    def draw_grid(self, state, highlight=None):
        highlight = highlight or set()
        ax = self.ax_grid
        ax.cla()
        self._style_ax(ax)
        ax.set_title('PUZZLE GRID', color=self.MUTED,
                     fontsize=8, fontfamily='monospace', pad=6)
        ax.set_xlim(-0.15, COLS + 0.05)
        ax.set_ylim(-0.1, ROWS + 0.15)
        ax.invert_yaxis()

        for r in range(ROWS):
            for c in range(COLS):
                cell = state.grid[r][c]
                face = COLOR_FACE[cell]
                hl   = (r, c) in highlight

                rect = FancyBboxPatch(
                    (c + 0.05, r + 0.05), 0.90, 0.90,
                    boxstyle='round,pad=0.04',
                    facecolor=face,
                    edgecolor='white' if hl else self.BORDER,
                    linewidth=2.5 if hl else 0.8,
                    zorder=2
                )
                ax.add_patch(rect)

                if cell:
                    ax.text(c + 0.5, r + 0.42, cell,
                            ha='center', va='center',
                            fontsize=21, fontweight='bold',
                            fontfamily='monospace',
                            color=COLOR_TEXT[cell], zorder=3)
                    ax.text(c + 0.5, r + 0.75, COLOR_LABEL[cell],
                            ha='center', va='center',
                            fontsize=6.5, fontfamily='monospace',
                            color=COLOR_TEXT[cell], zorder=3)
                else:
                    ax.text(c + 0.5, r + 0.5, 'empty',
                            ha='center', va='center',
                            fontsize=7, color=self.MUTED,
                            fontfamily='monospace', zorder=3)

            # row label
            label = str(r + 1) if r < 4 else 'mv'
            ax.text(-0.08, r + 0.5, label,
                    ha='right', va='center',
                    fontsize=7.5, color=self.MUTED, fontfamily='monospace')

        for c, lbl in enumerate(['C1', 'C2', 'C3']):
            ax.text(c + 0.5, -0.08, lbl,
                    ha='center', va='bottom',
                    fontsize=7, color=self.MUTED, fontfamily='monospace')

        self.fig.canvas.draw_idle()

    def _diff(self, s1, s2):
        return {(r, c) for r in range(ROWS) for c in range(COLS)
                if s1.grid[r][c] != s2.grid[r][c]}

    def _set_step(self, text):
        self.step_text.set_text(text)
        self.fig.canvas.draw_idle()

    def _set_metrics(self, status, steps, nodes, time_ms, color=None):
        self.metric_text.set_text(
            f'Status   : {status}\n'
            f'Steps    : {steps}\n'
            f'Nodes    : {nodes}\n'
            f'Time(ms) : {time_ms}'
        )
        self.metric_text.set_color(color or self.TEXT)
        self.fig.canvas.draw_idle()

    # ── ALGORITHM DISPATCH ───────────────────────
    def _get_algo(self):
        label = self.radio.value_selected
        if label == 'BFS':
            return lambda s: bfs(s)
        if label == 'IDDFS':
            return lambda s: iddfs(s)
        if label == 'A* - Row Distance':
            return lambda s: astar(s, h_row_distance)
        if label == 'A* - Row Conflict':
            return lambda s: astar(s, h_row_conflict)
        if label == 'Greedy BFS':
            return lambda s: gbfs(s, h_row_conflict)
        if label == 'Uniform Cost Search':
            return lambda s: ucs(s)

    # ── BUTTON CALLBACKS ─────────────────────────
    def new_puzzle(self, event):
        self.stop_anim(None)
        self.state    = State.generate_random(150)
        self.initial  = self.state.clone()
        self.solution = []
        self.step_idx = 0
        self.draw_grid(self.state)
        self._set_metrics('Ready', '--', '--', '--')
        self._set_step('')

    def reset(self, event):
        self.stop_anim(None)
        self.state    = self.initial.clone()
        self.solution = []
        self.step_idx = 0
        self.draw_grid(self.state)
        self._set_metrics('Reset', '--', '--', '--')
        self._set_step('')

    def solve(self, event):
        if self.solving:
            return
        self.stop_anim(None)
        self.solution = []
        self.step_idx = 0
        self.solving  = True
        self._set_metrics('Solving...', '--', '--', '--', color=self.YELLOW)

        algo_fn = self._get_algo()
        snap    = self.initial.clone()

        def _run():
            t0 = time.perf_counter()
            path, nodes, msg = algo_fn(snap)
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            self.solving = False
            if path:
                self.solution = path
                self.step_idx = 0
                self.state    = path[0]
                steps = len(path) - 1
                self.draw_grid(self.state)
                self._set_metrics(msg, steps, f'{nodes:,}', elapsed,
                                  color=self.GREEN)
                self._set_step(f'Step 0 / {steps}  — press STEP FWD or ANIMATE')
            else:
                self._set_metrics(msg, '--', f'{nodes:,}', elapsed,
                                  color=self.RED)
                self._set_step('No solution — try another algorithm or new puzzle')

        threading.Thread(target=_run, daemon=True).start()

    def step_fwd(self, event):
        if not self.solution:
            self._set_step('Solve first, then navigate.')
            return
        if self.step_idx < len(self.solution) - 1:
            prev = self.solution[self.step_idx]
            self.step_idx += 1
            curr = self.solution[self.step_idx]
            self.state = curr
            self.draw_grid(curr, self._diff(prev, curr))
            total = len(self.solution) - 1
            tag = '  >>> GOAL REACHED!' if self.step_idx == total else ''
            self._set_step(f'Step {self.step_idx} / {total}{tag}')

    def step_back(self, event):
        if not self.solution or self.step_idx == 0:
            return
        self.step_idx -= 1
        curr = self.solution[self.step_idx]
        self.state = curr
        self.draw_grid(curr)
        total = len(self.solution) - 1
        self._set_step(f'Step {self.step_idx} / {total}')

    def animate(self, event):
        if not self.solution:
            self._set_step('Solve first.')
            return
        self.animating = True
        self.step_idx  = 0
        self.state     = self.solution[0]
        self.draw_grid(self.state)
        threading.Thread(target=self._anim_loop, daemon=True).start()

    def _anim_loop(self):
        total = len(self.solution) - 1
        while self.animating and self.step_idx < total:
            prev = self.solution[self.step_idx]
            self.step_idx += 1
            curr = self.solution[self.step_idx]
            self.state = curr
            self.draw_grid(curr, self._diff(prev, curr))
            self._set_step(f'Step {self.step_idx} / {total}')
            time.sleep(self.ANIM_INTERVAL)
        if self.animating:
            self.animating = False
            self._set_step(f'Animation complete — {total} steps')

    def stop_anim(self, event):
        self.animating = False

    # ── COMPARE ALL ──────────────────────────────
    def compare_all(self, event):
        if self.solving:
            return
        self.solving = True
        self._set_metrics('Comparing all...', '--', '--', '--', color=self.YELLOW)
        snap = self.initial.clone()

        def _run():
            configs = [
                ('BFS',             lambda s: bfs(s)),
                ('IDDFS',           lambda s: iddfs(s)),
                ('A* Row Dist',     lambda s: astar(s, h_row_distance)),
                ('A* Row Conflict', lambda s: astar(s, h_row_conflict)),
                ('Greedy BFS',      lambda s: gbfs(s, h_row_conflict)),
                ('UCS',             lambda s: ucs(s)),
            ]
            results = []
            for name, fn in configs:
                t0 = time.perf_counter()
                path, nodes, msg = fn(snap.clone())
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                steps = len(path) - 1 if path else None
                results.append((name, steps, nodes, elapsed, msg))

            self.solving = False
            self._show_comparison(results)

        threading.Thread(target=_run, daemon=True).start()

    def _show_comparison(self, results):
        fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig2.patch.set_facecolor(self.BG)
        fig2.suptitle('Algorithm Comparison — Same Puzzle',
                      color=self.ACCENT, fontsize=13,
                      fontfamily='monospace', fontweight='bold')

        names  = [r[0] for r in results]
        steps  = [r[1] if r[1] is not None else 0 for r in results]
        nodes  = [r[2] for r in results]
        times  = [r[3] for r in results]
        bcolors = [self.ACCENT, self.GREEN, self.GREEN, self.GREEN,
                   self.YELLOW, self.PURPLE]

        def bar_chart(ax, vals, title, ylabel):
            ax.set_facecolor(self.PANEL)
            bars = ax.bar(range(len(names)), vals,
                          color=bcolors, edgecolor=self.BORDER, linewidth=0.6)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha='right',
                               fontfamily='monospace', fontsize=7.5,
                               color=self.TEXT)
            ax.set_title(title, fontfamily='monospace',
                         fontsize=9, color=self.MUTED)
            ax.set_ylabel(ylabel, fontfamily='monospace',
                          fontsize=8, color=self.TEXT)
            ax.tick_params(colors=self.TEXT, labelsize=7.5)
            for spine in ax.spines.values():
                spine.set_edgecolor(self.BORDER)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2,
                        b.get_height() + max(vals)*0.01 if max(vals) > 0 else 0.1,
                        str(v), ha='center', va='bottom',
                        fontsize=7.5, color=self.TEXT, fontfamily='monospace')

        bar_chart(axes[0], steps, 'Steps to Solution', 'Steps')
        bar_chart(axes[1], nodes, 'Nodes Explored',    'Nodes')
        bar_chart(axes[2], times, 'Time (ms)',          'ms')

        # Summary table
        col_labels = ['Algorithm', 'Steps', 'Nodes', 'Time(ms)', 'Status']
        table_data = [
            [r[0], r[1] if r[1] is not None else '--',
             f'{r[2]:,}', r[3], r[4]]
            for r in results
        ]
        ax_tbl = fig2.add_axes([0.03, 0.0, 0.94, 0.2])
        ax_tbl.axis('off')
        ax_tbl.set_facecolor(self.BG)
        tbl = ax_tbl.table(
            cellText=table_data, colLabels=col_labels,
            loc='center', cellLoc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor(self.PANEL if row > 0 else self.BORDER)
            cell.set_edgecolor(self.BORDER)
            cell.set_text_props(color=self.TEXT, fontfamily='monospace')

        fig2.tight_layout(rect=[0, 0.2, 1, 0.95])
        self._set_metrics('Compare done', '--', '--', '--', color=self.ACCENT)
        plt.show()


# ─────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    app = ColorSorterGUI()
