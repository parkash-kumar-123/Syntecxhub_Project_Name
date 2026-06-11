"""
Maze Solver using A* Search
============================
- Represent a maze/grid (start, goal, walls)
- Model cells as nodes
- A* search with Manhattan or Euclidean heuristic
- Returns shortest path and handles unreachable cases
- Console + matplotlib visualization
"""

import heapq
import math
import os

# ── Optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────────────────────────

class Node:
    """Represents a single cell in the maze grid."""

    def __init__(self, position: tuple[int, int], parent=None):
        self.position = position   # (row, col)
        self.parent   = parent

        self.g = 0   # cost from start to this node
        self.h = 0   # heuristic estimate to goal
        self.f = 0   # f = g + h

    # Heap comparisons use f; break ties with position for stability
    def __lt__(self, other):
        return (self.f, self.position) < (other.f, other.position)

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

    def __repr__(self):
        return f"Node({self.position}, g={self.g}, h={self.h:.2f}, f={self.f:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
#  Heuristics
# ─────────────────────────────────────────────────────────────────────────────

def manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Manhattan distance – admissible for 4-directional grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean distance – admissible for any grid."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Maze
# ─────────────────────────────────────────────────────────────────────────────

class Maze:
    """
    Grid-based maze.

    Legend
    ------
    0  – open cell
    1  – wall
    S  – start  (stored separately; treated as open)
    G  – goal   (stored separately; treated as open)
    """

    OPEN  = 0
    WALL  = 1

    def __init__(self, grid: list[list[int]],
                 start: tuple[int, int],
                 goal:  tuple[int, int]):
        self.grid  = grid
        self.rows  = len(grid)
        self.cols  = len(grid[0])
        self.start = start
        self.goal  = goal
        self._validate()

    # ── validation ────────────────────────────────────────────────────────────

    def _validate(self):
        for pos, name in [(self.start, "start"), (self.goal, "goal")]:
            r, c = pos
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"{name} {pos} is outside the grid.")
            if self.grid[r][c] == self.WALL:
                raise ValueError(f"{name} {pos} is on a wall.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def is_walkable(self, row: int, col: int) -> bool:
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.grid[row][col] != self.WALL)

    def neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """4-directional (N, S, E, W) neighbours."""
        r, c = pos
        candidates = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return [(nr, nc) for nr, nc in candidates if self.is_walkable(nr, nc)]


# ─────────────────────────────────────────────────────────────────────────────
#  A* Solver
# ─────────────────────────────────────────────────────────────────────────────

class AStarSolver:
    """
    Solves a Maze with A* search.

    Parameters
    ----------
    maze       : Maze instance
    heuristic  : 'manhattan' (default) or 'euclidean'
    """

    def __init__(self, maze: Maze, heuristic: str = "manhattan"):
        self.maze      = maze
        self.heuristic = HEURISTICS.get(heuristic, manhattan)
        # Internals populated after solve()
        self.path:      list[tuple[int, int]] = []
        self.visited:   set[tuple[int, int]]  = set()
        self.open_log:  list[tuple[int, int]] = []   # order nodes were expanded

    # ── core search ───────────────────────────────────────────────────────────

    def solve(self) -> list[tuple[int, int]] | None:
        """
        Run A* and return the path as a list of (row, col) tuples
        from start to goal, or None if no path exists.
        """
        maze   = self.maze
        h      = self.heuristic

        start_node = Node(maze.start)
        start_node.h = h(maze.start, maze.goal)
        start_node.f = start_node.h

        open_heap: list[Node] = []
        heapq.heappush(open_heap, start_node)

        open_set:   dict[tuple, Node] = {maze.start: start_node}
        closed_set: set[tuple]        = set()

        while open_heap:
            current = heapq.heappop(open_heap)

            # Lazy deletion: skip stale heap entries
            if current.position in closed_set:
                continue

            closed_set.add(current.position)
            self.visited.add(current.position)
            self.open_log.append(current.position)

            # ── goal reached ──────────────────────────────────────────────────
            if current.position == maze.goal:
                self.path = self._reconstruct(current)
                return self.path

            # ── expand neighbours ─────────────────────────────────────────────
            for nb_pos in maze.neighbors(current.position):
                if nb_pos in closed_set:
                    continue

                tentative_g = current.g + 1   # uniform step cost

                if nb_pos in open_set and open_set[nb_pos].g <= tentative_g:
                    continue

                nb_node   = Node(nb_pos, parent=current)
                nb_node.g = tentative_g
                nb_node.h = h(nb_pos, maze.goal)
                nb_node.f = nb_node.g + nb_node.h

                open_set[nb_pos] = nb_node
                heapq.heappush(open_heap, nb_node)

        # Open list exhausted → no path
        self.path = []
        return None

    # ── path reconstruction ───────────────────────────────────────────────────

    @staticmethod
    def _reconstruct(node: Node) -> list[tuple[int, int]]:
        path = []
        current = node
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation – Console
# ─────────────────────────────────────────────────────────────────────────────

def print_maze(maze: Maze,
               path:    list[tuple[int, int]] | None = None,
               visited: set[tuple[int, int]]  | None = None):
    """Pretty-print the maze in the terminal."""

    WALL_CH    = "██"
    OPEN_CH    = "  "
    VISIT_CH   = "· "
    PATH_CH    = "○ "
    START_CH   = "S "
    GOAL_CH    = "G "

    path_set    = set(path)    if path    else set()
    visited_set = set(visited) if visited else set()

    lines = []
    # top border
    lines.append("┌" + "──" * maze.cols + "┐")

    for r in range(maze.rows):
        row_str = "│"
        for c in range(maze.cols):
            pos = (r, c)
            if pos == maze.start:
                row_str += START_CH
            elif pos == maze.goal:
                row_str += GOAL_CH
            elif maze.grid[r][c] == Maze.WALL:
                row_str += WALL_CH
            elif pos in path_set:
                row_str += PATH_CH
            elif pos in visited_set:
                row_str += VISIT_CH
            else:
                row_str += OPEN_CH
        row_str += "│"
        lines.append(row_str)

    lines.append("└" + "──" * maze.cols + "┘")

    legend = (
        f"  Legend:  S=Start  G=Goal  {WALL_CH}=Wall  "
        f"{PATH_CH.strip()}=Path  {VISIT_CH.strip()}=Visited"
    )
    lines.append(legend)
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation – Matplotlib
# ─────────────────────────────────────────────────────────────────────────────

def plot_maze(maze: Maze,
              path:    list[tuple[int, int]] | None = None,
              visited: set[tuple[int, int]]  | None = None,
              title: str = "A* Maze Solver",
              save_path: str | None = None):
    """Plot the maze using matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        print("[plot_maze] matplotlib not available – skipping plot.")
        return

    COLORS = {
        "wall":    "#2c3e50",
        "open":    "#ecf0f1",
        "visited": "#aed6f1",
        "path":    "#2ecc71",
        "start":   "#e74c3c",
        "goal":    "#f39c12",
    }

    grid_img = np.zeros((maze.rows, maze.cols, 3))

    # Fill base colours
    for r in range(maze.rows):
        for c in range(maze.cols):
            color_hex = COLORS["wall"] if maze.grid[r][c] == Maze.WALL else COLORS["open"]
            grid_img[r, c] = _hex_to_rgb(color_hex)

    # Overlay visited
    if visited:
        for (r, c) in visited:
            if (r, c) not in (maze.start, maze.goal):
                grid_img[r, c] = _hex_to_rgb(COLORS["visited"])

    # Overlay path
    if path:
        for (r, c) in path:
            if (r, c) not in (maze.start, maze.goal):
                grid_img[r, c] = _hex_to_rgb(COLORS["path"])

    # Start / Goal
    sr, sc = maze.start
    gr, gc = maze.goal
    grid_img[sr, sc] = _hex_to_rgb(COLORS["start"])
    grid_img[gr, gc] = _hex_to_rgb(COLORS["goal"])

    fig, ax = plt.subplots(figsize=(max(6, maze.cols * 0.55),
                                    max(5, maze.rows * 0.55)))
    ax.imshow(grid_img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Draw grid lines
    for x in range(maze.cols + 1):
        ax.axvline(x - 0.5, color="#bdc3c7", linewidth=0.4)
    for y in range(maze.rows + 1):
        ax.axhline(y - 0.5, color="#bdc3c7", linewidth=0.4)

    # Labels
    ax.text(sc, sr, "S", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    ax.text(gc, gr, "G", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

    # Path arrows
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            ax.annotate("", xy=(c2, r2), xytext=(c1, r1),
                        arrowprops=dict(arrowstyle="->",
                                        color="#1a5276",
                                        lw=1.2))

    # Legend
    patches = [
        mpatches.Patch(color=COLORS["wall"],    label="Wall"),
        mpatches.Patch(color=COLORS["open"],    label="Open"),
        mpatches.Patch(color=COLORS["visited"], label="Visited"),
        mpatches.Patch(color=COLORS["path"],    label="Path"),
        mpatches.Patch(color=COLORS["start"],   label="Start"),
        mpatches.Patch(color=COLORS["goal"],    label="Goal"),
    ]
    ax.legend(handles=patches, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              fontsize=8, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved to {save_path}")
    else:
        plt.show()


def _hex_to_rgb(hex_color: str) -> list[float]:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build maze from ASCII string
# ─────────────────────────────────────────────────────────────────────────────

def maze_from_ascii(ascii_map: str) -> Maze:
    """
    Build a Maze from an ASCII string.

    Characters
    ----------
    '#'  → wall
    ' '  → open
    'S'  → start
    'G'  → goal
    """
    lines = [line for line in ascii_map.strip("\n").splitlines()]
    grid  = []
    start = goal = None

    for r, line in enumerate(lines):
        row = []
        for c, ch in enumerate(line):
            if ch == "#":
                row.append(Maze.WALL)
            elif ch == "S":
                start = (r, c)
                row.append(Maze.OPEN)
            elif ch == "G":
                goal = (r, c)
                row.append(Maze.OPEN)
            else:
                row.append(Maze.OPEN)
        grid.append(row)

    if start is None or goal is None:
        raise ValueError("ASCII map must contain 'S' (start) and 'G' (goal).")

    return Maze(grid, start, goal)


# ─────────────────────────────────────────────────────────────────────────────
#  Demo / Main
# ─────────────────────────────────────────────────────────────────────────────

DEMO_MAZE_ASCII = """\
#############
#S    #     #
### # # ### #
#   # #   # #
# ### ##### #
#       #   #
####### # # #
#     # # # #
# ### # # # #
# #   #   # #
# # ####  # #
#         G #
#############"""


def run_demo(heuristic: str = "manhattan",
             show_plot: bool = True,
             save_plot: str | None = None):
    print("=" * 55)
    print("       A* Maze Solver Demo")
    print("=" * 55)

    maze   = maze_from_ascii(DEMO_MAZE_ASCII)
    solver = AStarSolver(maze, heuristic=heuristic)

    print(f"\nHeuristic : {heuristic.capitalize()}")
    print(f"Grid size : {maze.rows} × {maze.cols}")
    print(f"Start     : {maze.start}")
    print(f"Goal      : {maze.goal}\n")

    path = solver.solve()

    print("── Initial Maze ──────────────────────────────────")
    print_maze(maze)

    if path:
        print(f"\n✔  Path found!  Length = {len(path) - 1} steps")
        print(f"   Cells visited during search : {len(solver.visited)}")
        print(f"   Path : {' → '.join(str(p) for p in path)}\n")

        print("── Maze with Visited Cells & Final Path ──────────")
        print_maze(maze, path=path, visited=solver.visited)

        if show_plot:
            plot_maze(
                maze,
                path    = path,
                visited = solver.visited,
                title   = f"A* Maze Solver ({heuristic.capitalize()} heuristic)",
                save_path = save_path,
            )
    else:
        print("✘  No path found – goal is unreachable from start.")
        print_maze(maze, visited=solver.visited)

    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Programmatic API example
# ─────────────────────────────────────────────────────────────────────────────

def custom_example():
    """Show how to build and solve a maze programmatically."""
    grid = [
        [0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    maze   = Maze(grid, start=(0, 0), goal=(4, 4))
    solver = AStarSolver(maze, heuristic="euclidean")
    path   = solver.solve()

    print("\n── Custom 5×5 Maze (Euclidean) ───────────────────")
    if path:
        print(f"Path ({len(path)-1} steps): {path}")
        print_maze(maze, path=path, visited=solver.visited)
    else:
        print("No path found.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    heuristic = "manhattan"
    show_plot = True
    save_path = None

    # Simple CLI: python maze_solver.py [manhattan|euclidean] [--no-plot] [--save FILE]
    args = sys.argv[1:]
    if args and args[0] in HEURISTICS:
        heuristic = args.pop(0)
    if "--no-plot" in args:
        show_plot = False
        args.remove("--no-plot")
    if "--save" in args:
        idx = args.index("--save")
        save_path = args[idx + 1] if idx + 1 < len(args) else "maze.png"

    run_demo(heuristic=heuristic, show_plot=show_plot, save_plot=save_path)
    custom_example()
