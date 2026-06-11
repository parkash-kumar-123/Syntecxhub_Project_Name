## About

A clean, well-structured Python implementation of the
**A\* search algorithm** applied to grid-based mazes.

Built as an educational project to demonstrate:
- Graph search with a priority queue (min-heap)
- Admissible heuristics: Manhattan and Euclidean distance
- Path reconstruction via parent pointers
- Handling of unreachable goal states

## Features

- ✅ A\* with Manhattan or Euclidean heuristic
- ✅ ASCII maze input and programmatic grid API
- ✅ Console visualization (Unicode, no dependencies)
- ✅ Matplotlib plot with path arrows + visited cells
- ✅ Handles walls, boundaries, and unreachable goals
- ✅ Clean OOP design: `Maze`, `Node`, `AStarSolver`

## Usage

```bash
python maze_solver.py                  # Manhattan heuristic
python maze_solver.py euclidean        # Euclidean heuristic
python maze_solver.py --no-plot        # Console only
python maze_solver.py --save maze.png  # Save plot
```

## Requirements

Python 3.10+ · matplotlib (optional, for plot)
