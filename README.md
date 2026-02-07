Overview
--------
This repository reproduces two Multi-Agent Path Finding (MAPF) algorithms implemented on a 3D grid:

- `CBS` (Conflict-Based Search)
- `EECBS` (Explicit Estimation CBS, with a high-level selection strategy using weighted A*)

You can start the demo by running `visualize.py`. The script runs both algorithms and shows 3D visualizations of the results; it also prints each algorithm's SOC (Sum of Costs) and solving time (milliseconds) to the console.

Main files
----------
- `cbs3d.py` — Implementation of CBS (low-level A*, conflict detection, high-level splitting).
- `eecbs3d.py` — Implementation of EECBS (weighted A* on the low level, EES-like high-level selection).
- `visualize.py` — Generates random maps/obstacles/starts-goals and runs visualization (static and animated).

Dependencies
------------
- Python 3.8+
- numpy
- matplotlib

Installation (example)
----------------------
Install the Python dependencies with pip:

```bash
pip install numpy matplotlib
```

Quick start
-----------
From the project root run:

```bash
python visualize.py
```

What the script does:
1. Randomly generates obstacles as well as start and goal positions (parameters can be modified inside `visualize.py`).
2. Runs `CBS` and `EECBS` in sequence (if available). The console prints SOC and runtime (ms) for each algorithm.
3. Pops up a 3D visualization window showing every agent's trajectory, with start positions as filled circles and goal positions marked by crosses.

Main configurable options (in `visualize.py`)
-------------------------------------------
- Grid size: `grid = (X, Y, Z)`
- Number of agents: `NUM_AGENTS`
- Obstacle generation:
  - Box obstacles: parameters of `gen_random_cuboid_obstacles` such as `num_cuboids`, `min_size`, `max_size`, `seed`
  - Density obstacles: `gen_random_density_obstacles`
- Start/goal generation: `gen_random_starts_goals` options like `min_start_goal_dist`, `seed`
- Visualization options (passed to `visualize_cbs` / `animate_cbs`):
  - `obs_mode`: `'boxes'` (smoother), `'surface'`, `'voxels'`
  - `grid_mode`: `'outer'` / `'fine'` / `'none'`
  - `annotate_time`, `show_nodes`, `show_legend`
  - `interactive_lod`: temporarily hide obstacles while dragging to improve interactivity
- EECBS weight: set `w` when constructing `EECBS3D(...)` (for example `w=1.05`).

Outputs
-------
- SOC (Sum of Costs): sum over agents of (path length - 1), i.e., total movement steps of all agents.
- Solving time: printed to console in milliseconds.
- The visualization draws each agent's path; starts are shown as green circles and goals as red crosses.

Saving images / GIFs
--------------------
- Static image: call `visualize_cbs(..., save_path='out.png')`.
- Animation: call `animate_cbs(..., save_gif='out.gif')` (requires Pillow). If GIF saving fails, the script falls back to showing the animation interactively.

Performance notes
-----------------
- Large grids or many agents significantly increase solve time and memory usage. Lower `NUM_AGENTS`, reduce obstacle density, or increase grid size to improve success rates.
- For interactive visualization use `obs_mode='boxes'` together with `interactive_lod=True` for smoother interaction.


