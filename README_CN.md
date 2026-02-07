简介
----
本项目复现了两种多机器人路径规划（MAPF）算法在三维栅格上的实现：
- `CBS`（Conflict\-Based Search）
- `EECBS`（Explicit Estimation CBS，带有加权 A\* 的高层选择策略）

直接运行 `visualize.py` 即可启动示例：程序将分别运行两种算法并显示结果的三维可视化，同时在控制台输出每种算法的 SOC（Sum of Costs）和求解时间（毫秒）。

主要文件
----
- `cbs3d.py` — CBS 算法实现（低层 A\*、冲突检测、高层分裂）。
- `eecbs3d.py` — EECBS 算法实现（加权 A\* 低层、EES 风格高层选择）。
- `visualize.py` — 随机生成地图/障碍/起终点并运行可视化（静态/动画）。

依赖
----
- Python 3.8+
- numpy
- matplotlib

安装（示例）
----
```bash
pip install numpy matplotlib
```

快速运行
----
在项目根目录运行：
```bash
python visualize.py
```
程序将：
1. 随机生成障碍、起点和终点（可在 `visualize.py` 中修改参数）。
2. 先后运行 `CBS` 和 `EECBS`（若可用），控制台打印 SOC 与耗时（ms）。
3. 弹出三维可视化窗口，显示每个机器人的轨迹、起点（圆点）和终点（叉号）。

主要可配置项（在 `visualize.py` 中）
----
- 网格尺寸：`grid = (X, Y, Z)`
- 机器人数量：`NUM_AGENTS`
- 障碍生成：
  - 盒式障碍：`gen_random_cuboid_obstacles` 的 `num_cuboids`, `min_size`, `max_size`, `seed`
  - 密度型障碍：`gen_random_density_obstacles`
- 起终点生成：`gen_random_starts_goals` 的 `min_start_goal_dist`, `seed`
- 可视化选项（传入 `visualize_cbs` / `animate_cbs`）：
  - `obs_mode`: `'boxes'`（更流畅）、`'surface'`、`'voxels'`
  - `grid_mode`: `'outer'` / `'fine'` / `'none'`
  - `annotate_time`, `show_nodes`, `show_legend`
  - `interactive_lod`: 拖动时临时隐藏障碍以提升交互性能
- EECBS 权重：在 `EECBS3D(...)` 的构造中设置 `w`（例如 `w=1.05`）

输出说明
----
- SOC（Sum of Costs）= 每个路径长度减一之和（所有机器人移动步数之和）。
- 求解时间以毫秒打印（控制台信息）。
- 可视化会绘制每个机器人的轨迹，起点标为绿色圆点，终点标为红色叉号。

保存图像 / GIF
----
- 静态图：在调用 `visualize_cbs(..., save_path='out.png')`。
- 动画：使用 `animate_cbs(..., save_gif='out.gif')`（需安装 pillow，当动画保存失败会回退显示）。

性能提示
----
- 大规模网格或大量机器人会显著增加求解时间与内存占用。调小 `NUM_AGENTS`、降低障碍密度或增大网格可提高成功率。
- 建议在可视化时使用 `obs_mode='boxes'` 与 `interactive_lod=True` 以获得更流畅的交互体验。
