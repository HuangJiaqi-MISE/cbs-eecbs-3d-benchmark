from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional, Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




Coord3D = Tuple[int, int, int]
Path = List[Coord3D]

# ---------------------------- 小工具 ---------------------------- #

def manhattan_3d(a: Coord3D, b: Coord3D) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])


def _pad_paths(paths: Dict[int, Path]) -> Dict[int, Path]:
    if not paths:
        return {}
    T = max(len(p) for p in paths.values())
    out: Dict[int, Path] = {}
    for a, p in paths.items():
        if len(p) < T:
            out[a] = p + [p[-1]] * (T - len(p))
        else:
            out[a] = p
    return out


def _centers(xs: Iterable[int], ys: Iterable[int], zs: Iterable[int]):
    for x, y, z in zip(xs, ys, zs):
        yield x + 0.5, y + 0.5, z + 0.5


# ---------------------------- 随机障碍生成 ---------------------------- #

def gen_random_cuboid_obstacles(
    grid_size: Tuple[int, int, int],
    num_cuboids: int = 6,
    min_size: Tuple[int, int, int] = (1, 1, 1),
    max_size: Tuple[int, int, int] = (2, 2, 2),
    seed: Optional[int] = None,
    keepout: Optional[Set[Coord3D]] = None,
) -> Set[Coord3D]:
    """生成若干随机轴对齐长方体障碍（体素集合）。"""
    rng = np.random.default_rng(seed)
    X, Y, Z = grid_size
    occ: Set[Coord3D] = set()
    keep = keepout or set()

    def clamp_range(s_min: int, s_max: int, limit: int) -> Tuple[int, int]:
        s_min = max(1, int(s_min))
        s_max = max(s_min, int(s_max))
        s_max = min(s_max, limit)
        return s_min, s_max

    sx_min, sx_max = clamp_range(min_size[0], max_size[0], X)
    sy_min, sy_max = clamp_range(min_size[1], max_size[1], Y)
    sz_min, sz_max = clamp_range(min_size[2], max_size[2], Z)

    for _ in range(int(num_cuboids)):
        sx = int(rng.integers(sx_min, sx_max + 1))
        sy = int(rng.integers(sy_min, sy_max + 1))
        sz = int(rng.integers(sz_min, sz_max + 1))
        x0 = int(rng.integers(0, max(1, X - sx + 1)))
        y0 = int(rng.integers(0, max(1, Y - sy + 1)))
        z0 = int(rng.integers(0, max(1, Z - sz + 1)))
        for x in range(x0, x0 + sx):
            for y in range(y0, y0 + sy):
                for z in range(z0, z0 + sz):
                    cell: Coord3D = (x, y, z)
                    if cell not in keep:
                        occ.add(cell)
    return occ


def gen_random_density_obstacles(
    grid_size: Tuple[int, int, int],
    density: float = 0.05,
    seed: Optional[int] = None,
    keepout: Optional[Set[Coord3D]] = None,
) -> Set[Coord3D]:
    """按给定密度逐体素采样生成障碍。density ∈ [0,1)。"""
    rng = np.random.default_rng(seed)
    X, Y, Z = grid_size
    occ: Set[Coord3D] = set()
    keep = keepout or set()
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                c: Coord3D = (x, y, z)
                if c in keep:
                    continue
                if rng.random() < density:
                    occ.add(c)
    return occ




# ---------------------------- 随机起终点生成 ---------------------------- #

def gen_random_starts_goals(
    grid_size: Tuple[int, int, int],
    num_agents: int,
    obstacles: Set[Coord3D],
    seed: Optional[int] = None,
    min_start_goal_dist: int = 0,
) -> Tuple[List[Coord3D], List[Coord3D]]:
    """
    在非障碍体素上随机生成 num_agents 对 (start, goal)。
    - min_start_goal_dist: 要求每对 (s,g) 的曼哈顿距离至少该值（0 表示不约束）。
    """
    X, Y, Z = grid_size
    free: List[Coord3D] = []
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                c = (x, y, z)
                if c not in obstacles:
                    free.append(c)

    need = 2 * num_agents
    if len(free) < need:
        raise ValueError(f"可用体素不足：需要 {need}，但只有 {len(free)}。请降低机器人数量或障碍密度/数量。")

    rng = np.random.default_rng(seed)

    # 尝试若干次满足最小起终点距离的分配
    attempts = 200
    for _ in range(attempts):
        rng.shuffle(free)
        starts = free[:num_agents]
        goals  = free[num_agents:2*num_agents]
        if min_start_goal_dist <= 0:
            return list(starts), list(goals)
        ok = True
        for s, g in zip(starts, goals):
            if abs(s[0]-g[0]) + abs(s[1]-g[1]) + abs(s[2]-g[2]) < min_start_goal_dist:
                ok = False
                break
        if ok:
            return list(starts), list(goals)

    # 兜底：忽略距离约束
    return list(free[:num_agents]), list(free[num_agents:2*num_agents])








# ---------------------------- 障碍体素 -> 合并长方体（贪心） ---------------------------- #

def pack_voxels_to_boxes(grid_size: Tuple[int,int,int], obstacles: Set[Coord3D]) -> List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]:
    """
    把占用体素集合近似合并成若干个不相交的大长方体（贪心）。
    返回 [(min_xyz), (max_xyz_exclusive)] 列表，右闭左开区间。
    复杂度 O(N) 左右，能显著减少绘制面数，拖动更流畅。
    """
    X, Y, Z = grid_size
    occ = np.zeros((X, Y, Z), dtype=bool)
    for (x,y,z) in obstacles:
        if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
            occ[x,y,z] = True
    visited = np.zeros_like(occ, dtype=bool)

    boxes: List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = []

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not occ[x,y,z] or visited[x,y,z]:
                    continue
                # 扩展 x 方向
                x1 = x
                while x1 < X and occ[x1,y,z] and not visited[x1,y,z]:
                    x1 += 1
                # 扩展 y 方向（要求这条 x-跨度上都为 True 且未访问）
                y1 = y
                while True:
                    ok = True
                    ny = y1
                    if ny >= Y:
                        break
                    for xi in range(x, x1):
                        if not occ[xi, ny, z] or visited[xi, ny, z]:
                            ok = False
                            break
                    if ok:
                        y1 += 1
                    else:
                        break
                # 扩展 z 方向（要求整个 x×y 范围都为 True 且未访问）
                z1 = z
                while True:
                    ok = True
                    nz = z1
                    if nz >= Z:
                        break
                    for xi in range(x, x1):
                        for yi in range(y, y1):
                            if not occ[xi, yi, nz] or visited[xi, yi, nz]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        z1 += 1
                    else:
                        break
                # 记录盒子并标记访问
                boxes.append(((x,y,z), (x1,y1,z1)))
                visited[x:x1, y:y1, z:z1] = True
    return boxes


# ---------------------------- 栅格/障碍绘制 ---------------------------- #

def _draw_lattice(ax, grid_size: Tuple[int, int, int], alpha: float = 0.15, grid: str = 'outer') -> None:
    """grid: 'outer' 只画外框, 'fine' 画细网格, 'none' 不画网格。"""
    if grid == 'none':
        return
    X, Y, Z = grid_size
    # 画出立方体边框
    for x in (0, X):
        for y in (0, Y):
            ax.plot([x, x], [y, y], [0, Z], lw=0.6, alpha=alpha, color='k')
    for x in (0, X):
        for z in (0, Z):
            ax.plot([x, x], [0, Y], [z, z], lw=0.6, alpha=alpha, color='k')
    for y in (0, Y):
        for z in (0, Z):
            ax.plot([0, X], [y, y], [z, z], lw=0.6, alpha=alpha, color='k')

    if grid == 'fine':
        for x in range(1, X):
            ax.plot([x, x], [0, Y], [0, 0], lw=0.2, alpha=alpha, color='k')
            ax.plot([x, x], [0, 0], [0, Z], lw=0.2, alpha=alpha, color='k')
            ax.plot([x, x], [Y, Y], [0, Z], lw=0.2, alpha=alpha, color='k')
        for y in range(1, Y):
            ax.plot([0, X], [y, y], [0, 0], lw=0.2, alpha=alpha, color='k')
            ax.plot([0, 0], [y, y], [0, Z], lw=0.2, alpha=alpha, color='k')
            ax.plot([X, X], [y, y], [0, Z], lw=0.2, alpha=alpha, color='k')
        for z in range(1, Z):
            ax.plot([0, 0], [0, Y], [z, z], lw=0.2, alpha=alpha, color='k')
            ax.plot([0, X], [0, 0], [z, z], lw=0.2, alpha=alpha, color='k')
            ax.plot([0, X], [Y, Y], [z, z], lw=0.2, alpha=alpha, color='k')

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_zlim(0, Z)


def _draw_obstacles(
    ax,
    grid_size: Tuple[int, int, int],
    obstacles: Set[Coord3D],
    facecolor=(0.2, 0.2, 0.2, 0.6),
    mode: str = 'surface',    # 'surface' | 'voxels' | 'boxes'
    edge: bool = False,       # 画边会非常慢，默认关
) -> Poly3DCollection | None:
    """返回 Poly3DCollection 以便交互时开关可见性。"""
    X, Y, Z = grid_size
    if not obstacles:
        return None

    if mode == 'voxels':
        occ = np.zeros((X, Y, Z), dtype=bool)
        for (x, y, z) in obstacles:
            if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
                occ[x, y, z] = True
        ax.voxels(occ, facecolors=facecolor, edgecolor='k' if edge else None, linewidth=0.2 if edge else 0)
        return None

    if mode == 'boxes':
        boxes = pack_voxels_to_boxes(grid_size, obstacles)
        quads = []
        for (x0,y0,z0), (x1,y1,z1) in boxes:
            # 6 个面
            quads.extend([
                [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],  # -X
                [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)],  # +X
                [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],  # -Y
                [(x0,y1,z0),(x0,y1,z1),(x1,y1,z1),(x1,y1,z0)],  # +Y
                [(x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0)],  # -Z
                [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # +Z
            ])
        poly = Poly3DCollection(quads, facecolors=[facecolor], edgecolor='k' if edge else 'none', linewidths=0.2 if edge else 0, antialiased=False)
        ax.add_collection3d(poly)
        return poly

    # 默认：surface —— 仅绘制可见外表面
    occ = np.zeros((X, Y, Z), dtype=bool)
    for (x, y, z) in obstacles:
        if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
            occ[x, y, z] = True

    quads = []
    dirs = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
    for x, y, z in obstacles:
        for dx, dy, dz in dirs:
            nx, ny, nz = x+dx, y+dy, z+dz
            if not (0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z) or not occ[nx, ny, nz]:
                x0, y0, z0 = x, y, z
                x1, y1, z1 = x+1, y+1, z+1
                if dx == -1:
                    quad = [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)]
                elif dx == 1:
                    quad = [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)]
                elif dy == -1:
                    quad = [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)]
                elif dy == 1:
                    quad = [(x0,y1,z0),(x0,y1,z1),(x1,y1,z1),(x1,y1,z0)]
                elif dz == -1:
                    quad = [(x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0)]
                else:
                    quad = [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
                quads.append(quad)

    poly = Poly3DCollection(quads, facecolors=[facecolor], edgecolor='k' if edge else 'none', linewidths=0.2 if edge else 0, antialiased=False)
    ax.add_collection3d(poly)
    return poly


# ---------------------------- 路径绘制 ---------------------------- #

def _color_cycle(n: int):
    cmap = plt.get_cmap('tab20')
    for i in range(n):
        yield cmap(i % cmap.N)


def _enable_drag_lod(fig, heavy_artists: List[Poly3DCollection], make_light=None):
    """拖动时暂时隐藏重物体（障碍），鼠标松开再显示；可选创建更轻的占位图形。"""
    light = []
    if make_light is None:
        def make_light():
            return []
    dragging = {'on': False}

    def on_press(event):
        if event.button != 1:
            return
        dragging['on'] = True
        for art in heavy_artists:
            if art is not None:
                art.set_visible(False)
        # 生成轻量占位
        new_light = make_light()
        if new_light:
            light.extend(new_light)
        fig.canvas.draw_idle()

    def on_release(event):
        if dragging['on']:
            dragging['on'] = False
            for art in heavy_artists:
                if art is not None:
                    art.set_visible(True)
            # 移除占位
            for art in light:
                try:
                    art.remove()
                except Exception:
                    pass
            light.clear()
            fig.canvas.draw_idle()

    cid1 = fig.canvas.mpl_connect('button_press_event', on_press)
    cid2 = fig.canvas.mpl_connect('button_release_event', on_release)
    return (cid1, cid2)


def visualize_cbs(
    grid_size: Tuple[int, int, int],
    obstacles: Set[Coord3D],
    paths: Dict[int, Path],
    starts: Optional[List[Coord3D]] = None,
    goals: Optional[List[Coord3D]] = None,
    title: str = '3D CBS Visualization',
    annotate_time: bool = False,
    save_path: Optional[str] = None,
    elev: int = 18,
    azim: int = -60,
    grid_mode: str = 'outer',
    obs_mode: str = 'boxes',   # 默认用 boxes，更流畅
    show_nodes: bool = False,
    show_legend: bool = True,
    interactive_lod: bool = True,
) -> None:
    """
    静态渲染：三维栅格 + 障碍 + 多机器人路径（不同颜色）。
    性能选项：grid_mode='outer'/'fine'/'none'，obs_mode='surface'|'voxels'|'boxes'，show_nodes=False。
    interactive_lod=True 时，拖动时临时隐藏障碍，松开后恢复。
    """
    X, Y, Z = grid_size
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    _draw_lattice(ax, grid_size, grid=grid_mode)
    poly = _draw_obstacles(ax, grid_size, obstacles, mode=obs_mode)

    ax.set_box_aspect((X, Y, Z))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    colors = list(_color_cycle(len(paths)))

    for idx, (agent, p) in enumerate(sorted(paths.items())):
        xs, ys, zs = zip(*list(_centers([q[0] for q in p], [q[1] for q in p], [q[2] for q in p])))
        color = colors[idx]
        ax.plot(xs, ys, zs, '-', lw=1.8, color=color, label=f'A{agent}')
        if show_nodes:
            ax.scatter(xs, ys, zs, s=10, color=color, depthshade=False)
        if annotate_time:
            for t, (x, y, z) in enumerate(p):
                cx, cy, cz = x + 0.5, y + 0.5, z + 0.5
                ax.text(cx, cy, cz, str(t), fontsize=7, color=color)

    if starts is not None:
        for a, s in enumerate(starts):
            sx, sy, sz = s[0] + 0.5, s[1] + 0.5, s[2] + 0.5
            ax.scatter([sx], [sy], [sz], s=70, marker='o', color='lime', edgecolors='k', zorder=5)
    if goals is not None:
        for a, g in enumerate(goals):
            gx, gy, gz = g[0] + 0.5, g[1] + 0.5, g[2] + 0.5
            ax.scatter([gx], [gy], [gz], s=95, marker='X', color='red', edgecolors='k', zorder=6)

    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.02))
    plt.tight_layout()

    # 拖动 LOD：隐藏重障碍，拖动更流畅
    if interactive_lod:
        def _make_light():
            # 可选占位：画一个外框或简单的边界框。这里直接不画占位。
            return []
        _enable_drag_lod(fig, [poly], make_light=_make_light)

    if save_path:
        fig.savefig(save_path, dpi=130)
    else:
        plt.show()


# ---------------------------- 动画（可选） ---------------------------- #

def animate_cbs(
    grid_size: Tuple[int, int, int],
    obstacles: Set[Coord3D],
    paths: Dict[int, Path],
    starts: Optional[List[Coord3D]] = None,
    goals: Optional[List[Coord3D]] = None,
    interval_ms: int = 600,
    title: str = '3D CBS Animation',
    save_gif: Optional[str] = None,
    elev: int = 18,
    azim: int = -60,
    grid_mode: str = 'outer',
    obs_mode: str = 'boxes',
    tail_len: Optional[int] = 20,
    frame_skip: int = 1,
    show_legend: bool = False,
    interactive_lod: bool = True,
) -> None:
    """时间步动画（按 t 播放）。
    性能选项：tail_len 仅绘制最近 L 长度轨迹，frame_skip 跳帧；obs_mode='boxes' 更快。
    interactive_lod=True：拖动时隐藏障碍。
    """
    X, Y, Z = grid_size
    paths_pad = _pad_paths(paths)
    agents = sorted(paths_pad.keys())
    T = max(len(p) for p in paths_pad.values())

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    _draw_lattice(ax, grid_size, grid=grid_mode)
    poly = _draw_obstacles(ax, grid_size, obstacles, mode=obs_mode)

    ax.set_box_aspect((X, Y, Z))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    colors = list(_color_cycle(len(agents)))

    lines: Dict[int, any] = {}
    points: Dict[int, any] = {}
    for i, a in enumerate(agents):
        color = colors[i]
        line, = ax.plot([], [], [], '-', lw=1.8, color=color, label=f'A{a}')
        pt = ax.scatter([], [], [], s=60, color=color, edgecolors='k', zorder=10)
        lines[a] = line
        points[a] = pt

    if starts is not None:
        for a, s in enumerate(starts):
            sx, sy, sz = s[0] + 0.5, s[1] + 0.5, s[2] + 0.5
            ax.scatter([sx], [sy], [sz], s=70, marker='o', color='lime', edgecolors='k', zorder=5)
    if goals is not None:
        for a, g in enumerate(goals):
            gx, gy, gz = g[0] + 0.5, g[1] + 0.5, g[2] + 0.5
            ax.scatter([gx], [gy], [gz], s=95, marker='X', color='red', edgecolors='k', zorder=6)

    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.02))
    plt.tight_layout()

    def _frame(t: int):
        for i, a in enumerate(agents):
            p = paths_pad[a]
            start_idx = 0 if (tail_len is None) else max(0, t - int(tail_len) + 1)
            seg = p[start_idx: t + 1]
            xs = [q[0] + 0.5 for q in seg]
            ys = [q[1] + 0.5 for q in seg]
            zs = [q[2] + 0.5 for q in seg]
            lines[a].set_data(xs, ys)
            lines[a].set_3d_properties(zs)
            points[a]._offsets3d = ([xs[-1]], [ys[-1]], [zs[-1]])
        ax.set_title(f"{title}  |  t={t}")
        return list(lines.values()) + list(points.values())

    frames_idx = list(range(0, T, max(1, int(frame_skip))))
    if frames_idx[-1] != T-1:
        frames_idx.append(T-1)
    anim = animation.FuncAnimation(fig, _frame, frames=frames_idx, interval=interval_ms, blit=False)

    if interactive_lod:
        def _make_light():
            return []
        _enable_drag_lod(fig, [poly], make_light=_make_light)

    if save_gif:
        try:
            anim.save(save_gif, writer='pillow', dpi=90)
        except Exception as e:
            print('保存 GIF 失败：', e)
            plt.show()
    else:
        plt.show()


# ---------------------------- 启动 ---------------------------- #

if __name__ == '__main__':
    try:
        from cbs3d import CBS3D, solution_cost  # type: ignore
        from eecbs3d import EECBS3D
        has_solver = True
    except Exception:
        has_solver = False

    grid = (100, 100, 20)
    NUM_AGENTS = 110

    obstacles = gen_random_cuboid_obstacles(
        grid_size=grid,
        num_cuboids=60,
        min_size=(2,2,2),
        max_size=(7,5,4),
        seed=None,
    )

    starts, goals = gen_random_starts_goals(grid, NUM_AGENTS, obstacles, seed=None, min_start_goal_dist=3)

    if has_solver:
        cbs = CBS3D(grid, obstacles)  # type: ignore
        sol_cbs, cbs_ms = cbs.solve(starts, goals, return_time=True)  # type: ignore

        eecbs = EECBS3D(grid, obstacles, w=1.05, max_nodes=100000)
        sol_eecbs, eecbs_ms = eecbs.solve(starts, goals, return_time=True)

    # --- 分别绘制 CBS 与 EECBS 两张图 ---
    if sol_cbs is None:
        print('CBS 无解：降低障碍密度/数量、减少 NUM_AGENTS，或放大 grid。')
    else:
        cbs_soc = solution_cost(sol_cbs)
        print(f'CBS: SOC={cbs_soc}, time={cbs_ms:.2f} ms')
        visualize_cbs(
            grid, obstacles, sol_cbs, starts, goals,
            title=f'CBS | SOC={cbs_soc} | {cbs_ms:.2f} ms',
            annotate_time=False,
            grid_mode='outer',
            obs_mode='boxes',
            show_nodes=False,
            show_legend=False,
            interactive_lod=True,
        )

    if sol_eecbs is None:
        print('EECBS 无解：尝试调大 w 或减少障碍/机器人密度。')
    else:
        eecbs_soc = solution_cost(sol_eecbs)
        print(f'EECBS: SOC={eecbs_soc}, time={eecbs_ms:.2f} ms')
        visualize_cbs(
            grid, obstacles, sol_eecbs, starts, goals,
            title=f'EECBS (w=1.05) | SOC={eecbs_soc} | {eecbs_ms:.2f} ms',
            annotate_time=False,
            grid_mode='outer',
            obs_mode='boxes',
            show_nodes=False,
            show_legend=False,
            interactive_lod=True,
        )



