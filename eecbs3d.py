# eecbs3d.py
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional, Set, Union
from dataclasses import dataclass, field
import heapq
import time

Coord3D = Tuple[int, int, int]
Path = List[Coord3D]

# ---------------------------- Helpers ---------------------------- #

def manhattan_3d(a: Coord3D, b: Coord3D) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def solution_cost(paths: Dict[int, Path]) -> int:
    """Sum of costs (SOC): sum of path lengths-1."""
    return sum(max(0, len(p) - 1) for p in paths.values())

# ---------------------------- Constraints 先定义了两种，顶点冲突和边冲突 ---------------------------- #

@dataclass(frozen=True)
class Constraint:
    agent: int
    ctype: str                    # 'vertex' or 'edge'
    time: int
    loc: Union[Coord3D, Tuple[Coord3D, Coord3D]]  # vertex: (x,y,z) ; edge: ((ux,uy,uz),(vx,vy,vz))

# ---------------------------- Low-level: Weighted A* ---------------------------- #

class WAStar3D:
    """Weighted A* in space-time with vertex & edge constraints. Returns (path, fmin)."""
    def __init__(self, grid_size: Tuple[int,int,int], obstacles: Set[Coord3D], w: float = 1.0, max_slack: int = 120):
        self.X, self.Y, self.Z = grid_size
        self.occ = set(obstacles)
        self.w = max(1.0, float(w))
        self.max_slack = int(max_slack)

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.X and 0 <= y < self.Y and 0 <= z < self.Z

    def passable(self, x: int, y: int, z: int) -> bool:
        return (x, y, z) not in self.occ

    def neighbors(self, x: int, y: int, z: int) -> Iterable[Coord3D]:
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if self.in_bounds(nx, ny, nz) and self.passable(nx, ny, nz):
                yield (nx, ny, nz)
        # wait
        yield (x, y, z)

    def admissible_h(self, u: Coord3D, goal: Coord3D) -> int:
        return manhattan_3d(u, goal)

    def plan(self, agent: int, start: Coord3D, goal: Coord3D,
             constraints: List[Constraint], stats: Optional[dict] = None) -> Tuple[Optional[Path], int]:
        """Return (path, fmin) where fmin is minimal g+h encountered (unweighted lower bound)."""
        c_by_v = {}
        c_by_e = {}
        for c in constraints:
            if c.agent != agent:
                continue
            if c.ctype == 'vertex':
                c_by_v.setdefault(c.time, set()).add(c.loc)  # type: ignore
            else:
                c_by_e.setdefault(c.time, set()).add(c.loc)  # type: ignore

        T_cap = self.max_slack + self.admissible_h(start, goal)

        start_state = (start[0], start[1], start[2], 0)
        goal_cell = goal

        def forbidden_vertex(pos: Coord3D, t: int) -> bool:
            S = c_by_v.get(t);  return (S is not None) and (pos in S)

        def forbidden_edge(u: Coord3D, v: Coord3D, t: int) -> bool:
            S = c_by_e.get(t);  return (S is not None) and ((u, v) in S)

        openq: List[Tuple[float, int, Tuple[int,int,int,int]]] = []
        g: Dict[Tuple[int,int,int,int], int] = {}
        parent: Dict[Tuple[int,int,int,int], Tuple[int,int,int,int]] = {}

        h0 = self.admissible_h(start, goal_cell)
        fstart = 0 + self.w*h0
        heapq.heappush(openq, (fstart, 0, start_state))
        g[start_state] = 0

        fmin = 0 + h0  # best unweighted f seen so far

        while openq:
            f_w, _, s = heapq.heappop(openq)
            x, y, z, t = s
            gs = g[s]
            # Update fmin using unweighted g+h for LB tracking
            fmin = min(fmin, gs + self.admissible_h((x,y,z), goal_cell))

            if (x, y, z) == goal_cell:
                # reconstruct path (collapse waiting tail)
                path_cells: List[Coord3D] = []
                cur = s
                while True:
                    cx, cy, cz, ct = cur
                    path_cells.append((cx, cy, cz))
                    if cur == start_state:
                        break
                    cur = parent[cur]
                path_cells.reverse()
                # Trim trailing waits at goal (optional)
                while len(path_cells) >= 2 and path_cells[-1] == path_cells[-2]:
                    path_cells.pop()
                return path_cells, fmin

            if t > T_cap:
                continue

            # expand
            for v in self.neighbors(x, y, z):
                nx, ny, nz = v
                nt = t + 1
                if forbidden_vertex((nx,ny,nz), nt):
                    continue
                if forbidden_edge((x,y,z), (nx,ny,nz), t):
                    continue
                s2 = (nx, ny, nz, nt)
                ng = gs + 1
                if ng < g.get(s2, 10**9):
                    g[s2] = ng
                    parent[s2] = s
                    h = self.admissible_h((nx,ny,nz), goal_cell)
                    # weighted A* priority
                    f = ng + self.w * h
                    heapq.heappush(openq, (f, nt, s2))
                    # track lower bound too
                    fmin = min(fmin, ng + h)

        return None, fmin

# ---------------------------- Conflict detection ---------------------------- #

@dataclass
class Conflict:
    a1: int
    a2: int
    time: int
    ctype: str   # 'vertex' or 'edge'
    loc: Union[Coord3D, Tuple[Coord3D, Coord3D]]  # vertex: pos ; edge: ((u1->v1),(u2->v2))

def pad_paths(paths: Dict[int, Path]) -> Dict[int, Path]:
    T = max(len(p) for p in paths.values())
    out = {}
    for a, p in paths.items():
        if len(p) < T:
            out[a] = p + [p[-1]] * (T - len(p))
        else:
            out[a] = p
    return out

def detect_conflict(paths: Dict[int, Path]) -> Optional[Conflict]:
    """Return earliest conflict (vertex or edge), else None."""
    if not paths:
        return None
    P = pad_paths(paths)
    agents = sorted(P.keys())
    T = max(len(p) for p in P.values())
    for t in range(T):
        # vertex conflicts
        pos_map = {}
        for a in agents:
            v = P[a][t]
            if v in pos_map:
                return Conflict(pos_map[v], a, t, 'vertex', v)
            pos_map[v] = a
        # edge conflicts
        if t+1 < T:
            for i in range(len(agents)):
                a = agents[i]
                u1, v1 = P[a][t], P[a][t+1]
                for j in range(i+1, len(agents)):
                    b = agents[j]
                    u2, v2 = P[b][t], P[b][t+1]
                    if u1 == v2 and v1 == u2:
                        # return both directed edges
                        return Conflict(a, b, t, 'edge', (u1, v1))  # (u1->v1) for a
                        # NOTE: we will forbid (u1->v1) for agent a, and (u2->v2) for agent b in the high-level
    return None

def count_conflicts(paths: Dict[int, Path]) -> int:
    """Count all conflicts across timeline (for heuristics)."""
    if not paths:
        return 0
    P = pad_paths(paths)
    agents = sorted(P.keys())
    T = max(len(p) for p in P.values())
    cnt = 0
    for t in range(T):
        # vertex
        seen = {}
        for a in agents:
            v = P[a][t]
            if v in seen:
                cnt += 1
            else:
                seen[v] = a
        # edges
        if t+1 < T:
            for i in range(len(agents)):
                a = agents[i]
                u1, v1 = P[a][t], P[a][t+1]
                for j in range(i+1, len(agents)):
                    b = agents[j]
                    u2, v2 = P[b][t], P[b][t+1]
                    if u1 == v2 and v1 == u2:
                        cnt += 1
    return cnt

# ---------------------------- High-level: EECBS (EES selection) ---------------------------- #

@dataclass(order=True)
class HLNode:
    priority: float
    lb: int
    cost: int
    uid: int
    constraints: List[Constraint] = field(compare=False, default_factory=list)
    paths: Dict[int, Path] = field(compare=False, default_factory=dict)
    fmins: Dict[int, int] = field(compare=False, default_factory=dict)  # per-agent lower bound

class EECBS3D:
    """
    EECBS算法简介：
    EECBS (Explicit Estimation CBS) in 3D grid.
    - w: weight >= 1 for bounded-suboptimality (w=1 -> CBS).
    - max_nodes: safety cap on expanded HL nodes.
    Notes:
      * Low-level uses Weighted A* (priority g + w*h) and also returns per-agent fmin (= min(g+h) seen).
      * HL lower bound lb(N) = sum_i fmin_i(N). This varies with constraints and keeps the search informed.
      * EES selection (simplified): among nodes with lb <= w * best_lb, prefer:
           1) minimal d(N) = #conflicts (distance-to-go proxy),
           2) then minimal hat_f(N) = cost(N) + alpha * d(N),
           3) else fallback to minimal lb.
        alpha is updated online from observed delta cost per conflict reduction.
    """
    def __init__(self, grid_size: Tuple[int,int,int], obstacles: Set[Coord3D], w: float = 1.05, max_nodes: int = 100000):
        self.grid = grid_size
        self.obstacles = set(obstacles)
        self.w = max(1.0, float(w))
        self.max_nodes = int(max_nodes)
        self.low = WAStar3D(grid_size, obstacles, w=self.w)
        self._uid = 0
        self.alpha = 1.0  # online estimate of cost increase per remaining conflict

    def _next_uid(self) -> int:
        self._uid += 1
        return self._uid

    def _replan_agent(self, agent: int, start: Coord3D, goal: Coord3D, constraints: List[Constraint]) -> Tuple[Optional[Path], int]:
        return self.low.plan(agent, start, goal, constraints)

    def _build_root(self, starts: List[Coord3D], goals: List[Coord3D]) -> Optional[HLNode]:
        n = len(starts)
        paths: Dict[int, Path] = {}
        fmins: Dict[int, int] = {}
        cons: List[Constraint] = []
        for a in range(n):
            p, fmin = self._replan_agent(a, starts[a], goals[a], cons)
            if p is None:
                return None
            paths[a] = p
            fmins[a] = fmin
        lb = sum(fmins.values())
        cost = solution_cost(paths)
        return HLNode(priority=lb, lb=lb, cost=cost, uid=self._next_uid(), constraints=cons, paths=paths, fmins=fmins)

    def _choose_node_ees(self, openq: List[HLNode]) -> HLNode:
        # best by lb (f)
        best_f = openq[0].lb
        # focal: nodes within w * best_f
        bound = self.w * best_f + 1e-9
        focal = [n for n in openq if n.lb <= bound]
        # 1) best_d: min number of conflicts
        def d_of(n: HLNode) -> int:
            return count_conflicts(n.paths)
        focal.sort(key=lambda n: d_of(n))
        if focal:
            return focal[0]
        # 2) best_hatf
        def hatf(n: HLNode) -> float:
            return n.cost + self.alpha * d_of(n)
        focal2 = [n for n in openq if n.lb <= bound]
        focal2.sort(key=lambda n: hatf(n))
        if focal2:
            return focal2[0]
        # 3) fallback: best_f
        return openq[0]

    def solve(self, starts: List[Coord3D], goals: List[Coord3D], return_time: bool = False
              ) -> Union[Optional[Dict[int, Path]], Tuple[Optional[Dict[int, Path]], float]]:

        assert len(starts) == len(goals), "starts/goals mismatch"

        t0 = time.perf_counter()

        root = self._build_root(starts, goals)
        if root is None:
            print("[EECBS] infeasible root")
            return None

        openq: List[HLNode] = [root]
        heapq.heapify(openq)  # order by priority (==lb)
        expanded = 0

        while openq:
            # we also need EES rule -> sort view by lb, then choose within focal
            open_sorted = sorted(openq, key=lambda nd: nd.lb)
            node = self._choose_node_ees(open_sorted)

            openq.remove(node)
            heapq.heapify(openq)

            expanded += 1
            if expanded > self.max_nodes:
                dt = time.perf_counter() - t0
                print(f"[EECBS] expanded>{self.max_nodes}, time={dt * 1000:.2f} ms")
                return (None, dt * 1000.0) if return_time else None

            conflict = detect_conflict(node.paths)
            if conflict is None:
                dt = time.perf_counter() - t0
                print(f"[EECBS] solve time = {dt * 1000:.2f} ms, expanded={expanded}")
                return (node.paths, dt * 1000.0) if return_time else node.paths

            a1, a2, t = conflict.a1, conflict.a2, conflict.time

            children: List[HLNode] = []
            if conflict.ctype == 'vertex':
                v = conflict.loc  # type: ignore
                cons1 = node.constraints + [Constraint(a1, 'vertex', t, v)]
                cons2 = node.constraints + [Constraint(a2, 'vertex', t, v)]
                for (agent, cons) in [(a1, cons1), (a2, cons2)]:
                    paths = dict(node.paths)
                    fmins = dict(node.fmins)
                    p, fmin = self._replan_agent(agent, starts[agent], goals[agent], cons)
                    if p is None:
                        continue
                    paths[agent] = p
                    fmins[agent] = fmin
                    lb = sum(fmins.values())
                    cost = solution_cost(paths)
                    child = HLNode(priority=lb, lb=lb, cost=cost, uid=self._next_uid(), constraints=cons, paths=paths, fmins=fmins)
                    children.append(child)

            elif conflict.ctype == 'edge':
                (u1, v1) = conflict.loc  # (u1->v1) for a1
                # 对称边约束：a1 禁 u1->v1；a2 禁 v1->u1
                cons1 = node.constraints + [Constraint(a1, 'edge', t, (u1, v1))]
                cons2 = node.constraints + [Constraint(a2, 'edge', t, (v1, u1))]
                for (agent, cons) in [(a1, cons1), (a2, cons2)]:
                    paths = dict(node.paths)
                    fmins = dict(node.fmins)
                    p, fmin = self._replan_agent(agent, starts[agent], goals[agent], cons)
                    if p is None:
                        continue
                    paths[agent] = p
                    fmins[agent] = fmin
                    lb = sum(fmins.values())
                    cost = solution_cost(paths)
                    child = HLNode(priority=lb, lb=lb, cost=cost, uid=self._next_uid(), constraints=cons, paths=paths, fmins=fmins)
                    children.append(child)
            else:
                raise ValueError("unknown conflict type ---请检查冲突类型定义")

            # online alpha update (coarse): use best child
            if children:
                children.sort(key=lambda c: (count_conflicts(c.paths), c.cost))
                best_child = children[0]
                d_par = count_conflicts(node.paths)
                d_ch = count_conflicts(best_child.paths)
                if d_ch < d_par:
                    delta_cost = best_child.cost - node.cost
                    delta_d = d_par - d_ch
                    if delta_d > 0:
                        est = delta_cost / float(delta_d)
                        self.alpha = 0.8 * self.alpha + 0.2 * max(0.0, est)

            for ch in children:
                heapq.heappush(openq, ch)

        dt = time.perf_counter() - t0
        print("[EECBS] open exhausted (no solution)")
        return (None, dt * 1000.0) if return_time else None


__all__ = ["EECBS3D", "Constraint", "solution_cost"]
