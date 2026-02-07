from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Iterable, Union
import heapq
from collections import defaultdict

import time


Coord3D = Tuple[int, int, int]
Edge3D = Tuple[Coord3D, Coord3D]
Path = List[Coord3D]

# ---------------------------- Utility helpers ---------------------------- #

def manhattan_3d(a: Coord3D, b: Coord3D) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])


def neighbors_6(pos: Coord3D, grid_size: Tuple[int, int, int], obstacles: Set[Coord3D], allow_wait: bool=True) -> Iterable[Coord3D]:
    """
    定义搜索邻域，6+1
    6-connected 3D grid neighbors (axis-aligned), optionally including wait move.
    Returns only in-bounds, non-obstacle cells.
    """
    x, y, z = pos
    X, Y, Z = grid_size
    moves = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    if allow_wait:
        moves.append((0,0,0))
    for dx,dy,dz in moves:
        nx, ny, nz = x+dx, y+dy, z+dz
        if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z and (nx,ny,nz) not in obstacles:
            yield (nx, ny, nz)


# ---------------------------- Constraint model ---------------------------- #

@dataclass(frozen=True)
class Constraint:
    agent: int
    ctype: str   # 'vertex' or 'edge'
    time: int    # time index of arrival (t for being at a vertex at time t; t for edge arrival)
    loc: Tuple   # vertex: (x,y,z); edge: ((x,y,z),(x,y,z))


# ---------------------------- Low-level search (A*) ---------------------------- #

class AStar3D:
    def __init__(self, grid_size: Tuple[int,int,int], obstacles: Set[Coord3D], allow_wait: bool=True):
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.allow_wait = allow_wait

    def _compile_constraints(self, constraints: List[Constraint], agent: int):
        """Preprocess constraints for fast lookup for this agent only."""
        vtx = defaultdict(set)  # time -> set of forbidden vertices
        edge = defaultdict(set) # time -> set of forbidden edges (u->v)
        max_t = 0
        for c in constraints:
            if c.agent != agent:
                continue
            max_t = max(max_t, c.time)
            if c.ctype == 'vertex':
                vtx[c.time].add(c.loc)
            elif c.ctype == 'edge':
                edge[c.time].add(c.loc)
            else:
                raise ValueError(f"Unknown constraint type: {c.ctype}")
        return vtx, edge, max_t

    def plan(self, agent: int, start: Coord3D, goal: Coord3D, constraints: List[Constraint], max_slack: int = 100) -> Optional[Path]:
        """
        A* in space-time. We treat time as an explicit dimension.
        - Vertex constraint: cannot be at vertex v at time t.
        - Edge constraint: cannot traverse u->v arriving at time t (i.e., move at t-1 -> t).

        Goal condition: reach goal at a time T >= max_constraint_time_for_agent so that
        we can accommodate 'post-arrival' constraints that refer to future times.
        """
        vtx_forbid, edge_forbid, max_ct_time = self._compile_constraints(constraints, agent)

        start_t = 0
        goal_ready_time = max_ct_time  # must be at goal at/after this time to honor future constraints

        # Upper bound for time expansion to keep search finite.
        # Allow waiting if needed due to conflicts/constraints.
        max_time = max(goal_ready_time, manhattan_3d(start, goal)) + max_slack

        # Priority queue entries: (f, g=time, tie, (pos, t))
        openq: List[Tuple[int,int,int,Tuple[Coord3D,int]]] = []
        counter = 0

        # g-values best-known (pos, t) -> g
        best_g: Dict[Tuple[Coord3D,int], int] = {}

        # Parents for path reconstruction: (pos,t) -> (prev_pos, prev_t)
        parent: Dict[Tuple[Coord3D,int], Tuple[Coord3D,int]] = {}

        start_state = (start, start_t)
        h0 = manhattan_3d(start, goal)
        heapq.heappush(openq, (h0, 0, counter, start_state))
        best_g[start_state] = 0

        def is_forbidden(next_pos: Coord3D, curr_pos: Coord3D, arrive_t: int) -> bool:
            # Vertex constraint at time arrive_t
            if next_pos in vtx_forbid.get(arrive_t, ()):
                return True
            # Edge constraint for (curr_pos -> next_pos) at arrival time arrive_t
            if (curr_pos, next_pos) in edge_forbid.get(arrive_t, ()):
                return True
            return False

        while openq:
            f, g, _, (pos, t) = heapq.heappop(openq)

            # If we've already expanded a better g for this state, skip
            if best_g.get((pos, t), float('inf')) < g:
                continue

            # Goal check: at goal and past required time
            if pos == goal and t >= goal_ready_time:
                # Reconstruct path (only positions; time indices are implicit 0..t)
                path: List[Coord3D] = []
                cur = (pos, t)
                while cur in parent:
                    ppos, pt = cur
                    path.append(ppos)
                    cur = parent[cur]
                # add the start state's position
                path.append(start)
                path.reverse()
                return path

            if t >= max_time:
                # do not expand beyond max_time
                continue

            for nxt in neighbors_6(pos, self.grid_size, self.obstacles, allow_wait=self.allow_wait):
                arrive_t = t + 1
                if is_forbidden(nxt, pos, arrive_t):
                    continue
                # Standard A*
                ng = g + 1
                h = manhattan_3d(nxt, goal)
                nf = ng + h
                state = (nxt, arrive_t)
                if ng < best_g.get(state, float('inf')):
                    best_g[state] = ng
                    counter += 1
                    parent[state] = (pos, t)
                    heapq.heappush(openq, (nf, ng, counter, state))

        return None  # No feasible path under constraints within horizon


# ---------------------------- Conflict detection ---------------------------- #

@dataclass
class Conflict:
    a1: int
    a2: int
    time: int
    ctype: str    # 'vertex' or 'edge'
    loc: Tuple    # for vertex: (x,y,z); for edge: ((x1..),(x2..)) where x1 is a1's move and x2 is a2's move


def pad_paths(paths: Dict[int, Path]) -> Dict[int, Path]:
    """Pad all paths to the same length by repeating their final position."""
    if not paths:
        return {}
    T = max(len(p) for p in paths.values())
    padded = {}
    for a, p in paths.items():
        if len(p) < T:
            padded[a] = p + [p[-1]] * (T - len(p))
        else:
            padded[a] = p
    return padded


def detect_conflict(paths: Dict[int, Path]) -> Optional[Conflict]:
    """Return earliest conflict if any (vertex or edge). Times are indices into the padded path (0-based)."""
    padded = pad_paths(paths)
    agents = list(padded.keys())
    if not agents:
        return None
    T = len(next(iter(padded.values())))

    # Vertex conflicts at time t
    for t in range(T):
        for i in range(len(agents)):
            a1 = agents[i]
            p1 = padded[a1]
            for j in range(i+1, len(agents)):
                a2 = agents[j]
                p2 = padded[a2]
                if p1[t] == p2[t]:
                    return Conflict(a1=a1, a2=a2, time=t, ctype='vertex', loc=p1[t])

    # Edge conflicts between t-1 -> t
    for t in range(1, T):
        for i in range(len(agents)):
            a1 = agents[i]
            p1 = padded[a1]
            for j in range(i+1, len(agents)):
                a2 = agents[j]
                p2 = padded[a2]
                if p1[t-1] == p2[t] and p1[t] == p2[t-1]:
                    return Conflict(a1=a1, a2=a2, time=t, ctype='edge', loc=((p1[t-1], p1[t]), (p2[t-1], p2[t])))
    return None


def solution_cost(paths: Dict[int, Path]) -> int:
    return sum(max(0, len(p) - 1) for p in paths.values())


# ---------------------------- High-level CBS ---------------------------- #

@dataclass(order=True)
class HLNode:
    priority: int
    cost: int = field(compare=False)
    constraints: List[Constraint] = field(compare=False, default_factory=list)
    paths: Dict[int, Path] = field(compare=False, default_factory=dict)
    uid: int = field(compare=False, default=0)  # for tie-breaking stability


class CBS3D:
    def __init__(self, grid_size: Tuple[int,int,int], obstacles: Optional[Set[Coord3D]] = None, allow_wait: bool=True):
        self.grid_size = grid_size
        self.obstacles = obstacles or set()
        self.allow_wait = allow_wait
        self.lowlevel = AStar3D(grid_size, self.obstacles, allow_wait=allow_wait)
        self.uid_counter = 0

    def _next_uid(self) -> int:
        self.uid_counter += 1
        return self.uid_counter

    def _replan_agent(self, agent: int, start: Coord3D, goal: Coord3D, constraints: List[Constraint]) -> Optional[Path]:
        return self.lowlevel.plan(agent, start, goal, constraints)

    def solve(self, starts: List[Coord3D], goals: List[Coord3D], max_nodes: int = 100000, return_time: bool = False
              ) -> Union[Optional[Dict[int, Path]], Tuple[Optional[Dict[int, Path]], float]]:

        assert len(starts) == len(goals), "Starts and goals count must match"
        n_agents = len(starts)

        # 计时开始
        t0 = time.perf_counter()

        # Root: plan paths for all agents with no constraints
        constraints: List[Constraint] = []
        root_paths: Dict[int, Path] = {}
        for a in range(n_agents):
            p = self._replan_agent(a, starts[a], goals[a], constraints)
            if p is None:
                dt = time.perf_counter() - t0
                print(f"[CBS] solve time = {dt * 1000:.2f} ms (root infeasible)")
                return (None, dt * 1000.0) if return_time else None
            root_paths[a] = p
        root_cost = solution_cost(root_paths)
        root = HLNode(priority=root_cost, cost=root_cost, constraints=constraints, paths=root_paths,
                      uid=self._next_uid())

        openq: List[HLNode] = []
        heapq.heappush(openq, root)
        expanded = 0

        while openq:
            node = heapq.heappop(openq)
            expanded += 1
            if expanded > max_nodes:
                # Safeguard against pathological cases
                dt = time.perf_counter() - t0
                print(f"[CBS] solve time = {dt * 1000:.2f} ms (exceeded max_nodes={max_nodes})")
                return (None, dt * 1000.0) if return_time else None

            conflict = detect_conflict(node.paths)
            if conflict is None:
                dt = time.perf_counter() - t0
                print(f"[CBS] solve time = {dt * 1000:.2f} ms, expanded={expanded}")
                return (node.paths, dt * 1000.0) if return_time else node.paths

            # Split on the conflict -> two children with additional constraints
            a1, a2, t = conflict.a1, conflict.a2, conflict.time
            if conflict.ctype == 'vertex':
                v = conflict.loc  # type: ignore
                # Child 1: forbid a1 at v at time t
                c1 = Constraint(agent=a1, ctype='vertex', time=t, loc=v)
                # Child 2: forbid a2 at v at time t
                c2 = Constraint(agent=a2, ctype='vertex', time=t, loc=v)

                for agent_to_constrain, new_c in [(a1, c1), (a2, c2)]:
                    child_constraints = node.constraints + [new_c]
                    child_paths = dict(node.paths)
                    # replan only the constrained agent
                    p = self._replan_agent(agent_to_constrain, starts[agent_to_constrain], goals[agent_to_constrain],
                                           child_constraints)
                    if p is None:
                        continue  # infeasible child
                    child_paths[agent_to_constrain] = p
                    cost = solution_cost(child_paths)
                    child = HLNode(priority=cost, cost=cost, constraints=child_constraints, paths=child_paths,
                                   uid=self._next_uid())
                    heapq.heappush(openq, child)

            elif conflict.ctype == 'edge':
                (u1, v1), (u2, v2) = conflict.loc  # type: ignore
                # Child 1: forbid a1 moving u1->v1 at time t
                c1 = Constraint(agent=a1, ctype='edge', time=t, loc=(u1, v1))
                # Child 2: forbid a2 moving u2->v2 at time t
                c2 = Constraint(agent=a2, ctype='edge', time=t, loc=(u2, v2))

                for agent_to_constrain, new_c in [(a1, c1), (a2, c2)]:
                    child_constraints = node.constraints + [new_c]
                    child_paths = dict(node.paths)
                    p = self._replan_agent(agent_to_constrain, starts[agent_to_constrain], goals[agent_to_constrain],
                                           child_constraints)
                    if p is None:
                        continue
                    child_paths[agent_to_constrain] = p
                    cost = solution_cost(child_paths)
                    child = HLNode(priority=cost, cost=cost, constraints=child_constraints, paths=child_paths,
                                   uid=self._next_uid())
                    heapq.heappush(openq, child)
            else:
                raise ValueError("Unknown conflict type")

        # 无解兜底
        dt = time.perf_counter() - t0
        print(f"[CBS] solve time = {dt * 1000:.2f} ms (no solution)")
        return (None, dt * 1000.0) if return_time else None


# ---------------------------- Example usage ---------------------------- #
if __name__ == "__main__":
    # Simple demo: 3 agents in a 5x5x3 grid
    grid = (5, 5, 3)
    obstacles = {
        (2,2,0), (2,2,1), (2,2,2),  # a vertical obstacle column in the middle
        (1,3,1)
    }
    starts = [(0,0,0), (4,4,0), (0,4,2)]
    goals  = [(4,0,2), (0,4,0), (4,4,2)]

    cbs = CBS3D(grid, obstacles)
    result = cbs.solve(starts, goals)
    if result is None:
        print("No solution found.")
    else:
        print("Solution (sum of costs =", solution_cost(result), ")")
        T = max(len(p) for p in result.values())
        # Pad for a clean table
        padded = pad_paths(result)
        for t in range(T):
            row = [f"t={t:02d}"]
            for a in range(len(starts)):
                row.append(f"A{a}:{padded[a][t]}")
            print(" | ".join(row))
