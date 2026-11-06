"""Small, rule-free square-grid operations for the village."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

from .rules import GROUND_BY_CODE, Frame

Cell = tuple[int, int]
Point = tuple[float, float]


class Grid:
    """Immutable ground rows indexed south first as ``rows[y][x]``.

    The southwest cell is (0, 0), so increasing y follows the map north rather
    than the usual screen-coordinate convention. Each row is one string, which is
    also the shape the observation contract publishes, so it is shared rather than
    rebuilt for every reader.
    """

    def __init__(self, frame: Frame, rows: Iterable[Iterable[str]]) -> None:
        frozen = tuple("".join(row) for row in rows)
        if len(frozen) != frame.cells_y or any(len(row) != frame.cells_x for row in frozen):
            raise ValueError("grid rows do not fit the frame")
        unknown = sorted({code for row in frozen for code in row} - set(GROUND_BY_CODE))
        if unknown:
            raise ValueError(f"grid rows use ground codes the rules do not define: {unknown}")
        self.frame = frame
        self.rows = frozen

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Grid) and self.frame == other.frame and self.rows == other.rows

    def __hash__(self) -> int:
        return hash((self.frame, self.rows))

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.frame.cells_x and 0 <= y < self.frame.cells_y

    def contains_point(self, point: Point) -> bool:
        x, y = point
        return 0.0 <= x < self.frame.width and 0.0 <= y < self.frame.height

    def cell_at(self, point: Point) -> Cell | None:
        if not self.contains_point(point):
            return None
        x, y = point
        return int(x // self.frame.cell_size), int(y // self.frame.cell_size)

    def center(self, cell: Cell) -> Point:
        if not self.in_bounds(cell):
            raise ValueError("cell is outside the grid")
        x, y = cell
        size = self.frame.cell_size
        return (x + 0.5) * size, (y + 0.5) * size

    def value_at(self, cell: Cell) -> str:
        if not self.in_bounds(cell):
            raise ValueError("cell is outside the grid")
        x, y = cell
        return self.rows[y][x]

    def neighbours(self, cell: Cell) -> tuple[Cell, ...]:
        x, y = cell
        return tuple(
            candidate
            for candidate in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1))
            if self.in_bounds(candidate)
        )

    def flood(self, start: Cell, allowed: Callable[[Cell], bool]) -> frozenset[Cell]:
        if not self.in_bounds(start) or not allowed(start):
            return frozenset()
        found = {start}
        pending = deque((start,))
        while pending:
            cell = pending.popleft()
            for neighbour in self.neighbours(cell):
                if neighbour not in found and allowed(neighbour):
                    found.add(neighbour)
                    pending.append(neighbour)
        return frozenset(found)

    def supercover(self, start: Point, end: Point) -> tuple[Cell, ...]:
        """Return every crossed cell, including both endpoint cells when in bounds."""
        first = self.cell_at(start)
        last = self.cell_at(end)
        if first is None or last is None:
            return ()
        if first == last:
            return (first,)
        size = self.frame.cell_size
        x0, y0 = start[0] / size, start[1] / size
        x1, y1 = end[0] / size, end[1] / size
        cell_x, cell_y = first
        target_x, target_y = last
        dx, dy = x1 - x0, y1 - y0
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0
        t_delta_x = abs(1 / dx) if dx else float("inf")
        t_delta_y = abs(1 / dy) if dy else float("inf")
        t_max_x = ((cell_x + (step_x > 0)) - x0) / dx if dx else float("inf")
        t_max_y = ((cell_y + (step_y > 0)) - y0) / dy if dy else float("inf")
        cells: list[Cell] = [(cell_x, cell_y)]
        # Each crossing moves at least one coordinate toward the target. This bound turns
        # malformed floating-point edge cases into a visible failure instead of a live match hang.
        limit = 2 * (abs(target_x - cell_x) + abs(target_y - cell_y) + 1)
        for _ in range(limit):
            if (cell_x, cell_y) == (target_x, target_y):
                return tuple(cells)
            # An endpoint on a cell edge belongs to the positive-side cell under
            # ``cell_at``. Stop at t=1 rather than crossing past that endpoint.
            if min(t_max_x, t_max_y) >= 1 - 1e-12:
                if last not in cells:
                    cells.append(last)
                return tuple(cells)
            if abs(t_max_x - t_max_y) < 1e-12:
                # A corner touches both adjacent squares. Including them avoids sight leaking
                # through a diagonal wall corner.
                for candidate in ((cell_x + step_x, cell_y), (cell_x, cell_y + step_y)):
                    if self.in_bounds(candidate) and candidate not in cells:
                        cells.append(candidate)
                # A negative-direction integer endpoint can be one of these touched
                # side cells even though the diagonal traversal cell continues past it.
                if last in cells:
                    return tuple(cells)
                cell_x += step_x
                cell_y += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
            elif t_max_x < t_max_y:
                cell_x += step_x
                t_max_x += t_delta_x
            else:
                cell_y += step_y
                t_max_y += t_delta_y
            if (cell_x, cell_y) not in cells:
                cells.append((cell_x, cell_y))
            if last in cells:
                return tuple(cells)
        raise ValueError(f"supercover did not reach {last} from {first}")
