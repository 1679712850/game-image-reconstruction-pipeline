"""Conservative dynamic bbox index; query order is the original insertion order."""
from collections import defaultdict


class CandidateIndex:
    def __init__(self, cell_size=256, max_cells=256):
        self.cell_size, self.max_cells = cell_size, max_cells
        self.cells = defaultdict(set)
        self.large = defaultdict(set)
        self.by_category = defaultdict(set)
        self.locations = {}

    def _cells(self, bbox):
        x, y, right, bottom = bbox
        x0, y0 = x // self.cell_size, y // self.cell_size
        x1, y1 = (right - 1) // self.cell_size, (bottom - 1) // self.cell_size
        if (x1-x0+1)*(y1-y0+1) > self.max_cells:
            return None
        return [(i, j) for i in range(x0, x1+1) for j in range(y0, y1+1)]

    def update(self, index, candidate):
        if index in self.locations:
            category, cells = self.locations.pop(index)
            self.by_category[category].discard(index)
            if cells is None:
                self.large[category].discard(index)
            else:
                for cell in cells:
                    key = (category, *cell)
                    self.cells[key].discard(index)
                    if not self.cells[key]:
                        del self.cells[key]
        category = candidate.category
        cells = self._cells(candidate.bbox)
        self.locations[index] = (category, cells)
        self.by_category[category].add(index)
        if cells is None:
            self.large[category].add(index)
        else:
            for cell in cells:
                self.cells[(category, *cell)].add(index)

    def query(self, candidate):
        cells = self._cells(candidate.bbox)
        if cells is None:
            return sorted(self.by_category.get(candidate.category, ()))
        matches = set(self.large.get(candidate.category, ()))
        for cell in cells:
            matches.update(self.cells.get((candidate.category, *cell), ()))
        return sorted(matches)
