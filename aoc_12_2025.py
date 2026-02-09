from __future__ import annotations
import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

'''
## Problem Overview

This is a 2D shape packing solver for Advent of Code 2025, Day 12.
Problem URL: https://adventofcode.com/2025/day/12

### Task
Given:
- A set of 2D shapes (defined as grids with `#` marking filled cells, and `.` empty ones)
- A list of rectangular regions with specific shape requirements

Find: How many regions can be successfully filled with all required shapes?

### Key Constraints
- Shapes can be rotated (0°, 90°, 180°, 270°) and flipped
- Empty cells are allowed in the final solution (not an exact cover problem)
- Each shape placement gets a unique letter label for visualization

### Input Format

```
0:
###
#.#
###

1:
##.
.##
.#.

...

6x4: 2 0 1 0 0 1
8x3: 0 1 0 2 1 0
...
```

- Shape blocks: Shape index (`N:`) followed by consecutive lines denoting a "shape", consisting of empty (`.`) and occupied (`#`) spaces
- Region specs: `WIDTHxHEIGHT: count0 count1 count2 ...` for each of the shape indices

'''

@dataclass(frozen=True)
class Shape:
    '''
    A 2D representation of a shape
    '''
    cells: List[List[bool]]

    def rotate90(self) -> Shape:
        return Shape(cells=np.rot90(self.cells, k=-1))

    def rotate180(self) -> Shape:
        return Shape(cells=np.rot90(self.cells, k=2))

    def rotate270(self) -> Shape:
        return Shape(cells=np.rot90(self.cells, k=1))

    def flip(self) -> Shape:
        '''
        Mirrors about the Y-axis
        '''
        return Shape(cells=np.fliplr(self.cells))
    
    def __eq__(self, other):
        '''
        Fast __eq__ comparisons
        '''
        if not isinstance(other, Shape):
            return NotImplemented
        return (self.cells.shape == other.cells.shape
                and self.cells.tobytes() == other.cells.tobytes())

    def __hash__(self):
        return hash((self.cells.tobytes(), self.cells.shape))

@dataclass(frozen=True, repr=True)
class FilledRegion:
    '''
    Represents a filled-out region
    '''
    grid: List[List[str]]

    def __repr__(self) -> str:
        result = []
        for row in self.grid:
            result.append(''.join(symbol if symbol else '.' for symbol in row))
        return '\n'.join(result)

    def __str__(self) -> str:
        return self.__repr__()

    def print(self):
        '''Print the filled region'''
        _printv(self.__repr__())

@dataclass(frozen=True, repr=True)
class RegionSpec:
    '''
    Represents a region to fill with the fill requirements
    '''
    length: int
    width: int
    fill_requirement: Dict[int, int]

    def __repr__(self) -> str:
        return f"Region X={self.width} Y={self.length} REQ={self.fill_requirement}"
    
    def __str__(self) -> str:
        return self.__repr__()

    def fill_region(self, shapes_dict: Dict[int, Shape]) -> Optional[FilledRegion]:
        '''
        Attempts to fill a region with shapes according to the fill requirements (shape index -> quantity needed).
        We also pass in a shapes dictionary that maps the shape types to the index
        If successful, returns a 2D representation of the filled region using unique characters assigned to each shape.
        If fails to fill all of the shapes, it returns None

        ### Algorithm
        The solution uses **recursive backtracking with cell-skipping**:

        1. Find the first empty cell (top-left scan)
        2. For each shape type with remaining count > 0:
        - Try all unique orientations of that shape
        - If it fits, place it and recurse
        - If recursion succeeds, return success
        - Otherwise, backtrack and try next orientation
        3. Allow skipping the cell (mark as empty `.`)
        4. Return success if all shapes placed, failure otherwise

        ### Key Optimizations

        1. **Orientation caching**: Precompute all unique orientations per shape using `Shape.__eq__`/`__hash__` to deduplicate symmetric rotations/flips
        2. **Early area check**: Reject if total shape area > region area
        3. **Shape deduplication**: Use `dict.fromkeys()` to filter duplicate orientations efficiently

        ### Complexity Analysis

        **Runtime: O(n · m · s · o · b^(w·h))** (worst case, exponential)

        Where:
        - n = number of regions to process
        - m = number of shape types
        - s = total shapes to place per region
        - o = orientations per shape (≤ 8)
        - w = region width
        - h = region height
        - b = branching factor ≈ (m · o + 1) for each cell

        At each of w·h cells, we try up to m·o shape placements plus 1 skip option,
        leading to an exponential search tree of size O(b^(w·h)). However, pruning
        (early termination, area checks, orientation deduplication) reduces this
        significantly in practice. The problem is fundamentally NP-hard (2D bin packing).

        **Space: O(w · h + m · o)**
        - O(w · h) for grid storage
        - O(m · o) for orientation cache
        - O(w · h) for recursion stack (worst case: one call per cell)
        '''
        # Build remaining counts per shape type
        remaining_counts = dict(self.fill_requirement)
        total_shapes = sum(remaining_counts.values())

        # Early check: total shape area must fit within region area
        total_shape_area = 0
        for shape_idx, count in remaining_counts.items():
            if count > 0:
                total_shape_area += int(np.sum(shapes_dict[shape_idx].cells)) * count
        region_area = self.width * self.length
        if total_shape_area > region_area:
            return None

        # Precompute unique orientations for each shape type
        orientation_cache = RegionSpec._build_orientation_cache(shapes_dict)

        grid = [[None for _ in range(self.width)] for _ in range(self.length)]
        letter_idx = [0]

        if RegionSpec._backtrack(grid, self.length, self.width, remaining_counts,
                             orientation_cache, letter_idx, total_shapes):
            return FilledRegion(grid=grid)
        else:
            return None

    @staticmethod
    def _build_orientation_cache(shapes_dict: Dict[int, Shape]) -> Dict[int, List[Shape]]:
        orientation_cache = {}
        for shape_idx, shape_obj in shapes_dict.items():
            flipped = shape_obj.flip()
            orientations = [
                shape_obj,
                shape_obj.rotate90(),
                shape_obj.rotate180(),
                shape_obj.rotate270(),
                flipped,
                flipped.rotate90(),
                flipped.rotate180(),
                flipped.rotate270(),
            ]
            orientation_cache[shape_idx] = list(dict.fromkeys(orientations))
        return orientation_cache

    @staticmethod
    def _get_next_letter(letter_idx: List[int]) -> str:
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        letter = letters[letter_idx[0] % len(letters)]
        letter_idx[0] += 1
        return letter

    @staticmethod
    def _can_place(grid: List[List], length: int, width: int,
                   shape: Shape, row: int, col: int) -> bool:
        shape_h, shape_w = shape.cells.shape
        if row + shape_h > length or col + shape_w > width:
            return False
        for i in range(shape_h):
            for j in range(shape_w):
                if shape.cells[i, j] and grid[row + i][col + j] is not None:
                    return False
        return True

    @staticmethod
    def _place_shape(grid: List[List], shape: Shape, row: int, col: int, letter: str):
        shape_h, shape_w = shape.cells.shape
        for i in range(shape_h):
            for j in range(shape_w):
                if shape.cells[i, j]:
                    grid[row + i][col + j] = letter

    @staticmethod
    def _remove_shape(grid: List[List], shape: Shape, row: int, col: int):
        shape_h, shape_w = shape.cells.shape
        for i in range(shape_h):
            for j in range(shape_w):
                if shape.cells[i, j]:
                    grid[row + i][col + j] = None

    @staticmethod
    def _find_first_empty(grid: List[List], length: int, width: int) -> Optional[Tuple[int, int]]:
        for r in range(length):
            for c in range(width):
                if grid[r][c] is None:
                    return (r, c)
        return None

    @staticmethod
    def _backtrack(grid: List[List], length: int, width: int,
                   remaining_counts: Dict[int, int],
                   orientation_cache: Dict[int, List[Shape]],
                   letter_idx: List[int],
                   shapes_remaining: int) -> bool:
        '''Recursive backtracking: find first empty cell, try any shape or skip'''
        EMPTY = '.'

        pos = RegionSpec._find_first_empty(grid, length, width)
        if pos is None:
            return shapes_remaining == 0

        if shapes_remaining == 0:
            return True

        row, col = pos

        # Try placing each shape type that still has remaining count
        for shape_idx in remaining_counts:
            if remaining_counts[shape_idx] <= 0:
                continue

            for shape in orientation_cache[shape_idx]:
                if RegionSpec._can_place(grid, length, width, shape, row, col):
                    letter = RegionSpec._get_next_letter(letter_idx)
                    RegionSpec._place_shape(grid, shape, row, col, letter)
                    remaining_counts[shape_idx] -= 1

                    if RegionSpec._backtrack(grid, length, width, remaining_counts,
                                         orientation_cache, letter_idx,
                                         shapes_remaining - 1):
                        return True

                    # Backtrack
                    RegionSpec._remove_shape(grid, shape, row, col)
                    remaining_counts[shape_idx] += 1
                    letter_idx[0] -= 1

        # Option: skip this cell (leave it empty)
        grid[row][col] = EMPTY
        if RegionSpec._backtrack(grid, length, width, remaining_counts,
                             orientation_cache, letter_idx, shapes_remaining):
            result = True
        else:
            result = False
        grid[row][col] = None
        return result


    class InputParser:
        @staticmethod
        def _parse_shape(lines: List[str]) -> List[List[bool]]:
            return [[c == '#' for c in line] for line in lines]

        @staticmethod
        def _parse_shapes(blocks: List[str]) -> Dict[int, Shape]:
            shapes: Dict[int, Shape] = {}
            for block in blocks:
                block_lines = block.strip().split('\n')
                header = block_lines[0]
                # Skip blocks that don't match the shape header format (e.g. "0:")
                if not header[0].isdigit() or not header.endswith(':'):
                    continue
                shape_idx = int(header[:-1])
                shape_rows = RegionSpec.InputParser._parse_shape(block_lines[1:])
                shapes[shape_idx] = Shape(
                    cells=np.array(shape_rows, dtype=bool)
                )
            return shapes

        @staticmethod
        def _parse_region(line: str) -> RegionSpec:
            dims, counts = line.split(': ')
            w, l = dims.split('x')
            reqs = list(map(int, counts.split()))
            fill_requirement = {idx: count for idx, count in enumerate(reqs)}
            return RegionSpec(width=int(w), length=int(l), fill_requirement=fill_requirement)

        @staticmethod
        def parse_input(input_path: str) -> Tuple[Dict[int, Shape], List[RegionSpec]]:
            '''
            Parses the raw input file as specified in the instructions, and returns:
            (shape_idx -> shape, [specification to fill out regions])
            '''
            with open(input_path, 'r') as f:
                text = f.read()

            # Shapes and regions are separated by a double newline after the last shape
            blocks = text.split('\n\n')

            # Shape blocks come first (header like "0:", "1:", etc.)
            shape_blocks = [b for b in blocks if b.strip() and b.strip()[0].isdigit() and b.strip().split('\n')[0].endswith(':')]
            shapes_dict = RegionSpec.InputParser._parse_shapes(shape_blocks)

            # Region lines are the remaining blocks
            region_blocks = [b for b in blocks if b.strip() and 'x' in b.strip().split('\n')[0] and ': ' in b.strip().split('\n')[0]]
            regions = []
            for block in region_blocks:
                for line in block.strip().split('\n'):
                    if line.strip():
                        regions.append(RegionSpec.InputParser._parse_region(line.strip()))

            return shapes_dict, regions

__is_verbose__ = False

def _printv(*args, **kwargs):
    '''
    verbose print
    '''
    if __is_verbose__:
        print(*args, **kwargs)

def main():
    global __is_verbose__
    input_file_path = sys.argv[1]
    __is_verbose__ = len(sys.argv) >= 3 and sys.argv[2] == '--verbose'

    shapes_dict, regions = RegionSpec.InputParser.parse_input(input_file_path)

    _printv(f"Loaded {len(shapes_dict)} shapes and {len(regions)} regions")

    success_count = 0
    for i, region in enumerate(regions):
        _printv(f"Region {i+1}/{len(regions)}: {region.width}x{region.length}...", end=' ')
        filled_region = region.fill_region(shapes_dict)
        if filled_region:
            success_count += 1
            _printv()
            filled_region.print()
        else:
            _printv("N")

    print(success_count)

if __name__ == "__main__":
    main()