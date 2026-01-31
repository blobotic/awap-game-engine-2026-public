"""
BFS Pathfinding Utilities for Carnegie Cookoff

This module provides pathfinding utilities using BFS (Breadth-First Search).
All functions are designed to be copy-pasted into a single submission file.

Movement is 8-directional (Chebyshev distance 1, like a chess king).
"""

from collections import deque
from typing import Tuple, Optional, List, Callable, Any

# Type aliases for clarity
Position = Tuple[int, int]
Direction = Tuple[int, int]
Path = List[Direction]

# All 8 possible movement directions (Chebyshev neighbors)
DIRECTIONS: List[Direction] = [
    (0, 1), (0, -1), (1, 0), (-1, 0),  # Cardinal
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
]


def get_bot_position(rc, bot_id: int) -> Optional[Position]:
    """Get the current position of a bot."""
    state = rc.get_bot_state(bot_id)
    if state is None:
        return None
    return (state['x'], state['y'])


def get_bot_map_team(rc, bot_id: int):
    """Get which map the bot is currently on (handles sabotage switching)."""
    state = rc.get_bot_state(bot_id)
    if state is None:
        return rc.get_team()
    # map_team tells us which map the bot is actually on
    map_team_name = state.get('map_team', state['team'])
    # Convert string back to Team enum if needed
    from game_constants import Team
    if isinstance(map_team_name, str):
        return Team.RED if map_team_name == 'RED' else Team.BLUE
    return map_team_name


def get_occupancy_set(rc, map_team) -> set:
    """
    Get set of positions occupied by bots on the given map.
    Excludes our own bots' current positions from blocking consideration
    since we're planning paths for them.
    """
    occupied = set()
    # Check all bots from both teams
    for team in [rc.get_team(), rc.get_enemy_team()]:
        for bid in rc.get_team_bot_ids(team):
            state = rc.get_bot_state(bid)
            if state is None:
                continue
            # Check if this bot is on the same map
            bot_map = state.get('map_team', state['team'])
            from game_constants import Team
            if isinstance(bot_map, str):
                bot_map = Team.RED if bot_map == 'RED' else Team.BLUE
            if bot_map == map_team:
                occupied.add((state['x'], state['y']))
    return occupied


def is_walkable(rc, map_team, x: int, y: int, game_map=None) -> bool:
    """Check if a tile is walkable."""
    if game_map is None:
        game_map = rc.get_map(map_team)
    if not (0 <= x < game_map.width and 0 <= y < game_map.height):
        return False
    return game_map.is_tile_walkable(x, y)


def bfs_path(
    rc,
    bot_id: int,
    goal_fn: Callable[[int, int, Any], bool],
    avoid_bots: bool = True,
    exclude_bot_ids: Optional[List[int]] = None
) -> Optional[Path]:
    """
    BFS to find shortest path from bot's position to any tile satisfying goal_fn.

    Args:
        rc: RobotController instance
        bot_id: ID of the bot to pathfind for
        goal_fn: Function (x, y, tile) -> bool that returns True for goal tiles
        avoid_bots: If True, treat tiles occupied by other bots as blocked
        exclude_bot_ids: Bot IDs to exclude from occupancy check (e.g., self)

    Returns:
        List of (dx, dy) moves to reach the goal, or None if no path exists.
        Empty list [] means already at goal.
    """
    start = get_bot_position(rc, bot_id)
    if start is None:
        return None

    map_team = get_bot_map_team(rc, bot_id)
    game_map = rc.get_map(map_team)

    # Get occupied positions
    if avoid_bots:
        occupied = get_occupancy_set(rc, map_team)
        # Don't consider our own position as blocked
        occupied.discard(start)
        # Also exclude specified bot IDs
        if exclude_bot_ids:
            for bid in exclude_bot_ids:
                state = rc.get_bot_state(bid)
                if state:
                    occupied.discard((state['x'], state['y']))
    else:
        occupied = set()

    # BFS
    queue = deque([(start, [])])  # (position, path_so_far)
    visited = {start}

    while queue:
        (cx, cy), path = queue.popleft()

        # Check if current position is goal
        tile = rc.get_tile(map_team, cx, cy)
        if goal_fn(cx, cy, tile):
            return path

        # Explore neighbors
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy

            if (nx, ny) in visited:
                continue

            if not is_walkable(rc, map_team, nx, ny, game_map):
                continue

            if (nx, ny) in occupied:
                continue

            visited.add((nx, ny))
            queue.append(((nx, ny), path + [(dx, dy)]))

    return None  # No path found


def bfs_path_to_position(
    rc,
    bot_id: int,
    target_x: int,
    target_y: int,
    avoid_bots: bool = True
) -> Optional[Path]:
    """
    Find path to a specific walkable position.

    Returns:
        Path to the target, or None if unreachable.
    """
    def goal_fn(x, y, tile):
        return x == target_x and y == target_y

    return bfs_path(rc, bot_id, goal_fn, avoid_bots)


def bfs_path_to_adjacent(
    rc,
    bot_id: int,
    target_x: int,
    target_y: int,
    avoid_bots: bool = True
) -> Optional[Path]:
    """
    Find path to any walkable tile adjacent to target (Chebyshev distance 1).
    Use this when the target itself is not walkable (e.g., Counter, Cooker, Shop).

    Returns:
        Path to an adjacent tile, or None if unreachable.
    """
    def goal_fn(x, y, tile):
        return max(abs(x - target_x), abs(y - target_y)) == 1

    # If already adjacent, return empty path
    start = get_bot_position(rc, bot_id)
    if start and max(abs(start[0] - target_x), abs(start[1] - target_y)) <= 1:
        return []

    return bfs_path(rc, bot_id, goal_fn, avoid_bots)


def bfs_find_nearest(
    rc,
    bot_id: int,
    tile_predicate: Callable[[int, int, Any], bool],
    avoid_bots: bool = True
) -> Optional[Tuple[Position, Path]]:
    """
    Find the nearest tile matching a predicate and return both its position and path.

    Args:
        rc: RobotController instance
        bot_id: ID of the bot
        tile_predicate: Function (x, y, tile) -> bool
        avoid_bots: Whether to avoid other bots when pathfinding

    Returns:
        Tuple of (target_position, path_to_adjacent) or None if not found.
        The path goes to a tile adjacent to the target (since targets are often non-walkable).
    """
    start = get_bot_position(rc, bot_id)
    if start is None:
        return None

    map_team = get_bot_map_team(rc, bot_id)
    game_map = rc.get_map(map_team)

    # Get occupied positions
    if avoid_bots:
        occupied = get_occupancy_set(rc, map_team)
        occupied.discard(start)
    else:
        occupied = set()

    # BFS to find nearest matching tile
    # We search for walkable tiles adjacent to matching tiles
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (cx, cy), path = queue.popleft()

        # Check all neighbors for matching tiles (including non-walkable ones)
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < game_map.width and 0 <= ny < game_map.height):
                continue

            tile = rc.get_tile(map_team, nx, ny)
            if tile_predicate(nx, ny, tile):
                # Found a matching tile! Current position is adjacent to it.
                return ((nx, ny), path)

        # Continue BFS to walkable neighbors
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy

            if (nx, ny) in visited:
                continue

            if not is_walkable(rc, map_team, nx, ny, game_map):
                continue

            if (nx, ny) in occupied:
                continue

            visited.add((nx, ny))
            queue.append(((nx, ny), path + [(dx, dy)]))

    return None


def bfs_find_tile_by_name(
    rc,
    bot_id: int,
    tile_name: str,
    avoid_bots: bool = True
) -> Optional[Tuple[Position, Path]]:
    """
    Convenience function to find nearest tile by its tile_name attribute.

    Common tile names: "COUNTER", "COOKER", "SHOP", "SINK", "SINKTABLE",
                       "SUBMIT", "TRASH", "BOX"

    Returns:
        Tuple of (target_position, path_to_adjacent) or None if not found.
    """
    def predicate(x, y, tile):
        return tile is not None and getattr(tile, 'tile_name', None) == tile_name

    return bfs_find_nearest(rc, bot_id, predicate, avoid_bots)


def get_next_move(path: Optional[Path]) -> Optional[Direction]:
    """
    Extract the next move from a path.

    Returns:
        (dx, dy) for the next move, or None if path is empty/None.
    """
    if not path:
        return None
    return path[0]


def execute_next_move(rc, bot_id: int, path: Optional[Path]) -> bool:
    """
    Execute the next move in a path.

    Returns:
        True if move was executed (or already at destination), False otherwise.
    """
    if path is None:
        return False
    if len(path) == 0:
        return True  # Already at destination

    dx, dy = path[0]
    return rc.move(bot_id, dx, dy)


def is_adjacent_to(pos1: Position, pos2: Position) -> bool:
    """Check if two positions are adjacent (Chebyshev distance <= 1)."""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= 1


def chebyshev_distance(pos1: Position, pos2: Position) -> int:
    """Calculate Chebyshev (chess king) distance between two positions."""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
