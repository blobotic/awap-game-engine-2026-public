"""
Resource Management System

Handles resource locks, contention, and reservation to prevent
two bots from chasing the same thing.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Tuple, List, Any
from enum import Enum


class ResourceType(Enum):
    COOKER = "cooker"
    COUNTER = "counter"
    PLATE = "plate"
    FOOD = "food"
    PAN = "pan"
    TILE = "tile"  # For movement reservation


@dataclass
class Resource:
    """A lockable resource in the game."""
    id: str                          # Unique identifier, e.g., "cooker_4_2"
    type: ResourceType
    location: Tuple[int, int]
    locked_by: Optional[str] = None  # milestone_id or "bot_<id>"
    lock_turn: Optional[int] = None  # When lock was acquired
    item_id: Optional[int] = None    # For tracking specific items


class ResourceManager:
    """
    Manages resource locks and reservations.

    Lock semantics:
    - A resource can only be locked by one entity at a time
    - Locks have TTL to prevent deadlock from stuck bots
    - Tile reservations prevent collision
    """

    DEFAULT_LOCK_TTL = 50  # Release lock if no progress for this many turns

    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.tile_reservations: Dict[Tuple[int, int], int] = {}  # pos -> bot_id
        self.bot_assignments: Dict[int, str] = {}  # bot_id -> current milestone_id

    def register_resource(
        self,
        resource_type: ResourceType,
        location: Tuple[int, int],
        item_id: Optional[int] = None
    ) -> str:
        """Register a resource and return its ID."""
        resource_id = f"{resource_type.value}_{location[0]}_{location[1]}"
        if item_id is not None:
            resource_id += f"_item{item_id}"

        if resource_id not in self.resources:
            self.resources[resource_id] = Resource(
                id=resource_id,
                type=resource_type,
                location=location,
                item_id=item_id
            )
        return resource_id

    def try_lock(
        self,
        resource_id: str,
        requester: str,  # milestone_id or "bot_<id>"
        current_turn: int
    ) -> bool:
        """
        Attempt to lock a resource.
        Returns True if lock acquired or already held by requester.
        """
        if resource_id not in self.resources:
            return False

        resource = self.resources[resource_id]

        # Already locked by requester
        if resource.locked_by == requester:
            return True

        # Locked by someone else
        if resource.locked_by is not None:
            # Check TTL
            if resource.lock_turn and (current_turn - resource.lock_turn) > self.DEFAULT_LOCK_TTL:
                # TTL expired, force release
                self.release(resource_id)
            else:
                return False

        # Acquire lock
        resource.locked_by = requester
        resource.lock_turn = current_turn
        return True

    def release(self, resource_id: str) -> bool:
        """Release a resource lock."""
        if resource_id not in self.resources:
            return False

        resource = self.resources[resource_id]
        resource.locked_by = None
        resource.lock_turn = None
        return True

    def release_all_by(self, requester: str):
        """Release all resources locked by a specific requester."""
        for resource in self.resources.values():
            if resource.locked_by == requester:
                resource.locked_by = None
                resource.lock_turn = None

    def is_locked(self, resource_id: str) -> bool:
        """Check if a resource is locked."""
        if resource_id not in self.resources:
            return False
        return self.resources[resource_id].locked_by is not None

    def get_lock_holder(self, resource_id: str) -> Optional[str]:
        """Get who holds a lock."""
        if resource_id not in self.resources:
            return None
        return self.resources[resource_id].locked_by

    def get_available(self, resource_type: ResourceType) -> List[Resource]:
        """Get all unlocked resources of a type."""
        return [
            r for r in self.resources.values()
            if r.type == resource_type and r.locked_by is None
        ]

    def get_all_of_type(self, resource_type: ResourceType) -> List[Resource]:
        """Get all resources of a type (locked or not)."""
        return [r for r in self.resources.values() if r.type == resource_type]

    # ==========================================
    # Tile Reservations (for collision avoidance)
    # ==========================================

    def reserve_tile(self, pos: Tuple[int, int], bot_id: int) -> bool:
        """
        Reserve a tile for a bot's next move.
        Returns False if already reserved by another bot.
        """
        if pos in self.tile_reservations:
            return self.tile_reservations[pos] == bot_id
        self.tile_reservations[pos] = bot_id
        return True

    def release_tile(self, pos: Tuple[int, int], bot_id: int):
        """Release a tile reservation."""
        if pos in self.tile_reservations and self.tile_reservations[pos] == bot_id:
            del self.tile_reservations[pos]

    def clear_tile_reservations(self):
        """Clear all tile reservations (call at start of each turn)."""
        self.tile_reservations.clear()

    def is_tile_reserved(self, pos: Tuple[int, int], exclude_bot: Optional[int] = None) -> bool:
        """Check if a tile is reserved (optionally excluding a specific bot)."""
        if pos not in self.tile_reservations:
            return False
        if exclude_bot is not None and self.tile_reservations[pos] == exclude_bot:
            return False
        return True

    # ==========================================
    # Bot Assignment Tracking
    # ==========================================

    def assign_bot(self, bot_id: int, milestone_id: str):
        """Assign a bot to a milestone."""
        self.bot_assignments[bot_id] = milestone_id

    def get_bot_assignment(self, bot_id: int) -> Optional[str]:
        """Get a bot's current assignment."""
        return self.bot_assignments.get(bot_id)

    def clear_bot_assignment(self, bot_id: int):
        """Clear a bot's assignment."""
        if bot_id in self.bot_assignments:
            del self.bot_assignments[bot_id]

    def is_bot_assigned(self, bot_id: int) -> bool:
        """Check if a bot has an assignment."""
        return bot_id in self.bot_assignments


@dataclass
class KitchenPlan:
    """
    Pre-computed optimal station assignments.
    Binds abstract needs to concrete locations.
    """
    # Primary work stations
    cooker: Optional[Tuple[int, int]] = None
    assembly_counter: Optional[Tuple[int, int]] = None

    # Secondary stations (may be same as assembly)
    chop_counter: Optional[Tuple[int, int]] = None

    # Fixed stations
    shop: Optional[Tuple[int, int]] = None
    submit: Optional[Tuple[int, int]] = None
    sink: Optional[Tuple[int, int]] = None
    sinktable: Optional[Tuple[int, int]] = None
    trash: Optional[Tuple[int, int]] = None

    # All available stations (for fallback)
    all_cookers: List[Tuple[int, int]] = field(default_factory=list)
    all_counters: List[Tuple[int, int]] = field(default_factory=list)


def compute_kitchen_plan(rc, team=None) -> KitchenPlan:
    """
    Compute optimal station assignments based on map layout.

    Strategy: Minimize work triangle (shop -> cooker -> assembly -> submit)
    """
    if team is None:
        team = rc.get_team()

    game_map = rc.get_map(team)
    plan = KitchenPlan()

    # Collect all stations
    cookers = []
    counters = []
    shops = []
    submits = []
    sinks = []
    sinktables = []
    trashes = []

    for x in range(game_map.width):
        for y in range(game_map.height):
            tile = rc.get_tile(team, x, y)
            if tile is None:
                continue

            tile_name = getattr(tile, 'tile_name', None)

            if tile_name == 'COOKER':
                cookers.append((x, y))
            elif tile_name == 'COUNTER':
                counters.append((x, y))
            elif tile_name == 'SHOP':
                shops.append((x, y))
            elif tile_name == 'SUBMIT':
                submits.append((x, y))
            elif tile_name == 'SINK':
                sinks.append((x, y))
            elif tile_name == 'SINKTABLE':
                sinktables.append((x, y))
            elif tile_name == 'TRASH':
                trashes.append((x, y))

    plan.all_cookers = cookers
    plan.all_counters = counters

    # Set fixed stations (use first found)
    if shops:
        plan.shop = shops[0]
    if submits:
        plan.submit = submits[0]
    if sinks:
        plan.sink = sinks[0]
    if sinktables:
        plan.sinktable = sinktables[0]
    if trashes:
        plan.trash = trashes[0]

    # Choose optimal cooker (minimize distance to shop and submit)
    if cookers and plan.shop and plan.submit:
        def cooker_score(c):
            return chebyshev(c, plan.shop) + chebyshev(c, plan.submit)
        plan.cooker = min(cookers, key=cooker_score)
    elif cookers:
        plan.cooker = cookers[0]

    # Choose optimal assembly counter (minimize distance to cooker and submit)
    if counters and plan.cooker and plan.submit:
        def counter_score(c):
            return chebyshev(c, plan.cooker) + chebyshev(c, plan.submit)
        plan.assembly_counter = min(counters, key=counter_score)

        # Choose chop counter (can be same or near cooker)
        if len(counters) > 1:
            remaining = [c for c in counters if c != plan.assembly_counter]
            plan.chop_counter = min(remaining, key=lambda c: chebyshev(c, plan.cooker))
        else:
            plan.chop_counter = plan.assembly_counter
    elif counters:
        plan.assembly_counter = counters[0]
        plan.chop_counter = counters[0]

    return plan


def chebyshev(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Chebyshev (chess king) distance."""
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
