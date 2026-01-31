"""
High-Level Planner - Milestone management with station binding.

Handles:
- Order decomposition into milestones
- Station binding (cooker, counter, etc.)
- Dependency management
- Progress tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum


class MilestoneStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"  # Waiting for external event (cooking)
    COMPLETE = "complete"


# Ingredient properties
INGREDIENT_PROPS = {
    'EGG': {'can_chop': False, 'can_cook': True},
    'ONIONS': {'can_chop': True, 'can_cook': False},
    'MEAT': {'can_chop': True, 'can_cook': True},
    'NOODLES': {'can_chop': False, 'can_cook': False},
    'SAUCE': {'can_chop': False, 'can_cook': False},
}

INGREDIENT_COSTS = {
    'EGG': 20, 'ONIONS': 30, 'MEAT': 80, 'NOODLES': 40, 'SAUCE': 10,
}


@dataclass
class BoundMilestone:
    """
    A milestone with concrete station bindings.

    Key improvement: stations are bound at planning time, not dynamically.
    """
    id: str
    description: str

    # Dependencies
    deps: List[str] = field(default_factory=list)

    # Station bindings (concrete locations)
    bound: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Required resources (for locking)
    resources: Set[str] = field(default_factory=set)

    # Status
    status: MilestoneStatus = MilestoneStatus.PENDING
    assigned_bot: Optional[int] = None

    # Context
    ingredient: Optional[str] = None

    # For cooking milestones
    cook_started_turn: Optional[int] = None

    def is_ready(self, completed: Set[str]) -> bool:
        """Check if dependencies are satisfied."""
        return all(d in completed for d in self.deps)


@dataclass
class OrderPlan:
    """
    Complete plan for fulfilling an order.

    Contains bound milestones and tracks overall progress.
    """
    order_id: int
    order: Dict[str, Any]
    milestones: Dict[str, BoundMilestone]
    completed: Set[str] = field(default_factory=set)

    # Station bindings (shared across milestones)
    cooker_pos: Optional[Tuple[int, int]] = None
    assembly_counter_pos: Optional[Tuple[int, int]] = None
    chop_counter_pos: Optional[Tuple[int, int]] = None
    plate_id: Optional[str] = None  # Track specific plate instance

    def mark_complete(self, milestone_id: str):
        """Mark a milestone as complete."""
        if milestone_id in self.milestones:
            self.milestones[milestone_id].status = MilestoneStatus.COMPLETE
            self.completed.add(milestone_id)

    def is_complete(self) -> bool:
        """Check if order is fully complete."""
        return 'submitted' in self.completed

    def get_ready_milestones(self) -> List[BoundMilestone]:
        """Get milestones ready to work on."""
        ready = []
        for m in self.milestones.values():
            if m.status == MilestoneStatus.COMPLETE:
                continue
            if m.is_ready(self.completed):
                if m.status == MilestoneStatus.PENDING:
                    m.status = MilestoneStatus.READY
                if m.status == MilestoneStatus.READY:
                    ready.append(m)
        return ready

    def get_unassigned_ready(self) -> List[BoundMilestone]:
        """Get ready milestones not assigned to any bot."""
        return [m for m in self.get_ready_milestones() if m.assigned_bot is None]


def build_order_plan(order: Dict[str, Any], kitchen_plan) -> OrderPlan:
    """
    Build a complete plan for an order with bound stations.

    Key improvements:
    1. Cooking milestones split into start/wait/take
    2. Plate dependency only on plating milestone, not cooking
    3. Stations bound upfront from kitchen_plan
    """
    milestones: Dict[str, BoundMilestone] = {}
    ingredients = order['required']

    # Determine what equipment we need
    needs_cooking = any(
        INGREDIENT_PROPS.get(ing, {}).get('can_cook', False)
        for ing in ingredients
    )
    needs_chopping = any(
        INGREDIENT_PROPS.get(ing, {}).get('can_chop', False)
        for ing in ingredients
    )

    # Resource IDs based on bound locations
    cooker_resource = None
    if kitchen_plan.cooker:
        cooker_resource = f"cooker_{kitchen_plan.cooker[0]}_{kitchen_plan.cooker[1]}"

    assembly_resource = None
    if kitchen_plan.assembly_counter:
        assembly_resource = f"counter_{kitchen_plan.assembly_counter[0]}_{kitchen_plan.assembly_counter[1]}"

    # ==========================================
    # Equipment Milestones
    # ==========================================

    if needs_cooking:
        milestones['pan_ready'] = BoundMilestone(
            id='pan_ready',
            description='Pan placed on cooker',
            deps=[],
            bound={'cooker': kitchen_plan.cooker} if kitchen_plan.cooker else {},
            resources={cooker_resource} if cooker_resource else set()
        )

    # Plate milestone - can be achieved via buy or sinktable
    milestones['plate_ready'] = BoundMilestone(
        id='plate_ready',
        description='Clean plate available on assembly counter',
        deps=[],
        bound={
            'assembly': kitchen_plan.assembly_counter,
            'shop': kitchen_plan.shop,
            'sinktable': kitchen_plan.sinktable
        },
        resources={assembly_resource} if assembly_resource else set()
    )

    # ==========================================
    # Per-Ingredient Milestones
    # ==========================================

    for ing in ingredients:
        props = INGREDIENT_PROPS.get(ing, {})
        can_chop = props.get('can_chop', False)
        can_cook = props.get('can_cook', False)
        ing_lower = ing.lower()

        # Buy milestone
        buy_id = f'{ing_lower}_bought'
        milestones[buy_id] = BoundMilestone(
            id=buy_id,
            description=f'Buy {ing}',
            deps=[],
            bound={'shop': kitchen_plan.shop},
            ingredient=ing
        )

        prev_milestone = buy_id

        # Chop milestone (if needed)
        if can_chop:
            chop_id = f'{ing_lower}_chopped'
            milestones[chop_id] = BoundMilestone(
                id=chop_id,
                description=f'{ing} chopped',
                deps=[prev_milestone],
                bound={'counter': kitchen_plan.chop_counter or kitchen_plan.assembly_counter},
                ingredient=ing
            )
            prev_milestone = chop_id

        # Cooking milestones (split into stages)
        if can_cook:
            # Start cooking (place in pan)
            start_id = f'{ing_lower}_cooking_started'
            milestones[start_id] = BoundMilestone(
                id=start_id,
                description=f'{ing} placed in pan, cooking started',
                deps=[prev_milestone, 'pan_ready'],
                bound={'cooker': kitchen_plan.cooker},
                resources={cooker_resource} if cooker_resource else set(),
                ingredient=ing
            )

            # Cooking in progress (external event, no action needed)
            cooking_id = f'{ing_lower}_cooking'
            milestones[cooking_id] = BoundMilestone(
                id=cooking_id,
                description=f'{ing} cooking (waiting)',
                deps=[start_id],
                status=MilestoneStatus.BLOCKED,  # Can't be "worked on"
                ingredient=ing
            )

            # Take from pan (when cooked_stage == 1)
            take_id = f'{ing_lower}_cooked'
            milestones[take_id] = BoundMilestone(
                id=take_id,
                description=f'{ing} taken from pan (cooked)',
                deps=[cooking_id],
                bound={'cooker': kitchen_plan.cooker},
                ingredient=ing
            )
            prev_milestone = take_id

        # Add to plate milestone
        # NOTE: Only depends on plate_ready, NOT on cooking/chopping of OTHER ingredients
        plate_id = f'{ing_lower}_on_plate'
        milestones[plate_id] = BoundMilestone(
            id=plate_id,
            description=f'{ing} added to plate',
            deps=[prev_milestone, 'plate_ready'],
            bound={'assembly': kitchen_plan.assembly_counter},
            ingredient=ing
        )

    # ==========================================
    # Submit Milestone
    # ==========================================

    plate_deps = [f'{ing.lower()}_on_plate' for ing in ingredients]
    milestones['submitted'] = BoundMilestone(
        id='submitted',
        description='Order submitted',
        deps=plate_deps,
        bound={'submit': kitchen_plan.submit}
    )

    # Create plan
    plan = OrderPlan(
        order_id=order['order_id'],
        order=order,
        milestones=milestones,
        cooker_pos=kitchen_plan.cooker,
        assembly_counter_pos=kitchen_plan.assembly_counter,
        chop_counter_pos=kitchen_plan.chop_counter
    )

    return plan


def prioritize_milestones(milestones: List[BoundMilestone]) -> List[BoundMilestone]:
    """
    Sort milestones by priority.

    Priority (highest first):
    1. Cooking-related (start timer early)
    2. Equipment (pan, plate)
    3. Buying (enables downstream)
    4. Chopping
    5. Plating
    6. Submit (last)
    """
    def priority_key(m: BoundMilestone) -> tuple:
        # Lower = higher priority
        is_cooking_start = '_cooking_started' in m.id
        is_cooking_take = '_cooked' in m.id and '_cooking' not in m.id
        is_equipment = m.id in ['pan_ready', 'plate_ready']
        is_buy = '_bought' in m.id
        is_chop = '_chopped' in m.id
        is_plate = '_on_plate' in m.id
        is_submit = m.id == 'submitted'
        num_deps = len(m.deps)

        return (
            not is_cooking_start,  # Start cooking ASAP
            not is_cooking_take,   # Then take cooked food
            not is_equipment,      # Then equipment
            not is_buy,            # Then buying
            not is_chop,           # Then chopping
            not is_plate,          # Then plating
            is_submit,             # Submit last
            num_deps,              # Fewer deps = easier
        )

    return sorted(milestones, key=priority_key)


def get_parallel_milestones(plan: OrderPlan) -> List[BoundMilestone]:
    """
    Get milestones that can be worked on in parallel with cooking.

    While one ingredient is cooking (20 turns), other work can happen:
    - Buy other ingredients
    - Chop other ingredients
    - Get plate ready
    """
    cooking_in_progress = any(
        '_cooking' in m.id and m.status != MilestoneStatus.COMPLETE
        for m in plan.milestones.values()
    )

    if not cooking_in_progress:
        return []

    # Find milestones not blocked by cooking
    parallel = []
    for m in plan.get_ready_milestones():
        # Skip cooking-related milestones for the same ingredient
        if '_cooking' in m.id or '_cooked' in m.id:
            continue
        parallel.append(m)

    return parallel


def update_cooking_status(plan: OrderPlan, rc) -> List[str]:
    """
    Check cooker state and update cooking milestones.

    Returns list of milestones that became ready (cooked).
    """
    newly_ready = []
    team = rc.get_team()

    if plan.cooker_pos is None:
        return newly_ready

    tile = rc.get_tile(team, plan.cooker_pos[0], plan.cooker_pos[1])
    if tile is None:
        return newly_ready

    # Check pan state
    pan = getattr(tile, 'item', None)
    if pan is None:
        return newly_ready

    food = getattr(pan, 'food', None)
    if food is None:
        return newly_ready

    cooked_stage = getattr(food, 'cooked_stage', 0)
    food_name = getattr(food, 'food_name', None)

    if food_name is None:
        return newly_ready

    ing_lower = food_name.lower()
    cooking_id = f'{ing_lower}_cooking'
    take_id = f'{ing_lower}_cooked'

    # If cooked (stage 1), mark cooking complete and take ready
    if cooked_stage >= 1:
        if cooking_id in plan.milestones:
            if plan.milestones[cooking_id].status != MilestoneStatus.COMPLETE:
                plan.mark_complete(cooking_id)

        if take_id in plan.milestones:
            m = plan.milestones[take_id]
            if m.status not in [MilestoneStatus.COMPLETE, MilestoneStatus.READY]:
                m.status = MilestoneStatus.READY
                newly_ready.append(take_id)

    return newly_ready


def get_urgency_milestones(plan: OrderPlan, rc, current_turn: int) -> List[Tuple[BoundMilestone, float]]:
    """
    Get milestones with urgency scores.

    Urgent events:
    1. Cooked food about to burn (CRITICAL)
    2. Order about to expire
    """
    urgent = []
    team = rc.get_team()

    # Check for cooked food
    if plan.cooker_pos:
        tile = rc.get_tile(team, plan.cooker_pos[0], plan.cooker_pos[1])
        if tile:
            pan = getattr(tile, 'item', None)
            if pan:
                food = getattr(pan, 'food', None)
                if food:
                    cooked_stage = getattr(food, 'cooked_stage', 0)
                    cook_progress = getattr(tile, 'cook_progress', 0)

                    if cooked_stage == 1:
                        # Food is cooked, need to take it!
                        # Urgency increases as we approach burn time (40)
                        ticks_until_burn = 40 - cook_progress
                        urgency = 1000.0 / max(1, ticks_until_burn)

                        # Find the take milestone
                        food_name = getattr(food, 'food_name', '')
                        take_id = f'{food_name.lower()}_cooked'
                        if take_id in plan.milestones:
                            m = plan.milestones[take_id]
                            if m.status != MilestoneStatus.COMPLETE:
                                urgent.append((m, urgency))

    # Check order expiry
    expires = plan.order.get('expires_turn', 999999)
    turns_remaining = expires - current_turn

    if turns_remaining < 50:  # Getting close
        urgency = 100.0 / max(1, turns_remaining)
        # Boost all incomplete milestones
        for m in plan.milestones.values():
            if m.status != MilestoneStatus.COMPLETE:
                # Don't double-add cooking urgency
                if not any(u[0].id == m.id for u in urgent):
                    urgent.append((m, urgency * 0.5))

    return sorted(urgent, key=lambda x: x[1], reverse=True)
