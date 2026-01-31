"""
Milestone System for Order Decomposition

Breaks orders into milestones with dependencies, enabling parallel
execution and reactive scheduling.
"""

from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field


class MilestoneStatus(Enum):
    PENDING = "pending"       # Not started, deps not met
    READY = "ready"           # Deps met, can be worked on
    IN_PROGRESS = "in_progress"  # Currently being worked on
    WAITING = "waiting"       # Waiting for external condition (e.g., cooking)
    COMPLETE = "complete"     # Done


class TaskType(Enum):
    # Movement
    MOVE_TO = "move_to"

    # Shopping
    BUY_INGREDIENT = "buy_ingredient"
    BUY_PLATE = "buy_plate"
    BUY_PAN = "buy_pan"

    # Food processing
    PLACE_ON_COUNTER = "place_on_counter"
    CHOP = "chop"
    PICKUP = "pickup"

    # Cooking
    PLACE_IN_PAN = "place_in_pan"
    START_COOK = "start_cook"
    WAIT_FOR_COOK = "wait_for_cook"
    TAKE_FROM_PAN = "take_from_pan"

    # Plating
    ADD_TO_PLATE = "add_to_plate"

    # Submission
    SUBMIT = "submit"

    # Utility
    IDLE = "idle"


@dataclass
class Task:
    """Represents a single atomic task."""
    task_type: TaskType
    target: Optional[tuple] = None  # (x, y) position
    item: Optional[str] = None      # Item name/type
    ingredient: Optional[str] = None  # Ingredient name
    milestone_id: Optional[str] = None  # Associated milestone

    def __repr__(self):
        parts = [self.task_type.value]
        if self.ingredient:
            parts.append(self.ingredient)
        if self.item:
            parts.append(self.item)
        if self.target:
            parts.append(str(self.target))
        return f"Task({', '.join(parts)})"


@dataclass
class Milestone:
    """Represents a milestone in order completion."""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: MilestoneStatus = MilestoneStatus.PENDING
    assigned_bot: Optional[int] = None
    tasks: List[Task] = field(default_factory=list)

    # For tracking progress
    ingredient: Optional[str] = None  # If milestone is ingredient-specific
    cook_start_turn: Optional[int] = None  # When cooking started

    def is_ready(self, completed_milestones: Set[str]) -> bool:
        """Check if all dependencies are met."""
        return all(dep in completed_milestones for dep in self.dependencies)


# Ingredient properties (duplicated for standalone use)
INGREDIENT_PROPS = {
    'EGG': {'can_chop': False, 'can_cook': True},
    'ONIONS': {'can_chop': True, 'can_cook': False},
    'MEAT': {'can_chop': True, 'can_cook': True},
    'NOODLES': {'can_chop': False, 'can_cook': False},
    'SAUCE': {'can_chop': False, 'can_cook': False},
}


def build_milestones_for_order(order: Dict[str, Any]) -> Dict[str, Milestone]:
    """
    Build milestone dependency graph for an order.

    Returns dict mapping milestone_id -> Milestone
    """
    milestones: Dict[str, Milestone] = {}
    ingredients = order['required']

    # Track which ingredients need cooking (to determine if we need a pan)
    needs_cooking = any(
        INGREDIENT_PROPS.get(ing, {}).get('can_cook', False)
        for ing in ingredients
    )

    # ==========================================
    # Equipment milestones
    # ==========================================

    # Always need a plate
    milestones['have_plate'] = Milestone(
        id='have_plate',
        description='Acquire a plate',
        dependencies=[],
        tasks=[Task(TaskType.BUY_PLATE)]
    )

    # Need pan if any ingredient requires cooking
    if needs_cooking:
        milestones['have_pan'] = Milestone(
            id='have_pan',
            description='Acquire a pan and place on cooker',
            dependencies=[],
            tasks=[Task(TaskType.BUY_PAN)]
        )

    # ==========================================
    # Per-ingredient milestones
    # ==========================================

    for ing in ingredients:
        props = INGREDIENT_PROPS.get(ing, {})
        can_chop = props.get('can_chop', False)
        can_cook = props.get('can_cook', False)

        ing_lower = ing.lower()

        # Milestone: Buy ingredient
        buy_id = f'{ing_lower}_bought'
        milestones[buy_id] = Milestone(
            id=buy_id,
            description=f'Buy {ing}',
            dependencies=[],
            ingredient=ing,
            tasks=[Task(TaskType.BUY_INGREDIENT, ingredient=ing)]
        )

        prev_milestone = buy_id

        # Milestone: Chop (if choppable)
        if can_chop:
            chop_id = f'{ing_lower}_chopped'
            milestones[chop_id] = Milestone(
                id=chop_id,
                description=f'Chop {ing}',
                dependencies=[prev_milestone],
                ingredient=ing,
                tasks=[
                    Task(TaskType.PLACE_ON_COUNTER, ingredient=ing),
                    Task(TaskType.CHOP, ingredient=ing),
                    Task(TaskType.PICKUP, ingredient=ing),
                ]
            )
            prev_milestone = chop_id

        # Milestone: Cook (if cookable)
        if can_cook:
            cook_id = f'{ing_lower}_cooked'
            milestones[cook_id] = Milestone(
                id=cook_id,
                description=f'Cook {ing}',
                dependencies=[prev_milestone, 'have_pan'],
                ingredient=ing,
                tasks=[
                    Task(TaskType.PLACE_IN_PAN, ingredient=ing),
                    Task(TaskType.WAIT_FOR_COOK, ingredient=ing),
                    Task(TaskType.TAKE_FROM_PAN, ingredient=ing),
                ]
            )
            prev_milestone = cook_id

        # Milestone: Add to plate
        plate_id = f'{ing_lower}_on_plate'
        milestones[plate_id] = Milestone(
            id=plate_id,
            description=f'Add {ing} to plate',
            dependencies=[prev_milestone, 'have_plate'],
            ingredient=ing,
            tasks=[Task(TaskType.ADD_TO_PLATE, ingredient=ing)]
        )

    # ==========================================
    # Final milestone: Submit
    # ==========================================

    # Depends on all ingredients being on plate
    plate_deps = [f'{ing.lower()}_on_plate' for ing in ingredients]
    milestones['submitted'] = Milestone(
        id='submitted',
        description='Submit the order',
        dependencies=plate_deps,
        tasks=[Task(TaskType.SUBMIT)]
    )

    return milestones


class OrderProgress:
    """Tracks progress on an order's milestones."""

    def __init__(self, order: Dict[str, Any]):
        self.order = order
        self.order_id = order['order_id']
        self.milestones = build_milestones_for_order(order)
        self.completed: Set[str] = set()

        # Track current state
        self.bot_assignments: Dict[int, str] = {}  # bot_id -> milestone_id
        self.plate_location: Optional[tuple] = None  # Where plate is placed
        self.plate_contents: List[str] = []  # Ingredients on plate

    def mark_complete(self, milestone_id: str):
        """Mark a milestone as complete."""
        if milestone_id in self.milestones:
            self.milestones[milestone_id].status = MilestoneStatus.COMPLETE
            self.completed.add(milestone_id)

    def is_complete(self) -> bool:
        """Check if order is fully complete."""
        return 'submitted' in self.completed

    def get_ready_milestones(self) -> List[Milestone]:
        """Get all milestones that are ready to be worked on."""
        ready = []
        for m_id, milestone in self.milestones.items():
            if milestone.status in [MilestoneStatus.PENDING, MilestoneStatus.READY]:
                if milestone.is_ready(self.completed):
                    milestone.status = MilestoneStatus.READY
                    ready.append(milestone)
        return ready

    def get_unassigned_ready_milestones(self) -> List[Milestone]:
        """Get ready milestones not assigned to any bot."""
        assigned_ids = set(self.bot_assignments.values())
        return [m for m in self.get_ready_milestones() if m.id not in assigned_ids]

    def assign_milestone(self, bot_id: int, milestone_id: str):
        """Assign a milestone to a bot."""
        self.bot_assignments[bot_id] = milestone_id
        if milestone_id in self.milestones:
            self.milestones[milestone_id].assigned_bot = bot_id
            self.milestones[milestone_id].status = MilestoneStatus.IN_PROGRESS

    def unassign_bot(self, bot_id: int):
        """Remove a bot's assignment."""
        if bot_id in self.bot_assignments:
            m_id = self.bot_assignments[bot_id]
            if m_id in self.milestones:
                self.milestones[m_id].assigned_bot = None
                # Reset to ready if deps still met
                if self.milestones[m_id].is_ready(self.completed):
                    self.milestones[m_id].status = MilestoneStatus.READY
                else:
                    self.milestones[m_id].status = MilestoneStatus.PENDING
            del self.bot_assignments[bot_id]

    def get_bot_milestone(self, bot_id: int) -> Optional[Milestone]:
        """Get the milestone assigned to a bot."""
        m_id = self.bot_assignments.get(bot_id)
        if m_id:
            return self.milestones.get(m_id)
        return None

    def get_waiting_milestones(self) -> List[Milestone]:
        """Get milestones that are waiting (e.g., for cooking)."""
        return [m for m in self.milestones.values()
                if m.status == MilestoneStatus.WAITING]

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress."""
        total = len(self.milestones)
        complete = len(self.completed)
        ready = len(self.get_ready_milestones())
        in_progress = len([m for m in self.milestones.values()
                          if m.status == MilestoneStatus.IN_PROGRESS])
        waiting = len(self.get_waiting_milestones())

        return {
            'order_id': self.order_id,
            'total_milestones': total,
            'completed': complete,
            'ready': ready,
            'in_progress': in_progress,
            'waiting': waiting,
            'percent_complete': (complete / total * 100) if total > 0 else 0,
        }


def prioritize_milestones(milestones: List[Milestone]) -> List[Milestone]:
    """
    Sort milestones by priority.

    Priority rules:
    1. Cooking-related milestones first (to start the 20-turn timer)
    2. Equipment milestones (pan, plate) early
    3. Milestones with fewer dependencies
    """
    def priority_key(m: Milestone) -> tuple:
        # Lower tuple = higher priority
        is_cook = '_cooked' in m.id
        is_equipment = m.id in ['have_pan', 'have_plate']
        is_submit = m.id == 'submitted'
        num_deps = len(m.dependencies)

        return (
            not is_cook,      # Cooking first
            not is_equipment,  # Equipment second
            is_submit,         # Submit last
            num_deps,          # Fewer deps = easier to start
        )

    return sorted(milestones, key=priority_key)


def get_cooking_milestones(progress: OrderProgress) -> List[Milestone]:
    """Get all milestones that involve cooking."""
    return [m for m in progress.milestones.values() if '_cooked' in m.id]


def get_parallel_work(progress: OrderProgress) -> List[Milestone]:
    """
    Get milestones that can be worked on in parallel with cooking.
    Useful for utilizing the 20-turn cook time.
    """
    cooking = get_cooking_milestones(progress)
    cooking_ids = {m.id for m in cooking}
    cooking_deps = set()
    for m in cooking:
        cooking_deps.update(m.dependencies)

    # Find ready milestones that aren't cooking-related
    ready = progress.get_ready_milestones()
    parallel = [m for m in ready
                if m.id not in cooking_ids
                and m.id not in cooking_deps]

    return parallel
