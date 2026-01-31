"""
Atomic Operators - Single-turn actions aligned with the game API.

Each operator represents at most one move + one action per turn.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Set, Any, List
from enum import Enum, auto


class OpType(Enum):
    """Atomic operator types, each maps to at most one API action."""

    # Movement only (no action)
    MOVE_TO = auto()

    # Actions at shop
    BUY_INGREDIENT = auto()
    BUY_PLATE = auto()
    BUY_PAN = auto()

    # Item manipulation
    PLACE = auto()
    PICKUP = auto()
    TRASH = auto()

    # Food processing
    CHOP = auto()
    START_COOK = auto()  # Place food into pan (starts cooking)
    TAKE_FROM_PAN = auto()

    # Plating
    ADD_TO_PLATE = auto()

    # Submission
    SUBMIT = auto()

    # Plate management
    PUT_DIRTY_IN_SINK = auto()
    WASH_SINK = auto()
    TAKE_CLEAN_PLATE = auto()

    # No-op
    WAIT = auto()
    NOOP = auto()


@dataclass
class Operator:
    """
    An atomic operator that can be executed in one turn.

    Execution model:
    1. If move_target is set and bot not adjacent, move toward it
    2. If adjacent to action_target (or move put us adjacent), perform action
    3. A turn can do: move only, action only, or move + action
    """
    op_type: OpType
    action_target: Optional[Tuple[int, int]] = None  # Where to perform action
    item: Optional[str] = None  # Item type for buy/etc
    ingredient: Optional[str] = None  # Specific ingredient name
    priority: float = 0.0

    # Resource requirements
    required_resources: Set[str] = field(default_factory=set)

    # Preconditions (checked before execution)
    requires_empty_hands: bool = False
    requires_holding_type: Optional[str] = None  # 'Food', 'Plate', 'Pan'
    requires_holding_ingredient: Optional[str] = None

    # Associated milestone (for tracking)
    milestone_id: Optional[str] = None

    def __repr__(self):
        parts = [self.op_type.name]
        if self.ingredient:
            parts.append(self.ingredient)
        if self.item:
            parts.append(self.item)
        if self.action_target:
            parts.append(f"@{self.action_target}")
        return f"Op({', '.join(parts)})"


# ============================================
# Operator Builders (convenience functions)
# ============================================

def op_move_to(target: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Move adjacent to a target (no action)."""
    return Operator(
        op_type=OpType.MOVE_TO,
        action_target=target,
        priority=priority
    )


def op_buy_ingredient(ingredient: str, shop_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Buy an ingredient from shop."""
    return Operator(
        op_type=OpType.BUY_INGREDIENT,
        action_target=shop_pos,
        ingredient=ingredient,
        priority=priority,
        requires_empty_hands=True
    )


def op_buy_plate(shop_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Buy a plate from shop."""
    return Operator(
        op_type=OpType.BUY_PLATE,
        action_target=shop_pos,
        item='PLATE',
        priority=priority,
        requires_empty_hands=True
    )


def op_buy_pan(shop_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Buy a pan from shop."""
    return Operator(
        op_type=OpType.BUY_PAN,
        action_target=shop_pos,
        item='PAN',
        priority=priority,
        requires_empty_hands=True
    )


def op_place(target: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Place held item at target."""
    return Operator(
        op_type=OpType.PLACE,
        action_target=target,
        priority=priority,
        requires_holding_type='any'  # Must hold something
    )


def op_pickup(target: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Pick up item from target."""
    return Operator(
        op_type=OpType.PICKUP,
        action_target=target,
        priority=priority,
        requires_empty_hands=True
    )


def op_chop(counter_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Chop food on counter."""
    return Operator(
        op_type=OpType.CHOP,
        action_target=counter_pos,
        priority=priority,
        requires_empty_hands=True  # Chop requires empty hands
    )


def op_start_cook(cooker_pos: Tuple[int, int], ingredient: str, priority: float = 0.0) -> Operator:
    """Place food in pan to start cooking."""
    return Operator(
        op_type=OpType.START_COOK,
        action_target=cooker_pos,
        ingredient=ingredient,
        priority=priority,
        requires_holding_ingredient=ingredient
    )


def op_take_from_pan(cooker_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Take cooked food from pan."""
    return Operator(
        op_type=OpType.TAKE_FROM_PAN,
        action_target=cooker_pos,
        priority=priority,
        requires_empty_hands=True
    )


def op_add_to_plate(plate_pos: Tuple[int, int], ingredient: str, priority: float = 0.0) -> Operator:
    """Add held food to plate."""
    return Operator(
        op_type=OpType.ADD_TO_PLATE,
        action_target=plate_pos,
        ingredient=ingredient,
        priority=priority,
        requires_holding_ingredient=ingredient
    )


def op_submit(submit_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Submit plate at submit station."""
    return Operator(
        op_type=OpType.SUBMIT,
        action_target=submit_pos,
        priority=priority,
        requires_holding_type='Plate'
    )


def op_take_clean_plate(sinktable_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Take clean plate from sinktable."""
    return Operator(
        op_type=OpType.TAKE_CLEAN_PLATE,
        action_target=sinktable_pos,
        priority=priority,
        requires_empty_hands=True
    )


def op_put_dirty_in_sink(sink_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Put dirty plate in sink."""
    return Operator(
        op_type=OpType.PUT_DIRTY_IN_SINK,
        action_target=sink_pos,
        priority=priority,
        requires_holding_type='Plate'
    )


def op_wash_sink(sink_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Wash dishes in sink."""
    return Operator(
        op_type=OpType.WASH_SINK,
        action_target=sink_pos,
        priority=priority
    )


def op_wait(priority: float = -100.0) -> Operator:
    """Do nothing this turn (intentional wait)."""
    return Operator(
        op_type=OpType.WAIT,
        priority=priority
    )


def op_noop() -> Operator:
    """No operation (nothing to do)."""
    return Operator(
        op_type=OpType.NOOP,
        priority=-1000.0
    )


def op_trash(trash_pos: Tuple[int, int], priority: float = 0.0) -> Operator:
    """Trash held item."""
    return Operator(
        op_type=OpType.TRASH,
        action_target=trash_pos,
        priority=priority,
        requires_holding_type='any'
    )


# ============================================
# Operator Execution
# ============================================

class OperatorExecutor:
    """Executes operators using the game API."""

    def __init__(self, rc):
        self.rc = rc
        self.team = rc.get_team()

    def can_execute(self, bot_id: int, op: Operator) -> bool:
        """Check if operator preconditions are met."""
        state = self.rc.get_bot_state(bot_id)
        if state is None:
            return False

        holding = state.get('holding')

        # Check hand requirements
        if op.requires_empty_hands and holding is not None:
            return False

        if op.requires_holding_type:
            if op.requires_holding_type == 'any':
                if holding is None:
                    return False
            else:
                if holding is None or holding.get('type') != op.requires_holding_type:
                    return False

        if op.requires_holding_ingredient:
            if holding is None or holding.get('type') != 'Food':
                return False
            if holding.get('food_name') != op.requires_holding_ingredient:
                return False

        return True

    def execute(self, bot_id: int, op: Operator) -> Tuple[bool, bool]:
        """
        Execute an operator for a bot.

        Returns: (moved: bool, acted: bool)
        """
        if op.op_type == OpType.NOOP:
            return False, False

        if op.op_type == OpType.WAIT:
            return False, False

        state = self.rc.get_bot_state(bot_id)
        if state is None:
            return False, False

        bx, by = state['x'], state['y']
        moved = False
        acted = False

        # Move toward target if needed
        if op.action_target:
            tx, ty = op.action_target
            if not self._is_adjacent(bx, by, tx, ty):
                moved = self._move_toward(bot_id, tx, ty)
                # Update position after move
                state = self.rc.get_bot_state(bot_id)
                if state:
                    bx, by = state['x'], state['y']

        # Perform action if adjacent
        if op.action_target:
            tx, ty = op.action_target
            if self._is_adjacent(bx, by, tx, ty):
                acted = self._perform_action(bot_id, op, tx, ty)

        return moved, acted

    def _is_adjacent(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if two positions are adjacent (Chebyshev distance <= 1)."""
        return max(abs(x1 - x2), abs(y1 - y2)) <= 1

    def _move_toward(self, bot_id: int, tx: int, ty: int) -> bool:
        """Move one step toward target."""
        state = self.rc.get_bot_state(bot_id)
        if state is None:
            return False

        bx, by = state['x'], state['y']
        best_move = None
        best_dist = float('inf')

        # Try all 8 directions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if self.rc.can_move(bot_id, dx, dy):
                    new_x, new_y = bx + dx, by + dy
                    dist = max(abs(new_x - tx), abs(new_y - ty))
                    if dist < best_dist:
                        best_dist = dist
                        best_move = (dx, dy)

        if best_move:
            return self.rc.move(bot_id, best_move[0], best_move[1])

        return False

    def _perform_action(self, bot_id: int, op: Operator, tx: int, ty: int) -> bool:
        """Perform the action part of an operator."""
        op_type = op.op_type

        if op_type == OpType.MOVE_TO:
            return True  # No action needed, just movement

        elif op_type == OpType.BUY_INGREDIENT:
            from game_constants import FoodType
            food_type = getattr(FoodType, op.ingredient, None)
            if food_type:
                return self.rc.buy(bot_id, food_type, tx, ty)

        elif op_type == OpType.BUY_PLATE:
            from game_constants import ShopCosts
            return self.rc.buy(bot_id, ShopCosts.PLATE, tx, ty)

        elif op_type == OpType.BUY_PAN:
            from game_constants import ShopCosts
            return self.rc.buy(bot_id, ShopCosts.PAN, tx, ty)

        elif op_type == OpType.PLACE:
            return self.rc.place(bot_id, tx, ty)

        elif op_type == OpType.PICKUP:
            return self.rc.pickup(bot_id, tx, ty)

        elif op_type == OpType.TRASH:
            return self.rc.trash(bot_id, tx, ty)

        elif op_type == OpType.CHOP:
            return self.rc.chop(bot_id, tx, ty)

        elif op_type == OpType.START_COOK:
            # Place food into pan starts cooking automatically
            return self.rc.place(bot_id, tx, ty)

        elif op_type == OpType.TAKE_FROM_PAN:
            return self.rc.take_from_pan(bot_id, tx, ty)

        elif op_type == OpType.ADD_TO_PLATE:
            return self.rc.add_food_to_plate(bot_id, tx, ty)

        elif op_type == OpType.SUBMIT:
            return self.rc.submit(bot_id, tx, ty)

        elif op_type == OpType.TAKE_CLEAN_PLATE:
            return self.rc.take_clean_plate(bot_id, tx, ty)

        elif op_type == OpType.PUT_DIRTY_IN_SINK:
            return self.rc.put_dirty_plate_in_sink(bot_id, tx, ty)

        elif op_type == OpType.WASH_SINK:
            return self.rc.wash_sink(bot_id, tx, ty)

        return False


# ============================================
# Operator Sequencing
# ============================================

def get_operators_for_milestone(
    milestone_id: str,
    ingredient: Optional[str],
    plan: Any,  # KitchenPlan
    current_state: dict
) -> List[Operator]:
    """
    Generate the sequence of operators needed to complete a milestone.

    This is used for planning; actual execution picks one at a time.
    """
    ops = []

    if milestone_id == 'have_pan':
        if plan.shop:
            ops.append(op_buy_pan(plan.shop))
        if plan.cooker:
            ops.append(op_place(plan.cooker))

    elif milestone_id == 'have_plate':
        # Prefer sinktable if available
        if plan.sinktable and current_state.get('clean_plates_available', 0) > 0:
            ops.append(op_take_clean_plate(plan.sinktable))
        elif plan.shop:
            ops.append(op_buy_plate(plan.shop))
        if plan.assembly_counter:
            ops.append(op_place(plan.assembly_counter))

    elif '_bought' in milestone_id and ingredient:
        if plan.shop:
            ops.append(op_buy_ingredient(ingredient, plan.shop))

    elif '_chopped' in milestone_id and ingredient:
        if plan.chop_counter:
            ops.append(op_place(plan.chop_counter))
            ops.append(op_chop(plan.chop_counter))
            ops.append(op_pickup(plan.chop_counter))

    elif '_cooking' in milestone_id and ingredient:
        # Start cooking (food goes in pan)
        if plan.cooker:
            ops.append(op_start_cook(plan.cooker, ingredient))

    elif '_cooked' in milestone_id and ingredient:
        # Take from pan when ready
        if plan.cooker:
            ops.append(op_take_from_pan(plan.cooker))

    elif '_on_plate' in milestone_id and ingredient:
        if plan.assembly_counter:
            ops.append(op_add_to_plate(plan.assembly_counter, ingredient))

    elif milestone_id == 'submitted':
        if plan.assembly_counter:
            ops.append(op_pickup(plan.assembly_counter))
        if plan.submit:
            ops.append(op_submit(plan.submit))

    return ops
