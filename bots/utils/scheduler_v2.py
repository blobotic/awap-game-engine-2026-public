"""
Scheduler V2 - Reactive scheduling with resource locks and urgency.

Key features:
- Resource locking prevents double-assignment
- Station binding avoids oscillation
- Urgency preemption for time-critical events
- Two-bot coordination with collision avoidance
- Action failure handling with retry limits
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import deque

# These would be imported from other modules in full version
try:
    from .resources import (
        ResourceManager, ResourceType, KitchenPlan, compute_kitchen_plan, chebyshev
    )
    from .operators import (
        Operator, OpType, OperatorExecutor,
        op_move_to, op_buy_ingredient, op_buy_plate, op_buy_pan,
        op_place, op_pickup, op_chop, op_start_cook, op_take_from_pan,
        op_add_to_plate, op_submit, op_take_clean_plate, op_wait, op_noop, op_trash
    )
    from .planner import (
        OrderPlan, BoundMilestone, MilestoneStatus,
        build_order_plan, prioritize_milestones, get_parallel_milestones,
        update_cooking_status, get_urgency_milestones,
        INGREDIENT_PROPS, INGREDIENT_COSTS
    )
    from .orders import (
        get_active_orders, get_order_total_cost, get_order_profit,
        get_order_turns_remaining
    )
except ImportError:
    from resources import (
        ResourceManager, ResourceType, KitchenPlan, compute_kitchen_plan, chebyshev
    )
    from operators import (
        Operator, OpType, OperatorExecutor,
        op_move_to, op_buy_ingredient, op_buy_plate, op_buy_pan,
        op_place, op_pickup, op_chop, op_start_cook, op_take_from_pan,
        op_add_to_plate, op_submit, op_take_clean_plate, op_wait, op_noop, op_trash
    )
    from planner import (
        OrderPlan, BoundMilestone, MilestoneStatus,
        build_order_plan, prioritize_milestones, get_parallel_milestones,
        update_cooking_status, get_urgency_milestones,
        INGREDIENT_PROPS, INGREDIENT_COSTS
    )
    from orders import (
        get_active_orders, get_order_total_cost, get_order_profit,
        get_order_turns_remaining
    )


PLATE_COST = 2
PAN_COST = 4
MAX_RETRIES = 3


@dataclass
class BotState:
    """Per-bot state tracking."""
    bot_id: int
    current_milestone: Optional[str] = None
    current_op: Optional[Operator] = None
    retry_count: int = 0
    last_pos: Optional[Tuple[int, int]] = None
    stuck_turns: int = 0


class SchedulerV2:
    """
    Main scheduler coordinating two bots.

    Execution flow per turn:
    1. Clear tile reservations
    2. Update cooking status
    3. Check urgency (preempt if needed)
    4. For each bot:
       a. Check if current task still valid
       b. If not, select next best task with resource locking
       c. Execute: move + action
    5. Handle failures
    """

    def __init__(self, rc):
        self.rc = rc
        self.team = rc.get_team()

        # Managers
        self.resources = ResourceManager()
        self.executor = OperatorExecutor(rc)

        # Planning
        self.kitchen_plan: Optional[KitchenPlan] = None
        self.order_plan: Optional[OrderPlan] = None
        self.current_order: Optional[Dict[str, Any]] = None

        # Per-bot state
        self.bot_states: Dict[int, BotState] = {}

        # Turn tracking
        self.current_turn = 0

        # Caches (refreshed per turn)
        self._map_cache = None

    def initialize(self, rc):
        """Initialize scheduler (call once at start)."""
        self.rc = rc
        self.team = rc.get_team()
        self.kitchen_plan = compute_kitchen_plan(rc, self.team)
        self._register_resources()

        # Initialize bot states
        for bot_id in rc.get_team_bot_ids(self.team):
            self.bot_states[bot_id] = BotState(bot_id=bot_id)

    def _register_resources(self):
        """Register all resources from kitchen plan."""
        if self.kitchen_plan is None:
            return

        # Register cookers
        for pos in self.kitchen_plan.all_cookers:
            self.resources.register_resource(ResourceType.COOKER, pos)

        # Register counters
        for pos in self.kitchen_plan.all_counters:
            self.resources.register_resource(ResourceType.COUNTER, pos)

    def play_turn(self, rc):
        """Main entry point - execute one turn."""
        self.rc = rc
        self.current_turn = rc.get_turn()

        # Lazy initialization
        if self.kitchen_plan is None:
            self.initialize(rc)

        # Refresh caches
        self._map_cache = rc.get_map(self.team)
        self.resources.clear_tile_reservations()

        # Check if we need a new order
        if self._needs_new_order():
            self._select_new_order()

        if self.order_plan is None:
            return  # No work to do

        # Update cooking status
        update_cooking_status(self.order_plan, rc)

        # Get bot IDs
        bot_ids = rc.get_team_bot_ids(self.team)

        # Check for urgent tasks (preemption)
        urgent = get_urgency_milestones(self.order_plan, rc, self.current_turn)

        # Execute turn for each bot
        for bot_id in bot_ids:
            self._execute_bot_turn(bot_id, urgent)

    def _needs_new_order(self) -> bool:
        """Check if we need to select a new order."""
        if self.current_order is None:
            return True

        if self.order_plan and self.order_plan.is_complete():
            return True

        # Check if order expired
        expires = self.current_order.get('expires_turn', 0)
        if self.current_turn > expires:
            return True

        # Check if order was completed (by checking game state)
        orders = self.rc.get_orders(self.team)
        for o in orders:
            if o['order_id'] == self.current_order['order_id']:
                if o.get('completed_turn') is not None:
                    return True
                break

        return False

    def _select_new_order(self):
        """Select and plan for a new order."""
        # Release all resources from old plan
        if self.order_plan:
            for milestone in self.order_plan.milestones.values():
                self.resources.release_all_by(milestone.id)

        # Clear bot states
        for bs in self.bot_states.values():
            bs.current_milestone = None
            bs.current_op = None
            bs.retry_count = 0

        # Select best order
        orders = get_active_orders(self.rc, self.team)
        money = self.rc.get_team_money(self.team)

        # Filter affordable orders
        valid = [o for o in orders if get_order_total_cost(o) <= money]

        if not valid:
            self.current_order = None
            self.order_plan = None
            return

        # Score by profit per turn
        def score(o):
            turns = get_order_turns_remaining(o, self.current_turn)
            if turns <= 0:
                return float('-inf')
            return get_order_profit(o) / turns

        valid.sort(key=score, reverse=True)
        self.current_order = valid[0]

        # Build plan
        self.order_plan = build_order_plan(self.current_order, self.kitchen_plan)

    def _execute_bot_turn(self, bot_id: int, urgent: List[Tuple[BoundMilestone, float]]):
        """Execute a turn for one bot."""
        if self.order_plan is None:
            return

        bs = self.bot_states.get(bot_id)
        if bs is None:
            bs = BotState(bot_id=bot_id)
            self.bot_states[bot_id] = bs

        state = self.rc.get_bot_state(bot_id)
        if state is None:
            return

        # Track position for stuck detection
        pos = (state['x'], state['y'])
        if bs.last_pos == pos:
            bs.stuck_turns += 1
        else:
            bs.stuck_turns = 0
            bs.last_pos = pos

        # Reserve current position
        self.resources.reserve_tile(pos, bot_id)

        # Check if current task is still valid
        if not self._is_task_valid(bot_id, bs):
            bs.current_milestone = None
            bs.current_op = None
            bs.retry_count = 0

        # Handle urgent tasks (preemption)
        urgent_op = self._check_urgency(bot_id, state, urgent)
        if urgent_op:
            self._execute_op(bot_id, urgent_op, bs)
            return

        # Get next task if needed
        if bs.current_op is None:
            self._assign_next_task(bot_id, bs)

        # Execute current task
        if bs.current_op:
            self._execute_op(bot_id, bs.current_op, bs)

    def _is_task_valid(self, bot_id: int, bs: BotState) -> bool:
        """Check if bot's current task is still valid."""
        if bs.current_milestone is None:
            return False

        if self.order_plan is None:
            return False

        milestone = self.order_plan.milestones.get(bs.current_milestone)
        if milestone is None:
            return False

        # Check if already complete
        if milestone.status == MilestoneStatus.COMPLETE:
            return False

        # Check if blocked (cooking in progress)
        if milestone.status == MilestoneStatus.BLOCKED:
            return False

        # Check retry limit
        if bs.retry_count >= MAX_RETRIES:
            return False

        # Check stuck
        if bs.stuck_turns > 10:
            return False

        return True

    def _check_urgency(self, bot_id: int, state: dict, urgent: List[Tuple[BoundMilestone, float]]) -> Optional[Operator]:
        """Check for urgent tasks that should preempt current work."""
        if not urgent:
            return None

        holding = state.get('holding')
        bx, by = state['x'], state['y']

        for milestone, urgency in urgent:
            # Take from pan is highest priority
            if '_cooked' in milestone.id and urgency > 50:
                # Must have empty hands
                if holding is not None:
                    # Need to clear hands first
                    if self.kitchen_plan.assembly_counter:
                        return op_place(self.kitchen_plan.assembly_counter, priority=urgency)
                    continue

                # Take the food
                cooker = milestone.bound.get('cooker')
                if cooker:
                    return op_take_from_pan(cooker, priority=urgency)

        return None

    def _assign_next_task(self, bot_id: int, bs: BotState):
        """Assign the next best task to a bot."""
        if self.order_plan is None:
            return

        state = self.rc.get_bot_state(bot_id)
        if state is None:
            return

        holding = state.get('holding')
        bx, by = state['x'], state['y']

        # If holding something, we need to deal with it
        if holding:
            op = self._get_op_for_holding(bot_id, holding, bx, by)
            if op:
                bs.current_op = op
                return

        # Get ready milestones
        ready = self.order_plan.get_unassigned_ready()

        # Also consider parallel work while cooking
        parallel = get_parallel_milestones(self.order_plan)
        for m in parallel:
            if m not in ready and m.assigned_bot is None:
                ready.append(m)

        if not ready:
            bs.current_op = op_wait()
            return

        # Prioritize milestones
        ready = prioritize_milestones(ready)

        # Try to assign a milestone (check resource locks)
        for milestone in ready:
            if self._can_assign_milestone(bot_id, milestone):
                self._assign_milestone(bot_id, bs, milestone)
                return

        # Nothing available
        bs.current_op = op_wait()

    def _get_op_for_holding(self, bot_id: int, holding: dict, bx: int, by: int) -> Optional[Operator]:
        """Get operator based on what bot is holding."""
        h_type = holding.get('type')

        if h_type == 'Pan':
            # Place pan on cooker
            if self.kitchen_plan.cooker:
                return op_place(self.kitchen_plan.cooker)

        elif h_type == 'Plate':
            # Check if plate is ready to submit
            foods = holding.get('food', [])
            if self.current_order:
                required = set(self.current_order['required'])
                on_plate = {f.get('food_name') for f in foods if f}
                if required == on_plate:
                    if self.kitchen_plan.submit:
                        return op_submit(self.kitchen_plan.submit)

            # Otherwise place plate
            if self.kitchen_plan.assembly_counter:
                return op_place(self.kitchen_plan.assembly_counter)

        elif h_type == 'Food':
            food_name = holding.get('food_name')
            props = INGREDIENT_PROPS.get(food_name, {})
            chopped = holding.get('chopped', False)
            cooked = holding.get('cooked_stage', 0) >= 1

            # If needs chopping and not chopped, place on counter
            if props.get('can_chop') and not chopped:
                if self.kitchen_plan.chop_counter:
                    return op_place(self.kitchen_plan.chop_counter)

            # If needs cooking and not cooked, place in pan
            if props.get('can_cook') and not cooked:
                if self.kitchen_plan.cooker:
                    return op_start_cook(self.kitchen_plan.cooker, food_name)

            # Otherwise, add to plate
            if self.kitchen_plan.assembly_counter:
                return op_add_to_plate(self.kitchen_plan.assembly_counter, food_name)

        return None

    def _can_assign_milestone(self, bot_id: int, milestone: BoundMilestone) -> bool:
        """Check if milestone can be assigned (resource locks)."""
        # Check if resources are available
        for res_id in milestone.resources:
            holder = self.resources.get_lock_holder(res_id)
            if holder is not None and holder != milestone.id:
                # Resource locked by another milestone
                # Check if that milestone is assigned to us
                for bs in self.bot_states.values():
                    if bs.current_milestone == holder and bs.bot_id != bot_id:
                        return False

        return True

    def _assign_milestone(self, bot_id: int, bs: BotState, milestone: BoundMilestone):
        """Assign a milestone to a bot and generate operator."""
        bs.current_milestone = milestone.id
        milestone.assigned_bot = bot_id
        milestone.status = MilestoneStatus.IN_PROGRESS

        # Lock resources
        for res_id in milestone.resources:
            self.resources.try_lock(res_id, milestone.id, self.current_turn)

        # Generate operator
        op = self._get_op_for_milestone(bot_id, milestone)
        bs.current_op = op

    def _get_op_for_milestone(self, bot_id: int, milestone: BoundMilestone) -> Operator:
        """Generate operator for a milestone."""
        m_id = milestone.id
        ing = milestone.ingredient

        state = self.rc.get_bot_state(bot_id)
        holding = state.get('holding') if state else None

        if m_id == 'pan_ready':
            if holding and holding.get('type') == 'Pan':
                return op_place(self.kitchen_plan.cooker)
            return op_buy_pan(self.kitchen_plan.shop)

        elif m_id == 'plate_ready':
            if holding and holding.get('type') == 'Plate':
                return op_place(self.kitchen_plan.assembly_counter)
            # Try sinktable first
            if self.kitchen_plan.sinktable:
                tile = self.rc.get_tile(self.team, *self.kitchen_plan.sinktable)
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    return op_take_clean_plate(self.kitchen_plan.sinktable)
            return op_buy_plate(self.kitchen_plan.shop)

        elif '_bought' in m_id and ing:
            return op_buy_ingredient(ing, self.kitchen_plan.shop)

        elif '_chopped' in m_id and ing:
            # Check if food is on counter
            counter = self.kitchen_plan.chop_counter
            if counter:
                tile = self.rc.get_tile(self.team, *counter)
                if tile and hasattr(tile, 'item') and tile.item:
                    item = tile.item
                    if getattr(item, 'food_name', None) == ing:
                        if getattr(item, 'chopped', False):
                            return op_pickup(counter)
                        else:
                            return op_chop(counter)
            # If holding, place it
            if holding and holding.get('food_name') == ing:
                return op_place(counter)
            return op_wait()

        elif '_cooking_started' in m_id and ing:
            if self.kitchen_plan.cooker:
                return op_start_cook(self.kitchen_plan.cooker, ing)

        elif '_cooked' in m_id and ing:
            if self.kitchen_plan.cooker:
                return op_take_from_pan(self.kitchen_plan.cooker)

        elif '_on_plate' in m_id and ing:
            if holding and holding.get('food_name') == ing:
                return op_add_to_plate(self.kitchen_plan.assembly_counter, ing)
            return op_wait()

        elif m_id == 'submitted':
            if holding and holding.get('type') == 'Plate':
                return op_submit(self.kitchen_plan.submit)
            # Pickup plate
            if self.kitchen_plan.assembly_counter:
                return op_pickup(self.kitchen_plan.assembly_counter)

        return op_wait()

    def _execute_op(self, bot_id: int, op: Operator, bs: BotState):
        """Execute an operator."""
        if op.op_type in [OpType.WAIT, OpType.NOOP]:
            return

        # Check preconditions
        if not self.executor.can_execute(bot_id, op):
            bs.retry_count += 1
            return

        # Reserve target tile for movement
        if op.action_target:
            # Reserve a tile adjacent to target
            state = self.rc.get_bot_state(bot_id)
            if state:
                bx, by = state['x'], state['y']
                tx, ty = op.action_target
                # Find best adjacent tile
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        adj = (tx + dx, ty + dy)
                        if self._is_walkable(adj[0], adj[1]):
                            if self.resources.reserve_tile(adj, bot_id):
                                break

        # Execute
        moved, acted = self.executor.execute(bot_id, op)

        if not moved and not acted:
            bs.retry_count += 1
        else:
            bs.retry_count = 0

        # Check if milestone completed
        if acted and bs.current_milestone:
            self._check_milestone_complete(bs)

    def _check_milestone_complete(self, bs: BotState):
        """Check if current milestone is complete after action."""
        if self.order_plan is None or bs.current_milestone is None:
            return

        milestone = self.order_plan.milestones.get(bs.current_milestone)
        if milestone is None:
            return

        m_id = milestone.id

        # Milestone-specific completion checks
        if m_id == 'pan_ready':
            if self.kitchen_plan.cooker:
                tile = self.rc.get_tile(self.team, *self.kitchen_plan.cooker)
                if tile and hasattr(tile, 'item') and tile.item:
                    if str(type(tile.item).__name__) == 'Pan':
                        self.order_plan.mark_complete(m_id)

        elif m_id == 'plate_ready':
            if self.kitchen_plan.assembly_counter:
                tile = self.rc.get_tile(self.team, *self.kitchen_plan.assembly_counter)
                if tile and hasattr(tile, 'item') and tile.item:
                    if str(type(tile.item).__name__) == 'Plate':
                        self.order_plan.mark_complete(m_id)

        elif '_bought' in m_id:
            state = self.rc.get_bot_state(bs.bot_id)
            if state:
                holding = state.get('holding')
                if holding and holding.get('type') == 'Food':
                    if holding.get('food_name') == milestone.ingredient:
                        self.order_plan.mark_complete(m_id)

        elif '_chopped' in m_id:
            state = self.rc.get_bot_state(bs.bot_id)
            if state:
                holding = state.get('holding')
                if holding and holding.get('type') == 'Food':
                    if holding.get('food_name') == milestone.ingredient:
                        if holding.get('chopped'):
                            self.order_plan.mark_complete(m_id)

        elif '_cooking_started' in m_id:
            if self.kitchen_plan.cooker:
                tile = self.rc.get_tile(self.team, *self.kitchen_plan.cooker)
                if tile and hasattr(tile, 'item') and tile.item:
                    pan = tile.item
                    if hasattr(pan, 'food') and pan.food:
                        self.order_plan.mark_complete(m_id)

        elif '_cooked' in m_id:
            state = self.rc.get_bot_state(bs.bot_id)
            if state:
                holding = state.get('holding')
                if holding and holding.get('type') == 'Food':
                    if holding.get('food_name') == milestone.ingredient:
                        if holding.get('cooked_stage', 0) >= 1:
                            self.order_plan.mark_complete(m_id)

        elif '_on_plate' in m_id:
            # Check plate contents
            if self.kitchen_plan.assembly_counter:
                tile = self.rc.get_tile(self.team, *self.kitchen_plan.assembly_counter)
                if tile and hasattr(tile, 'item') and tile.item:
                    if str(type(tile.item).__name__) == 'Plate':
                        foods = getattr(tile.item, 'food', [])
                        for f in foods:
                            if getattr(f, 'food_name', None) == milestone.ingredient:
                                self.order_plan.mark_complete(m_id)
                                break

        elif m_id == 'submitted':
            # Check order completion
            orders = self.rc.get_orders(self.team)
            for o in orders:
                if o['order_id'] == self.order_plan.order_id:
                    if o.get('completed_turn') is not None:
                        self.order_plan.mark_complete(m_id)
                    break

        # Clear assignment if complete
        if milestone.status == MilestoneStatus.COMPLETE:
            bs.current_milestone = None
            bs.current_op = None
            milestone.assigned_bot = None
            self.resources.release_all_by(m_id)

    def _is_walkable(self, x: int, y: int) -> bool:
        """Check if tile is walkable."""
        if self._map_cache is None:
            return False
        if not (0 <= x < self._map_cache.width and 0 <= y < self._map_cache.height):
            return False
        return self._map_cache.is_tile_walkable(x, y)
