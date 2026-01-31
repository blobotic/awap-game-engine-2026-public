"""
Order Evaluation and Selection Utilities

Provides functions to evaluate, filter, and select orders based on
pluggable priority functions.
"""

from typing import List, Dict, Any, Callable, Optional

# Ingredient costs (from game_constants.py)
INGREDIENT_COSTS = {
    'EGG': 20,
    'ONIONS': 30,
    'MEAT': 80,
    'NOODLES': 40,
    'SAUCE': 10,
}

# Ingredient properties
INGREDIENT_PROPS = {
    'EGG': {'can_chop': False, 'can_cook': True},
    'ONIONS': {'can_chop': True, 'can_cook': False},
    'MEAT': {'can_chop': True, 'can_cook': True},
    'NOODLES': {'can_chop': False, 'can_cook': False},
    'SAUCE': {'can_chop': False, 'can_cook': False},
}

# Equipment costs
PLATE_COST = 2
PAN_COST = 4


def get_order_ingredient_cost(order: Dict[str, Any]) -> int:
    """Calculate total ingredient cost for an order."""
    total = 0
    for ingredient in order['required']:
        total += INGREDIENT_COSTS.get(ingredient, 0)
    return total


def get_order_equipment_cost(order: Dict[str, Any]) -> int:
    """
    Calculate equipment cost for an order.
    Assumes we need 1 plate always, and 1 pan if any ingredient is cookable.
    """
    cost = PLATE_COST  # Always need a plate

    for ingredient in order['required']:
        props = INGREDIENT_PROPS.get(ingredient, {})
        if props.get('can_cook', False):
            cost += PAN_COST
            break  # Only need one pan

    return cost


def get_order_total_cost(order: Dict[str, Any], include_equipment: bool = True) -> int:
    """Calculate total cost to fulfill an order."""
    cost = get_order_ingredient_cost(order)
    if include_equipment:
        cost += get_order_equipment_cost(order)
    return cost


def get_order_profit(order: Dict[str, Any], include_equipment: bool = True) -> int:
    """Calculate expected profit (reward - cost) for an order."""
    return order['reward'] - get_order_total_cost(order, include_equipment)


def get_order_turns_remaining(order: Dict[str, Any], current_turn: int) -> int:
    """Calculate turns remaining before order expires."""
    return max(0, order['expires_turn'] - current_turn)


def estimate_order_completion_time(order: Dict[str, Any]) -> int:
    """
    Estimate minimum turns needed to complete an order.
    This is a rough estimate based on:
    - Travel time (approximate)
    - Cooking time (20 turns if any cookable)
    - Action time (chop, place, etc.)
    """
    base_time = 10  # Base travel and actions

    has_cookable = False
    has_choppable = False

    for ingredient in order['required']:
        props = INGREDIENT_PROPS.get(ingredient, {})
        if props.get('can_cook', False):
            has_cookable = True
        if props.get('can_chop', False):
            has_choppable = True

    time = base_time
    if has_cookable:
        time += 20  # Cooking time
    if has_choppable:
        time += 5  # Chopping overhead

    # Add time per ingredient
    time += len(order['required']) * 3

    return time


def can_afford_order(order: Dict[str, Any], money: int, include_equipment: bool = True) -> bool:
    """Check if we have enough money to start this order."""
    return money >= get_order_total_cost(order, include_equipment)


def is_order_feasible(order: Dict[str, Any], current_turn: int, money: int) -> bool:
    """
    Check if an order is feasible to complete.
    Must be active, affordable, and have enough time.
    """
    if not order.get('is_active', False):
        return False

    if order.get('completed_turn') is not None:
        return False

    if not can_afford_order(order, money):
        return False

    turns_remaining = get_order_turns_remaining(order, current_turn)
    estimated_time = estimate_order_completion_time(order)

    # Add some buffer (80% of remaining time should be enough)
    return turns_remaining >= estimated_time * 0.8


# ============================================
# Priority Functions
# ============================================

def priority_profit_per_turn(order: Dict[str, Any], context: Dict[str, Any]) -> float:
    """
    Priority = (reward - cost) / turns_remaining
    Higher is better. Classic efficiency metric.
    """
    current_turn = context.get('current_turn', 0)
    turns_remaining = get_order_turns_remaining(order, current_turn)

    if turns_remaining <= 0:
        return float('-inf')

    profit = get_order_profit(order)
    return profit / turns_remaining


def priority_profit_simple(order: Dict[str, Any], context: Dict[str, Any]) -> float:
    """
    Priority = reward - cost
    Simple profit maximization, ignores time pressure.
    """
    return get_order_profit(order)


def priority_urgency(order: Dict[str, Any], context: Dict[str, Any]) -> float:
    """
    Priority = 1 / turns_remaining
    Prioritizes orders about to expire (avoid penalties).
    """
    current_turn = context.get('current_turn', 0)
    turns_remaining = get_order_turns_remaining(order, current_turn)

    if turns_remaining <= 0:
        return float('-inf')

    return 1.0 / turns_remaining


def priority_profit_with_urgency(order: Dict[str, Any], context: Dict[str, Any]) -> float:
    """
    Priority = profit + (penalty / turns_remaining) * urgency_weight
    Balances profit with avoiding penalties.
    """
    current_turn = context.get('current_turn', 0)
    turns_remaining = get_order_turns_remaining(order, current_turn)

    if turns_remaining <= 0:
        return float('-inf')

    profit = get_order_profit(order)
    penalty = order.get('penalty', 0)
    urgency_weight = context.get('urgency_weight', 10)

    urgency_bonus = (penalty / turns_remaining) * urgency_weight
    return profit + urgency_bonus


def priority_weighted(order: Dict[str, Any], context: Dict[str, Any]) -> float:
    """
    Configurable weighted priority function.

    Context can include:
    - profit_weight: weight for profit component
    - urgency_weight: weight for urgency component
    - penalty_weight: weight for penalty avoidance
    - feasibility_bonus: bonus for easily completable orders
    """
    current_turn = context.get('current_turn', 0)
    money = context.get('money', 150)

    turns_remaining = get_order_turns_remaining(order, current_turn)
    if turns_remaining <= 0:
        return float('-inf')

    # Weights (can be tuned)
    profit_weight = context.get('profit_weight', 1.0)
    urgency_weight = context.get('urgency_weight', 0.5)
    penalty_weight = context.get('penalty_weight', 2.0)
    feasibility_bonus = context.get('feasibility_bonus', 50)

    # Components
    profit = get_order_profit(order)
    urgency = 100.0 / turns_remaining  # Higher when less time
    penalty_risk = order.get('penalty', 0) * (100.0 / turns_remaining)

    # Feasibility check
    feasible = 1.0 if is_order_feasible(order, current_turn, money) else 0.0

    score = (
        profit_weight * profit +
        urgency_weight * urgency +
        penalty_weight * penalty_risk +
        feasibility_bonus * feasible
    )

    return score


# ============================================
# Order Selection
# ============================================

def get_active_orders(rc, team=None) -> List[Dict[str, Any]]:
    """Get all active (incomplete, not expired) orders for a team."""
    if team is None:
        team = rc.get_team()

    orders = rc.get_orders(team)
    return [o for o in orders if o.get('is_active', False) and o.get('completed_turn') is None]


def select_order(
    rc,
    priority_fn: Callable[[Dict[str, Any], Dict[str, Any]], float] = priority_profit_per_turn,
    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Select the best order based on priority function.

    Args:
        rc: RobotController instance
        priority_fn: Function (order, context) -> float (higher is better)
        filter_fn: Optional function to filter orders (order -> bool)
        context: Additional context for priority function

    Returns:
        Best order dict, or None if no valid orders.
    """
    team = rc.get_team()
    current_turn = rc.get_turn()
    money = rc.get_team_money(team)

    # Build context
    ctx = {
        'current_turn': current_turn,
        'money': money,
        'team': team,
    }
    if context:
        ctx.update(context)

    # Get and filter orders
    orders = get_active_orders(rc, team)

    if filter_fn:
        orders = [o for o in orders if filter_fn(o)]

    if not orders:
        return None

    # Score and sort
    scored = [(priority_fn(o, ctx), o) for o in orders]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return best order (if score is valid)
    best_score, best_order = scored[0]
    if best_score == float('-inf'):
        return None

    return best_order


def select_orders_for_bots(
    rc,
    num_bots: int = 2,
    priority_fn: Callable[[Dict[str, Any], Dict[str, Any]], float] = priority_profit_per_turn,
    context: Optional[Dict[str, Any]] = None
) -> List[Optional[Dict[str, Any]]]:
    """
    Select orders for multiple bots, avoiding duplicates.

    Returns:
        List of orders (one per bot), with None for bots without orders.
    """
    team = rc.get_team()
    current_turn = rc.get_turn()
    money = rc.get_team_money(team)

    ctx = {
        'current_turn': current_turn,
        'money': money,
        'team': team,
    }
    if context:
        ctx.update(context)

    orders = get_active_orders(rc, team)
    if not orders:
        return [None] * num_bots

    # Score all orders
    scored = [(priority_fn(o, ctx), o) for o in orders]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Assign orders to bots (no duplicates)
    assigned = []
    used_order_ids = set()

    for i in range(num_bots):
        order_found = None
        for score, order in scored:
            if score == float('-inf'):
                break
            if order['order_id'] not in used_order_ids:
                order_found = order
                used_order_ids.add(order['order_id'])
                break
        assigned.append(order_found)

    return assigned
