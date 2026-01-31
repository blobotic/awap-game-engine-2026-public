"""
Milestone Bot - Uses reactive milestone-based scheduling

This bot uses the milestone system to decompose orders into
parallel-executable tasks and assigns them dynamically to bots.
"""

from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from game_constants import Team, TileType, FoodType, ShopCosts
from robot_controller import RobotController


# ============================================
# Configuration
# ============================================

# Ingredient costs
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

PLATE_COST = 2
PAN_COST = 4

# 8-directional movement
DIRECTIONS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]


# ============================================
# BFS Pathfinding
# ============================================

def bfs_path_to_adjacent(rc: RobotController, bot_id: int, target_x: int, target_y: int) -> Optional[List[Tuple[int, int]]]:
    """Find path to get adjacent to target."""
    state = rc.get_bot_state(bot_id)
    if state is None:
        return None

    start = (state['x'], state['y'])
    map_team_name = state.get('map_team', state['team'])
    map_team = Team.RED if map_team_name == 'RED' else Team.BLUE
    game_map = rc.get_map(map_team)

    # Already adjacent?
    if max(abs(start[0] - target_x), abs(start[1] - target_y)) <= 1:
        return []

    # BFS
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (cx, cy), path = queue.popleft()

        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy

            if (nx, ny) in visited:
                continue

            if not (0 <= nx < game_map.width and 0 <= ny < game_map.height):
                continue

            if not game_map.is_tile_walkable(nx, ny):
                continue

            new_path = path + [(dx, dy)]

            # Check if adjacent to target
            if max(abs(nx - target_x), abs(ny - target_y)) <= 1:
                return new_path

            visited.add((nx, ny))
            queue.append(((nx, ny), new_path))

    return None


def is_adjacent_to(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
    """Check if two positions are adjacent."""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= 1


# ============================================
# Order Utilities
# ============================================

def get_order_cost(order: Dict[str, Any]) -> int:
    """Calculate total cost for an order."""
    cost = PLATE_COST
    needs_pan = False

    for ing in order['required']:
        cost += INGREDIENT_COSTS.get(ing, 0)
        if INGREDIENT_PROPS.get(ing, {}).get('can_cook', False):
            needs_pan = True

    if needs_pan:
        cost += PAN_COST

    return cost


def priority_profit_per_turn(order: Dict[str, Any], current_turn: int) -> float:
    """Priority function: (reward - cost) / turns_remaining."""
    turns_remaining = max(1, order['expires_turn'] - current_turn)
    profit = order['reward'] - get_order_cost(order)
    return profit / turns_remaining


def select_best_order(rc: RobotController) -> Optional[Dict[str, Any]]:
    """Select the best order to work on."""
    team = rc.get_team()
    orders = rc.get_orders(team)
    current_turn = rc.get_turn()
    money = rc.get_team_money(team)

    # Filter active orders we can afford
    valid = []
    for o in orders:
        if not o.get('is_active', False):
            continue
        if o.get('completed_turn') is not None:
            continue
        if get_order_cost(o) > money:
            continue
        valid.append(o)

    if not valid:
        return None

    # Score and select best
    scored = [(priority_profit_per_turn(o, current_turn), o) for o in valid]
    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[0][1]


# ============================================
# Main Bot Class
# ============================================

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.current_order = None
        self.state = {}  # Per-bot state tracking
        self.pan_location = None
        self.plate_location = None
        self.cooking_start_turn = None

        # Track what each bot is doing
        self.bot_tasks = {}  # bot_id -> current task
        self.ingredients_bought = set()
        self.ingredients_chopped = set()
        self.ingredients_cooked = set()
        self.ingredients_plated = set()

    def play_turn(self, rc: RobotController):
        """Main turn execution."""
        team = rc.get_team()
        bot_ids = rc.get_team_bot_ids(team)

        if not bot_ids:
            return

        # Refresh map cache
        self.map = rc.get_map(team)

        # Check if we need a new order
        if self.current_order is None or self._is_order_done(rc):
            self._select_new_order(rc)

        if self.current_order is None:
            return  # No orders available

        # Update state based on game
        self._update_state(rc)

        # Execute turn for each bot
        for bot_id in bot_ids:
            self._execute_bot_turn(rc, bot_id)

    def _is_order_done(self, rc: RobotController) -> bool:
        """Check if current order is complete or expired."""
        if self.current_order is None:
            return True

        current_turn = rc.get_turn()
        if current_turn > self.current_order.get('expires_turn', 0):
            return True

        # Check if submitted
        orders = rc.get_orders(rc.get_team())
        for o in orders:
            if o['order_id'] == self.current_order['order_id']:
                if o.get('completed_turn') is not None:
                    return True

        return False

    def _select_new_order(self, rc: RobotController):
        """Select a new order to work on."""
        self.current_order = select_best_order(rc)

        # Reset tracking
        self.pan_location = None
        self.plate_location = None
        self.cooking_start_turn = None
        self.bot_tasks = {}
        self.ingredients_bought = set()
        self.ingredients_chopped = set()
        self.ingredients_cooked = set()
        self.ingredients_plated = set()

    def _update_state(self, rc: RobotController):
        """Update tracked state from game."""
        team = rc.get_team()

        # Find pan on cooker
        for x in range(self.map.width):
            for y in range(self.map.height):
                tile = rc.get_tile(team, x, y)
                if tile and getattr(tile, 'tile_name', None) == 'COOKER':
                    if hasattr(tile, 'item') and tile.item is not None:
                        self.pan_location = (x, y)

        # Find plate on counter
        for x in range(self.map.width):
            for y in range(self.map.height):
                tile = rc.get_tile(team, x, y)
                if tile and getattr(tile, 'tile_name', None) == 'COUNTER':
                    if hasattr(tile, 'item') and tile.item is not None:
                        if str(type(tile.item).__name__) == 'Plate':
                            self.plate_location = (x, y)

    def _execute_bot_turn(self, rc: RobotController, bot_id: int):
        """Execute a turn for one bot."""
        if self.current_order is None:
            return

        team = rc.get_team()
        state = rc.get_bot_state(bot_id)
        if state is None:
            return

        bx, by = state['x'], state['y']
        holding = state.get('holding')

        # Determine what this bot should do
        task = self._get_next_task(rc, bot_id)
        if task is None:
            return

        task_type, task_data = task

        # Execute the task
        if task_type == 'buy_pan':
            self._do_buy_pan(rc, bot_id)
        elif task_type == 'place_pan':
            self._do_place_pan(rc, bot_id)
        elif task_type == 'buy_plate':
            self._do_buy_plate(rc, bot_id)
        elif task_type == 'place_plate':
            self._do_place_plate(rc, bot_id)
        elif task_type == 'buy_ingredient':
            self._do_buy_ingredient(rc, bot_id, task_data)
        elif task_type == 'chop':
            self._do_chop(rc, bot_id, task_data)
        elif task_type == 'cook':
            self._do_cook(rc, bot_id, task_data)
        elif task_type == 'wait_cook':
            self._do_wait_cook(rc, bot_id)
        elif task_type == 'take_cooked':
            self._do_take_cooked(rc, bot_id, task_data)
        elif task_type == 'plate':
            self._do_plate(rc, bot_id, task_data)
        elif task_type == 'submit':
            self._do_submit(rc, bot_id)

    def _get_next_task(self, rc: RobotController, bot_id: int):
        """Determine the next task for a bot."""
        if self.current_order is None:
            return None

        team = rc.get_team()
        state = rc.get_bot_state(bot_id)
        holding = state.get('holding') if state else None
        ingredients = self.current_order['required']

        # Check if any ingredient needs cooking
        needs_cooking = any(
            INGREDIENT_PROPS.get(ing, {}).get('can_cook', False)
            for ing in ingredients
        )

        # Priority 1: If holding something, deal with it
        if holding:
            h_type = holding.get('type')

            if h_type == 'Pan':
                return ('place_pan', None)

            if h_type == 'Plate':
                # If plate has all ingredients, submit
                plate_foods = holding.get('food', [])
                plate_ing_names = {f.get('food_name') for f in plate_foods}
                if set(ingredients) == plate_ing_names:
                    return ('submit', None)
                return ('place_plate', None)

            if h_type == 'Food':
                food_name = holding.get('food_name')
                props = INGREDIENT_PROPS.get(food_name, {})

                # If needs chopping and not chopped
                if props.get('can_chop') and not holding.get('chopped'):
                    return ('chop', food_name)

                # If needs cooking and not cooked
                if props.get('can_cook') and holding.get('cooked_stage', 0) == 0:
                    return ('cook', food_name)

                # Otherwise, plate it
                return ('plate', food_name)

        # Priority 2: Get pan if needed and don't have one
        if needs_cooking and self.pan_location is None:
            return ('buy_pan', None)

        # Priority 3: Check cooking status
        if self.pan_location:
            tile = rc.get_tile(team, self.pan_location[0], self.pan_location[1])
            if tile and hasattr(tile, 'item') and tile.item:
                pan = tile.item
                if hasattr(pan, 'food') and pan.food:
                    cooked_stage = getattr(pan.food, 'cooked_stage', 0)
                    food_name = getattr(pan.food, 'food_name', None)
                    if cooked_stage == 1:  # Cooked, take it
                        return ('take_cooked', food_name)
                    elif cooked_stage == 0:  # Still cooking
                        # Do parallel work while waiting
                        pass

        # Priority 4: Get plate if don't have one
        if self.plate_location is None:
            return ('buy_plate', None)

        # Priority 5: Work on ingredients
        for ing in ingredients:
            if ing in self.ingredients_plated:
                continue

            props = INGREDIENT_PROPS.get(ing, {})

            # Need to buy?
            if ing not in self.ingredients_bought:
                return ('buy_ingredient', ing)

            # Need to chop?
            if props.get('can_chop') and ing not in self.ingredients_chopped:
                return ('chop', ing)

            # Need to cook?
            if props.get('can_cook') and ing not in self.ingredients_cooked:
                # Check if currently cooking
                if self.pan_location:
                    tile = rc.get_tile(team, self.pan_location[0], self.pan_location[1])
                    if tile and hasattr(tile, 'item') and tile.item:
                        pan = tile.item
                        if hasattr(pan, 'food') and pan.food:
                            return ('wait_cook', None)
                return ('cook', ing)

            # Need to plate?
            if ing not in self.ingredients_plated:
                return ('plate', ing)

        # Priority 6: Submit if all done
        if len(self.ingredients_plated) == len(ingredients):
            return ('submit', None)

        return None

    # ==========================================
    # Task Executors
    # ==========================================

    def _move_adjacent(self, rc: RobotController, bot_id: int, tx: int, ty: int) -> bool:
        """Move bot adjacent to target. Returns True if adjacent."""
        state = rc.get_bot_state(bot_id)
        if state is None:
            return False

        bx, by = state['x'], state['y']
        if is_adjacent_to((bx, by), (tx, ty)):
            return True

        path = bfs_path_to_adjacent(rc, bot_id, tx, ty)
        if path and len(path) > 0:
            dx, dy = path[0]
            rc.move(bot_id, dx, dy)
            return False

        return False

    def _find_tile(self, rc: RobotController, tile_name: str) -> Optional[Tuple[int, int]]:
        """Find a tile by name."""
        team = rc.get_team()
        for x in range(self.map.width):
            for y in range(self.map.height):
                tile = rc.get_tile(team, x, y)
                if tile and getattr(tile, 'tile_name', None) == tile_name:
                    return (x, y)
        return None

    def _find_empty_counter(self, rc: RobotController) -> Optional[Tuple[int, int]]:
        """Find an empty counter."""
        team = rc.get_team()
        for x in range(self.map.width):
            for y in range(self.map.height):
                tile = rc.get_tile(team, x, y)
                if tile and getattr(tile, 'tile_name', None) == 'COUNTER':
                    if not hasattr(tile, 'item') or tile.item is None:
                        return (x, y)
        return None

    def _do_buy_pan(self, rc: RobotController, bot_id: int):
        shop = self._find_tile(rc, 'SHOP')
        if shop and self._move_adjacent(rc, bot_id, shop[0], shop[1]):
            if rc.get_team_money(rc.get_team()) >= PAN_COST:
                rc.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])

    def _do_place_pan(self, rc: RobotController, bot_id: int):
        cooker = self._find_tile(rc, 'COOKER')
        if cooker and self._move_adjacent(rc, bot_id, cooker[0], cooker[1]):
            if rc.place(bot_id, cooker[0], cooker[1]):
                self.pan_location = cooker

    def _do_buy_plate(self, rc: RobotController, bot_id: int):
        # Try sink table first
        sinktable = self._find_tile(rc, 'SINKTABLE')
        if sinktable:
            tile = rc.get_tile(rc.get_team(), sinktable[0], sinktable[1])
            if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                if self._move_adjacent(rc, bot_id, sinktable[0], sinktable[1]):
                    rc.take_clean_plate(bot_id, sinktable[0], sinktable[1])
                    return

        shop = self._find_tile(rc, 'SHOP')
        if shop and self._move_adjacent(rc, bot_id, shop[0], shop[1]):
            if rc.get_team_money(rc.get_team()) >= PLATE_COST:
                rc.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])

    def _do_place_plate(self, rc: RobotController, bot_id: int):
        counter = self._find_empty_counter(rc)
        if counter and self._move_adjacent(rc, bot_id, counter[0], counter[1]):
            if rc.place(bot_id, counter[0], counter[1]):
                self.plate_location = counter

    def _do_buy_ingredient(self, rc: RobotController, bot_id: int, ingredient: str):
        shop = self._find_tile(rc, 'SHOP')
        if shop and self._move_adjacent(rc, bot_id, shop[0], shop[1]):
            cost = INGREDIENT_COSTS.get(ingredient, 0)
            if rc.get_team_money(rc.get_team()) >= cost:
                food_type = getattr(FoodType, ingredient, None)
                if food_type and rc.buy(bot_id, food_type, shop[0], shop[1]):
                    self.ingredients_bought.add(ingredient)

    def _do_chop(self, rc: RobotController, bot_id: int, ingredient: str):
        state = rc.get_bot_state(bot_id)
        holding = state.get('holding') if state else None

        # If holding ingredient, place it
        if holding and holding.get('food_name') == ingredient:
            counter = self._find_empty_counter(rc)
            if counter and self._move_adjacent(rc, bot_id, counter[0], counter[1]):
                rc.place(bot_id, counter[0], counter[1])
            return

        # Find ingredient on counter and chop/pickup
        team = rc.get_team()
        for x in range(self.map.width):
            for y in range(self.map.height):
                tile = rc.get_tile(team, x, y)
                if tile and getattr(tile, 'tile_name', None) == 'COUNTER':
                    if hasattr(tile, 'item') and tile.item:
                        if getattr(tile.item, 'food_name', None) == ingredient:
                            if self._move_adjacent(rc, bot_id, x, y):
                                if getattr(tile.item, 'chopped', False):
                                    if rc.pickup(bot_id, x, y):
                                        self.ingredients_chopped.add(ingredient)
                                else:
                                    rc.chop(bot_id, x, y)
                            return

    def _do_cook(self, rc: RobotController, bot_id: int, ingredient: str):
        if self.pan_location is None:
            return

        state = rc.get_bot_state(bot_id)
        holding = state.get('holding') if state else None

        # Must be holding the ingredient
        if holding and holding.get('food_name') == ingredient:
            if self._move_adjacent(rc, bot_id, self.pan_location[0], self.pan_location[1]):
                rc.place(bot_id, self.pan_location[0], self.pan_location[1])

    def _do_wait_cook(self, rc: RobotController, bot_id: int):
        # While waiting, could do other tasks
        # For now, just wait near cooker
        if self.pan_location:
            self._move_adjacent(rc, bot_id, self.pan_location[0], self.pan_location[1])

    def _do_take_cooked(self, rc: RobotController, bot_id: int, ingredient: str):
        if self.pan_location is None:
            return

        state = rc.get_bot_state(bot_id)
        holding = state.get('holding') if state else None

        if holding is None:
            if self._move_adjacent(rc, bot_id, self.pan_location[0], self.pan_location[1]):
                if rc.take_from_pan(bot_id, self.pan_location[0], self.pan_location[1]):
                    if ingredient:
                        self.ingredients_cooked.add(ingredient)

    def _do_plate(self, rc: RobotController, bot_id: int, ingredient: str):
        if self.plate_location is None:
            return

        state = rc.get_bot_state(bot_id)
        holding = state.get('holding') if state else None

        if holding and holding.get('food_name') == ingredient:
            if self._move_adjacent(rc, bot_id, self.plate_location[0], self.plate_location[1]):
                if rc.add_food_to_plate(bot_id, self.plate_location[0], self.plate_location[1]):
                    self.ingredients_plated.add(ingredient)

    def _do_submit(self, rc: RobotController, bot_id: int):
        state = rc.get_bot_state(bot_id)
        holding = state.get('holding') if state else None

        # If not holding plate, pick it up
        if not holding or holding.get('type') != 'Plate':
            if self.plate_location:
                if self._move_adjacent(rc, bot_id, self.plate_location[0], self.plate_location[1]):
                    rc.pickup(bot_id, self.plate_location[0], self.plate_location[1])
            return

        # Submit
        submit = self._find_tile(rc, 'SUBMIT')
        if submit and self._move_adjacent(rc, bot_id, submit[0], submit[1]):
            rc.submit(bot_id, submit[0], submit[1])
