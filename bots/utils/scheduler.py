"""
Reactive Scheduler for Bot Task Assignment

Examines current game state and milestone progress to assign
the next best task to each bot.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import deque

# Import our modules (these would be copy-pasted in final submission)
try:
    from .milestones import (
        OrderProgress, Milestone, MilestoneStatus, Task, TaskType,
        prioritize_milestones, get_parallel_work, INGREDIENT_PROPS
    )
    from .orders import (
        select_order, get_active_orders, priority_profit_per_turn,
        INGREDIENT_COSTS, PLATE_COST, PAN_COST
    )
    from .bfs import (
        bfs_path_to_adjacent, bfs_find_tile_by_name, get_bot_position,
        get_bot_map_team, execute_next_move, is_adjacent_to, DIRECTIONS
    )
except ImportError:
    # For standalone testing
    from milestones import (
        OrderProgress, Milestone, MilestoneStatus, Task, TaskType,
        prioritize_milestones, get_parallel_work, INGREDIENT_PROPS
    )
    from orders import (
        select_order, get_active_orders, priority_profit_per_turn,
        INGREDIENT_COSTS, PLATE_COST, PAN_COST
    )
    from bfs import (
        bfs_path_to_adjacent, bfs_find_tile_by_name, get_bot_position,
        get_bot_map_team, execute_next_move, is_adjacent_to, DIRECTIONS
    )


# Constants
COOK_TIME = 20
BURN_TIME = 40


class BotScheduler:
    """
    Manages task assignment and execution for a team's bots.
    """

    def __init__(self, rc):
        self.rc = rc
        self.team = rc.get_team()
        self.current_order: Optional[Dict[str, Any]] = None
        self.progress: Optional[OrderProgress] = None

        # Caches (refreshed each turn)
        self._map_cache = None
        self._turn_cache = None

        # Track state
        self.pan_location: Optional[Tuple[int, int]] = None
        self.plate_location: Optional[Tuple[int, int]] = None
        self.cooking_start_turn: Optional[int] = None

    def refresh_cache(self):
        """Refresh per-turn caches."""
        self._turn_cache = self.rc.get_turn()
        self._map_cache = self.rc.get_map(self.team)

    def get_map(self):
        """Get cached map."""
        if self._map_cache is None:
            self._map_cache = self.rc.get_map(self.team)
        return self._map_cache

    def select_new_order(self, priority_fn=None):
        """Select a new order to work on."""
        if priority_fn is None:
            priority_fn = priority_profit_per_turn

        self.current_order = select_order(self.rc, priority_fn)

        if self.current_order:
            self.progress = OrderProgress(self.current_order)
            # Reset state tracking
            self.pan_location = None
            self.plate_location = None
            self.cooking_start_turn = None
        else:
            self.progress = None

        return self.current_order

    def is_order_complete(self) -> bool:
        """Check if current order is complete."""
        if self.progress is None:
            return True
        return self.progress.is_complete()

    def is_order_expired(self) -> bool:
        """Check if current order has expired."""
        if self.current_order is None:
            return True
        return self.rc.get_turn() > self.current_order.get('expires_turn', 0)

    # ==========================================
    # State inspection helpers
    # ==========================================

    def find_tile(self, tile_name: str) -> Optional[Tuple[int, int]]:
        """Find nearest tile of given type."""
        game_map = self.get_map()
        for x in range(game_map.width):
            for y in range(game_map.height):
                tile = game_map.tiles[x][y]
                if getattr(tile, 'tile_name', None) == tile_name:
                    return (x, y)
        return None

    def find_all_tiles(self, tile_name: str) -> List[Tuple[int, int]]:
        """Find all tiles of given type."""
        result = []
        game_map = self.get_map()
        for x in range(game_map.width):
            for y in range(game_map.height):
                tile = game_map.tiles[x][y]
                if getattr(tile, 'tile_name', None) == tile_name:
                    result.append((x, y))
        return result

    def get_tile_at(self, x: int, y: int):
        """Get tile at position."""
        return self.rc.get_tile(self.team, x, y)

    def get_bot_holding(self, bot_id: int) -> Optional[Dict]:
        """Get what a bot is holding."""
        state = self.rc.get_bot_state(bot_id)
        if state:
            return state.get('holding')
        return None

    def is_holding_nothing(self, bot_id: int) -> bool:
        """Check if bot is holding nothing."""
        return self.get_bot_holding(bot_id) is None

    def is_holding_type(self, bot_id: int, item_type: str) -> bool:
        """Check if bot is holding a specific item type."""
        holding = self.get_bot_holding(bot_id)
        if holding is None:
            return False
        return holding.get('type') == item_type

    def is_holding_ingredient(self, bot_id: int, ingredient: str) -> bool:
        """Check if bot is holding a specific ingredient."""
        holding = self.get_bot_holding(bot_id)
        if holding is None or holding.get('type') != 'Food':
            return False
        return holding.get('food_name') == ingredient

    def find_pan_on_cooker(self) -> Optional[Tuple[int, int]]:
        """Find a cooker that has a pan on it."""
        for pos in self.find_all_tiles('COOKER'):
            tile = self.get_tile_at(pos[0], pos[1])
            if tile and hasattr(tile, 'item') and tile.item is not None:
                if hasattr(tile.item, 'food') or str(type(tile.item).__name__) == 'Pan':
                    return pos
        return None

    def find_empty_cooker(self) -> Optional[Tuple[int, int]]:
        """Find a cooker without a pan."""
        for pos in self.find_all_tiles('COOKER'):
            tile = self.get_tile_at(pos[0], pos[1])
            if tile and (not hasattr(tile, 'item') or tile.item is None):
                return pos
        return None

    def find_empty_counter(self) -> Optional[Tuple[int, int]]:
        """Find an empty counter."""
        for pos in self.find_all_tiles('COUNTER'):
            tile = self.get_tile_at(pos[0], pos[1])
            if tile and (not hasattr(tile, 'item') or tile.item is None):
                return pos
        return None

    def get_cooker_state(self, pos: Tuple[int, int]) -> Dict[str, Any]:
        """Get detailed state of a cooker."""
        tile = self.get_tile_at(pos[0], pos[1])
        if tile is None:
            return {'has_pan': False}

        has_pan = hasattr(tile, 'item') and tile.item is not None
        food = None
        cook_progress = getattr(tile, 'cook_progress', 0)

        if has_pan and hasattr(tile.item, 'food'):
            food = tile.item.food

        cooked_stage = 0
        if food and hasattr(food, 'cooked_stage'):
            cooked_stage = food.cooked_stage

        return {
            'has_pan': has_pan,
            'has_food': food is not None,
            'cook_progress': cook_progress,
            'cooked_stage': cooked_stage,
            'is_cooked': cooked_stage == 1,
            'is_burnt': cooked_stage >= 2,
        }

    def can_afford(self, cost: int) -> bool:
        """Check if team can afford a cost."""
        return self.rc.get_team_money(self.team) >= cost

    # ==========================================
    # Task execution
    # ==========================================

    def execute_turn(self, bot_id: int) -> bool:
        """
        Execute one turn for a bot.
        Returns True if an action was taken.
        """
        # Ensure we have an order
        if self.current_order is None or self.is_order_complete() or self.is_order_expired():
            if not self.select_new_order():
                return False  # No orders available

        # Update milestone states based on current game state
        self.update_milestone_states()

        # Get assigned milestone for this bot
        milestone = self.progress.get_bot_milestone(bot_id)

        # If no milestone assigned, try to assign one
        if milestone is None or milestone.status == MilestoneStatus.COMPLETE:
            milestone = self.assign_next_milestone(bot_id)

        if milestone is None:
            # No work to do, maybe help with parallel work
            return self.do_parallel_work(bot_id)

        # Execute the milestone
        return self.execute_milestone(bot_id, milestone)

    def update_milestone_states(self):
        """Update milestone completion based on game state."""
        if self.progress is None:
            return

        # Check if we have pan on cooker
        pan_pos = self.find_pan_on_cooker()
        if pan_pos and 'have_pan' in self.progress.milestones:
            self.progress.mark_complete('have_pan')
            self.pan_location = pan_pos

        # Check cooking state
        if self.pan_location:
            cooker_state = self.get_cooker_state(self.pan_location)
            if cooker_state['is_cooked'] or cooker_state['is_burnt']:
                # Find which ingredient was cooking
                for m_id, m in self.progress.milestones.items():
                    if '_cooked' in m_id and m.status != MilestoneStatus.COMPLETE:
                        if cooker_state['is_cooked']:
                            self.progress.mark_complete(m_id)
                        break

    def assign_next_milestone(self, bot_id: int) -> Optional[Milestone]:
        """Assign the next best milestone to a bot."""
        if self.progress is None:
            return None

        # Get unassigned ready milestones
        ready = self.progress.get_unassigned_ready_milestones()
        if not ready:
            return None

        # Prioritize milestones
        prioritized = prioritize_milestones(ready)

        if prioritized:
            milestone = prioritized[0]
            self.progress.assign_milestone(bot_id, milestone.id)
            return milestone

        return None

    def execute_milestone(self, bot_id: int, milestone: Milestone) -> bool:
        """Execute work towards a milestone."""
        m_id = milestone.id

        # Route to specific handlers
        if m_id == 'have_pan':
            return self.execute_get_pan(bot_id)
        elif m_id == 'have_plate':
            return self.execute_get_plate(bot_id)
        elif '_bought' in m_id:
            return self.execute_buy_ingredient(bot_id, milestone.ingredient)
        elif '_chopped' in m_id:
            return self.execute_chop_ingredient(bot_id, milestone.ingredient)
        elif '_cooked' in m_id:
            return self.execute_cook_ingredient(bot_id, milestone.ingredient)
        elif '_on_plate' in m_id:
            return self.execute_add_to_plate(bot_id, milestone.ingredient)
        elif m_id == 'submitted':
            return self.execute_submit(bot_id)

        return False

    def do_parallel_work(self, bot_id: int) -> bool:
        """Do useful work while waiting (e.g., during cooking)."""
        if self.progress is None:
            return False

        parallel = get_parallel_work(self.progress)
        if parallel:
            milestone = parallel[0]
            self.progress.assign_milestone(bot_id, milestone.id)
            return self.execute_milestone(bot_id, milestone)

        return False

    # ==========================================
    # Specific task executors
    # ==========================================

    def move_adjacent_to(self, bot_id: int, target_x: int, target_y: int) -> bool:
        """Move bot adjacent to target. Returns True if already adjacent."""
        pos = get_bot_position(self.rc, bot_id)
        if pos is None:
            return False

        if is_adjacent_to(pos, (target_x, target_y)):
            return True

        path = bfs_path_to_adjacent(self.rc, bot_id, target_x, target_y)
        if path is None:
            return False

        if len(path) == 0:
            return True

        return execute_next_move(self.rc, bot_id, path)

    def execute_get_pan(self, bot_id: int) -> bool:
        """Get a pan and place it on a cooker."""
        holding = self.get_bot_holding(bot_id)

        # If holding pan, place on cooker
        if holding and holding.get('type') == 'Pan':
            cooker = self.find_empty_cooker() or self.find_tile('COOKER')
            if cooker:
                if self.move_adjacent_to(bot_id, cooker[0], cooker[1]):
                    if self.rc.place(bot_id, cooker[0], cooker[1]):
                        self.pan_location = cooker
                        self.progress.mark_complete('have_pan')
                        return True
            return False

        # Need to buy pan
        if not self.is_holding_nothing(bot_id):
            return False  # Hands full

        if not self.can_afford(PAN_COST):
            return False

        shop = self.find_tile('SHOP')
        if shop:
            if self.move_adjacent_to(bot_id, shop[0], shop[1]):
                from game_constants import ShopCosts
                return self.rc.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])

        return False

    def execute_get_plate(self, bot_id: int) -> bool:
        """Get a plate (buy or from sink table)."""
        holding = self.get_bot_holding(bot_id)

        # If holding plate, place on counter
        if holding and holding.get('type') == 'Plate':
            counter = self.find_empty_counter()
            if counter:
                if self.move_adjacent_to(bot_id, counter[0], counter[1]):
                    if self.rc.place(bot_id, counter[0], counter[1]):
                        self.plate_location = counter
                        self.progress.mark_complete('have_plate')
                        return True
            return False

        if not self.is_holding_nothing(bot_id):
            return False

        # Try sink table first (free clean plates)
        sinktable = self.find_tile('SINKTABLE')
        if sinktable:
            tile = self.get_tile_at(sinktable[0], sinktable[1])
            if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                if self.move_adjacent_to(bot_id, sinktable[0], sinktable[1]):
                    return self.rc.take_clean_plate(bot_id, sinktable[0], sinktable[1])

        # Buy plate
        if not self.can_afford(PLATE_COST):
            return False

        shop = self.find_tile('SHOP')
        if shop:
            if self.move_adjacent_to(bot_id, shop[0], shop[1]):
                from game_constants import ShopCosts
                return self.rc.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])

        return False

    def execute_buy_ingredient(self, bot_id: int, ingredient: str) -> bool:
        """Buy an ingredient from the shop."""
        if not self.is_holding_nothing(bot_id):
            return False

        cost = INGREDIENT_COSTS.get(ingredient, 0)
        if not self.can_afford(cost):
            return False

        shop = self.find_tile('SHOP')
        if shop:
            if self.move_adjacent_to(bot_id, shop[0], shop[1]):
                from game_constants import FoodType
                food_type = getattr(FoodType, ingredient, None)
                if food_type:
                    if self.rc.buy(bot_id, food_type, shop[0], shop[1]):
                        self.progress.mark_complete(f'{ingredient.lower()}_bought')
                        return True

        return False

    def execute_chop_ingredient(self, bot_id: int, ingredient: str) -> bool:
        """Chop an ingredient (place, chop, pickup)."""
        holding = self.get_bot_holding(bot_id)
        ing_lower = ingredient.lower()

        # If holding the ingredient, place it
        if self.is_holding_ingredient(bot_id, ingredient):
            counter = self.find_empty_counter()
            if counter:
                if self.move_adjacent_to(bot_id, counter[0], counter[1]):
                    return self.rc.place(bot_id, counter[0], counter[1])
            return False

        # Find ingredient on counter
        for pos in self.find_all_tiles('COUNTER'):
            tile = self.get_tile_at(pos[0], pos[1])
            if tile and hasattr(tile, 'item') and tile.item is not None:
                item = tile.item
                if hasattr(item, 'food_name') and item.food_name == ingredient:
                    if self.move_adjacent_to(bot_id, pos[0], pos[1]):
                        # Check if already chopped
                        if getattr(item, 'chopped', False):
                            # Pick it up
                            if self.rc.pickup(bot_id, pos[0], pos[1]):
                                self.progress.mark_complete(f'{ing_lower}_chopped')
                                return True
                        else:
                            # Chop it
                            return self.rc.chop(bot_id, pos[0], pos[1])

        return False

    def execute_cook_ingredient(self, bot_id: int, ingredient: str) -> bool:
        """Cook an ingredient."""
        ing_lower = ingredient.lower()

        # Check if already cooking
        if self.pan_location:
            cooker_state = self.get_cooker_state(self.pan_location)

            # If cooked, take from pan
            if cooker_state['is_cooked']:
                if self.is_holding_nothing(bot_id):
                    if self.move_adjacent_to(bot_id, self.pan_location[0], self.pan_location[1]):
                        if self.rc.take_from_pan(bot_id, self.pan_location[0], self.pan_location[1]):
                            self.progress.mark_complete(f'{ing_lower}_cooked')
                            return True
                return False

            # If burnt, handle error (trash and restart)
            if cooker_state['is_burnt']:
                # TODO: Error recovery
                return False

            # If cooking in progress, wait (do nothing or parallel work)
            if cooker_state['has_food']:
                return False  # Will trigger parallel work

        # Need to start cooking - must be holding the ingredient
        if self.is_holding_ingredient(bot_id, ingredient):
            cooker = self.pan_location or self.find_pan_on_cooker()
            if cooker:
                if self.move_adjacent_to(bot_id, cooker[0], cooker[1]):
                    # Use place to put food in pan (starts cooking automatically)
                    if self.rc.place(bot_id, cooker[0], cooker[1]):
                        self.cooking_start_turn = self.rc.get_turn()
                        return True

        return False

    def execute_add_to_plate(self, bot_id: int, ingredient: str) -> bool:
        """Add ingredient to plate."""
        ing_lower = ingredient.lower()

        # Need plate location
        if self.plate_location is None:
            # Find plate on counter
            for pos in self.find_all_tiles('COUNTER'):
                tile = self.get_tile_at(pos[0], pos[1])
                if tile and hasattr(tile, 'item'):
                    item = tile.item
                    if item and str(type(item).__name__) == 'Plate':
                        self.plate_location = pos
                        break

        if self.plate_location is None:
            return False

        # If holding the ingredient, add to plate
        if self.is_holding_ingredient(bot_id, ingredient):
            if self.move_adjacent_to(bot_id, self.plate_location[0], self.plate_location[1]):
                if self.rc.add_food_to_plate(bot_id, self.plate_location[0], self.plate_location[1]):
                    self.progress.mark_complete(f'{ing_lower}_on_plate')
                    return True

        return False

    def execute_submit(self, bot_id: int) -> bool:
        """Submit the completed plate."""
        holding = self.get_bot_holding(bot_id)

        # If holding plate, submit it
        if holding and holding.get('type') == 'Plate':
            submit = self.find_tile('SUBMIT')
            if submit:
                if self.move_adjacent_to(bot_id, submit[0], submit[1]):
                    if self.rc.submit(bot_id, submit[0], submit[1]):
                        self.progress.mark_complete('submitted')
                        return True
            return False

        # Need to pick up plate
        if self.plate_location and self.is_holding_nothing(bot_id):
            if self.move_adjacent_to(bot_id, self.plate_location[0], self.plate_location[1]):
                return self.rc.pickup(bot_id, self.plate_location[0], self.plate_location[1])

        return False
