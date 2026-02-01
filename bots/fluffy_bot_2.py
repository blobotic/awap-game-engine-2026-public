from collections import deque
import collections
from typing import Tuple, Optional, List, Dict
from enum import Enum

from game_constants import FoodType
from robot_controller import RobotController
from tiles import *


# ===============================================
# Ingredients
# ===============================================

ingredient_data = {"EGG": {"id": 0, "choppable": False, "cookable": True, "cost": 20, "shopItem": FoodType.EGG},
                   "ONION": {"id": 1, "choppable": True, "cookable": False, "cost": 30, "shopItem": FoodType.ONIONS},
                   "MEAT": {"id": 2, "choppable": True, "cookable": True, "cost": 80, "shopItem": FoodType.MEAT},
                   "NOODLES": {"id": 3, "choppable": False, "cookable": False, "cost": 40, "shopItem": FoodType.NOODLES},
                   "SAUCE": {"id": 4, "choppable": False, "cookable": False, "cost": 10, "shopItem": FoodType.SAUCE}}

class IngredientStatus(Enum):
    NOT_STARTED = 1
    BOUGHT = 2
    CHOPPED = 3
    COOKING = 4
    COOKED = 5
    READY = 6      # On counter, ready for plating
    PLATED = 7

class Ingredient:
    def __init__(self, name, order, index):
        self.name = name
        self.index = index
        self.id = ingredient_data[name]["id"]
        self.choppable = ingredient_data[name]["choppable"]
        self.cookable = ingredient_data[name]["cookable"]
        self.cost = ingredient_data[name]["cost"]
        self.shopItem = ingredient_data[name]["shopItem"]

        self.x = 0  # Location tracking (0,0 means bot is holding or not yet placed)
        self.y = 0

        self.status = IngredientStatus.NOT_STARTED
        self.order = order

    def __eq__(self, other):
        if not isinstance(other, Ingredient):
            return NotImplemented
        return self.order.id == other.order.id

    def __lt__(self, other):
        if not isinstance(other, Ingredient):
            return NotImplemented
        return self.order.id < other.order.id


# ===============================================
# Order
# ===============================================

class Order:
    def __init__(self, order):
        self.id = order["order_id"]
        self.abandoned = False
        self.completed = False
        self.reward = order["reward"]
        self.expires_turn = order["expires_turn"]
        self.ings = []

        i = 0
        for ing in order["required"]:
            self.ings.append(Ingredient(ing, self, i))
            i += 1

        self.active = False
        self.plate = None

    def all_plated(self):
        for ingredient in self.ings:
            if ingredient.status != IngredientStatus.PLATED:
                return False
        return True 

# ===============================================
# Tasks
# ===============================================

class Tasks(Enum):
    BUY_INGREDIENT = 1
    CHOP = 2
    COOK = 3  # start_cook - prep bot
    TAKE_FROM_COOKER = 4  # take_from_pan when cooked - cook bot
    STAGE = 5  # place ingredient on counter for assembly
    ACQUIRE_PLATE = 6
    ASSEMBLE = 7  # add ingredient to plate
    SUBMIT_PLATE = 8
    WASH_PLATE = 9

class Task:
    def __init__(self, task, ingredient, metadata, order, bot_id):
        self.task = task
        self.ingredient = ingredient
        self.metadata = metadata
        self.order = order
        self.bot_id = bot_id

    def get_closest_loc(self, bot_loc):
        """Get closest target location from nav_maps metadata."""
        if not self.metadata:
            return None
        dists = self.metadata.get("dists")
        if not dists:
            return None
        cell = dists[bot_loc[0]][bot_loc[1]]
        if not cell:
            return None
        return cell[0][1]  # (distance, (x, y)) -> (x, y)

    def get_closest_unclaimed_loc(self, bot_loc, controller):
        """Get closest target location that doesn't have an item on it."""
        if not self.metadata:
            return None
        dists = self.metadata.get("dists")
        if not dists:
            return None
        cell = dists[bot_loc[0]][bot_loc[1]]
        if not cell:
            return None

        # Iterate through sorted locations and find first unoccupied one
        for dist, (x, y) in cell:
            tile = controller.get_tile(controller.get_team(), x, y)
            tile_item = getattr(tile, "item", None)

            if tile_item is None:
                # Counter/tile with nothing on it
                return (x, y)

            # For Cooker: item is always a Pan, check if pan is empty (no food in it)
            # Pan has an 'item' attribute for the food being cooked
            pan_contents = getattr(tile_item, "item", None)
            if pan_contents is None:
                return (x, y)

        # All locations occupied, return closest anyway (will fail but retry later)
        return cell[0][1]

    def __str__(self):
        ing_name = self.ingredient.name if self.ingredient else "N/A"
        return f"Task({self.task.name}, {ing_name})"

    def __repr__(self):
        return self.__str__()

# ===============================================
# Bot
# ===============================================
class Bot:
    def __init__(self, bot_id, botplayer):
        self.id = bot_id
        self.botplayer = botplayer
        self.task = None

    def work(self, controller: RobotController):
        bot_state = controller.get_bot_state(self.id)
        bot_loc = (bot_state["x"], bot_state["y"])

        if self.task is None:
            return 
        elif self.task.task == Tasks.BUY_INGREDIENT:
            dest = self.task.get_closest_loc(bot_loc)
            if dest is None:
                return
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived and controller.get_team_money(team=controller.get_team()) >= self.task.ingredient.cost:
                if controller.buy(self.id, self.task.ingredient.shopItem, dest[0], dest[1]):
                    self.task.ingredient.status = IngredientStatus.BOUGHT
                    self.task = None
        elif self.task.task == Tasks.CHOP:
            ing = self.task.ingredient
            bot_state = controller.get_bot_state(self.id)
            holding = bot_state.get("holding")

            if holding:
                # Phase 1: We're holding the ingredient, need to place it on counter
                dest = self.task.get_closest_unclaimed_loc(bot_loc, controller)
                if dest is None:
                    return
                arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])
                if arrived:
                    if controller.place(self.id, dest[0], dest[1]):
                        ing.x, ing.y = dest[0], dest[1]
            else:
                # Phase 2: Ingredient is on counter, chop it
                arrived = self.botplayer.move_towards(controller, self.id, ing.x, ing.y)
                if arrived:
                    if controller.chop(self.id, ing.x, ing.y):
                        ing.status = IngredientStatus.CHOPPED
                        self.task = None
        elif self.task.task == Tasks.COOK:
            ing = self.task.ingredient
            bot_state = controller.get_bot_state(self.id)
            holding = bot_state.get("holding")

            if not holding and ing.x != 0 and ing.y != 0:
                # Phase 1: Ingredient is on counter (after chopping), need to pick it up
                arrived = self.botplayer.move_towards(controller, self.id, ing.x, ing.y)
                if arrived:
                    if controller.pickup(self.id, ing.x, ing.y):
                        ing.x, ing.y = 0, 0
            else:
                # Phase 2: We're holding the ingredient, move to cooker and start cooking
                dest = self.task.get_closest_unclaimed_loc(bot_loc, controller)
                if dest is None:
                    return
                arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])
                if arrived:
                    if controller.start_cook(self.id, dest[0], dest[1]):
                        ing.status = IngredientStatus.COOKING
                        ing.x, ing.y = dest[0], dest[1]
                        self.task = None
                
        elif self.task.task == Tasks.TAKE_FROM_COOKER:
            ing = self.task.ingredient
            bot_state = controller.get_bot_state(self.id)
            holding = bot_state.get("holding")

            # Can only take if hands are empty
            if holding:
                # TODO: need to put down what we're holding first
                return

            # Move to the cooker where the ingredient is cooking
            arrived = self.botplayer.move_towards(controller, self.id, ing.x, ing.y)
            if arrived:
                # Check cook progress
                cooker_tile = controller.get_tile(controller.get_team(), ing.x, ing.y)
                cook_progress = getattr(cooker_tile, "cook_progress", 0)

                # Take if cooked (20-39), avoid burnt (40+)
                if 20 <= cook_progress < 40:
                    if controller.take_from_pan(self.id, ing.x, ing.y):
                        ing.status = IngredientStatus.COOKED
                        ing.x, ing.y = 0, 0  # Now holding it
                        self.task = None
                elif cook_progress >= 40:
                    # Food is burnt - take it anyway and trash it (TODO: handle burnt)
                    if controller.take_from_pan(self.id, ing.x, ing.y):
                        ing.status = IngredientStatus.COOKED  # Will need to restart
                        ing.x, ing.y = 0, 0
                        self.task = None
                # else: still cooking, wait

        elif self.task.task == Tasks.STAGE:
            # Place ingredient on counter for assembly
            ing = self.task.ingredient
            dest = self.task.get_closest_unclaimed_loc(bot_loc, controller)
            if dest is None:
                return
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])
            if arrived:
                if controller.place(self.id, dest[0], dest[1]):
                    ing.x, ing.y = dest[0], dest[1]
                    ing.status = IngredientStatus.READY
                    self.task = None

        elif self.task.task == Tasks.ACQUIRE_PLATE:
            order = self.task.order
            PLATE_COST = 2

            if self.botplayer.plates_bought < 3 and controller.get_team_money(controller.get_team()) >= PLATE_COST:
                # Buy a plate from shop
                dest = self.task.get_closest_loc(bot_loc)
                if dest is None:
                    return
                arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])
                if arrived:
                    from game_constants import ShopCosts
                    if controller.buy(self.id, ShopCosts.PLATE, dest[0], dest[1]):
                        self.botplayer.plates_bought += 1
                        order.plate = "holding"
                        self.task = None
            else:
                # Try to get clean plate from SinkTable
                sink_table_nav = self.botplayer.nav_maps.get("SinkTable")
                if sink_table_nav:
                    cell = sink_table_nav["dists"][bot_loc[0]][bot_loc[1]]
                    if not cell:
                        return
                    dest = cell[0][1]
                    arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])
                    if arrived:
                        sink_table_tile = controller.get_tile(controller.get_team(), dest[0], dest[1])
                        if getattr(sink_table_tile, "num_clean_plates", 0) > 0:
                            if controller.take_clean_plate(self.id, dest[0], dest[1]):
                                order.plate = "holding"
                                self.task = None
                        # else: no clean plates, need to wash (TODO)

        elif self.task.task == Tasks.ASSEMBLE:
            # Add ingredient to plate
            ing = self.task.ingredient
            # Bot should be holding plate, ingredient is at ing.x, ing.y
            arrived = self.botplayer.move_towards(controller, self.id, ing.x, ing.y)
            if arrived:
                if controller.add_food_to_plate(self.id, ing.x, ing.y):
                    ing.status = IngredientStatus.PLATED
                    self.task = None

        elif self.task.task == Tasks.SUBMIT_PLATE:
            # Submit the completed plate
            dest = self.task.get_closest_loc(bot_loc)
            if dest is None:
                return
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])
            if arrived:
                if controller.submit(self.id, dest[0], dest[1]):
                    # Order completed!
                    self.task.order.completed = True
                    self.task = None

        else:
            print("self.task.task is ", self.task.task)
            raise(NotImplemented)


    def __str__(self):
        return f"Bot {self.id} - doing {self.task}"
    
    def __repr__(self):
        return self.__str__()



# ===============================================
# Bot Player
# ===============================================

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.parsed_map = False

        # Order tracking
        self.orders = {}  # order_id -> Order
        self.order_capacity = 2  # max orders to work on at once

        # Bot tracking
        self.bots = {}  # bot_id -> Bot
        self.prep_bot_id = None
        self.cook_bot_id = None

        # Plate tracking
        self.plates_bought = 0  # Buy plates until we have 3, then wash/reuse

    def refresh_orders(self, controller: RobotController):
        """Sync self.orders with game state. Create Order objects for new orders, remove completed/expired."""
        turn = controller.get_turn()
        game_orders = controller.get_orders(controller.get_team())
        game_order_map = {o["order_id"]: o for o in game_orders}

        # Remove completed/expired orders
        for oid in list(self.orders.keys()):
            game_order = game_order_map.get(oid)
            if game_order is None or game_order.get("completed_turn") is not None:
                del self.orders[oid]
                continue
            if not game_order.get("is_active") or turn > game_order.get("expires_turn", 0):
                del self.orders[oid]
                continue

        # Add new active orders (up to capacity)
        for game_order in game_orders:
            if len(self.orders) >= self.order_capacity:
                break
            oid = game_order["order_id"]
            if oid in self.orders:
                continue
            if not game_order.get("is_active") or game_order.get("completed_turn") is not None:
                continue
            if turn > game_order.get("expires_turn", 0):
                continue

            self.orders[oid] = Order(game_order)

    def get_bfs_path(self, controller: RobotController, start: Tuple[int, int], target_predicate) -> Optional[Tuple[int, int]]:
        queue = deque([(start, [])]) 
        visited = set([start])
        w, h = self.map.width, self.map.height

        while queue:
            (curr_x, curr_y), path = queue.popleft()
            tile = controller.get_tile(controller.get_team(), curr_x, curr_y)
            if target_predicate(curr_x, curr_y, tile):
                if not path: return (0, 0) 
                return path[0] 

            for dx in [0, -1, 1]:
                for dy in [0, -1, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
                            visited.add((nx, ny))
                            queue.append(((nx, ny), path + [(dx, dy)]))
        return None

    def move_towards(self, controller: RobotController, bot_id: int, target_x: int, target_y: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        bx, by = bot_state['x'], bot_state['y']
        def is_adjacent_to_target(x, y, tile):
            return max(abs(x - target_x), abs(y - target_y)) <= 1
        if is_adjacent_to_target(bx, by, None): return True
        step = self.get_bfs_path(controller, (bx, by), is_adjacent_to_target)
        if step and (step[0] != 0 or step[1] != 0):
            controller.move(bot_id, step[0], step[1])
            return False 
        return False 

    # ===== Map Parsing =====
    def parse_map(self, m) -> None:
        """
        Analyzes the map to generate flow fields for all relevant points of interest.
        Populates self.nav_maps where self.nav_maps[category] contains:
           - 'dists': 2D array of sorted lists of (distance, (target_x, target_y)) tuples
           - 'moves': 2D array of (dx, dy) tuples (direction to move to get closer)
        """
        self.parsed_map = True
        print("Parsing map flow fields...")
        self.nav_maps = {}

        # 1. Identify all relevant target locations by category
        relevant_categories = collections.defaultdict(list)

        width = m.width
        height = m.height

        for x in range(width):
            for y in range(height):
                tile = m.tiles[x][y]

                # Group generic stations
                if isinstance(tile, Counter):
                    relevant_categories["Counter"].append((x, y))
                elif isinstance(tile, Sink):
                    relevant_categories["Sink"].append((x, y))
                elif isinstance(tile, SinkTable):
                    relevant_categories["SinkTable"].append((x, y))
                elif isinstance(tile, Cooker):
                    relevant_categories["Cooker"].append((x, y))
                elif isinstance(tile, Submit):
                    relevant_categories["Submit"].append((x, y))
                elif isinstance(tile, Trash):
                    relevant_categories["Trash"].append((x, y))
                elif isinstance(tile, Shop):
                    relevant_categories["Shop"].append((x, y))
                elif isinstance(tile, Box):
                    relevant_categories["Box"].append((x, y))

        # 2. Generate Flow Field (BFS) for each category
        for category, targets in relevant_categories.items():
            if not targets:
                continue

            # Initialize grids - dists is now a 2D array of lists
            dist_matrix = [[[] for _ in range(height)] for _ in range(width)]
            move_matrix = [[{} for _ in range(height)] for _ in range(width)]  # Dictionary keyed by target

            # Run BFS from each target separately
            for target_idx, (tx, ty) in enumerate(targets):
                temp_dist = [[float('inf') for _ in range(height)] for _ in range(width)]
                queue = deque()

                # Seed BFS for this specific target
                temp_dist[tx][ty] = 0
                move_matrix[tx][ty][(tx, ty)] = (0, 0)
                queue.append((tx, ty))

                while queue:
                    curr_x, curr_y = queue.popleft()
                    current_dist = temp_dist[curr_x][curr_y]

                    # Check all 8 neighbors
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue

                            nx, ny = curr_x + dx, curr_y + dy

                            if 0 <= nx < width and 0 <= ny < height:
                                if temp_dist[nx][ny] == float('inf'):
                                    # Use is_tile_walkable to include Submit and other walkable tiles
                                    if m.is_tile_walkable(nx, ny):
                                        temp_dist[nx][ny] = current_dist + 1
                                        move_matrix[nx][ny][(tx, ty)] = (curr_x - nx, curr_y - ny)
                                        queue.append((nx, ny))

                # Add distance to this target to all reachable cells
                for x in range(width):
                    for y in range(height):
                        if temp_dist[x][y] != float('inf'):
                            dist_matrix[x][y].append((temp_dist[x][y], (tx, ty)))

            # Sort each list by distance
            for x in range(width):
                for y in range(height):
                    dist_matrix[x][y].sort(key=lambda item: item[0])

            # Convert move_matrix to use the closest target by default
            simple_move_matrix = [[None for _ in range(height)] for _ in range(width)]
            for x in range(width):
                for y in range(height):
                    if dist_matrix[x][y]:
                        closest_target = dist_matrix[x][y][0][1]
                        simple_move_matrix[x][y] = move_matrix[x][y].get(closest_target)

            self.nav_maps[category] = {
                "dists": dist_matrix,
                "moves": simple_move_matrix,
                "targets": targets
            }

    # ===== Task Generation =====

    def generate_tasks(self, controller: RobotController) -> List[Task]:
        """Generate tasks based on current ingredient and order statuses."""
        tasks = []

        for order in self.orders.values():
            # Generate ingredient-level tasks
            for ing in order.ings:
                task = self.get_next_task_for_ingredient(ing, controller)
                if task:
                    tasks.append(task)

            # Generate order-level tasks (plate acquisition, assembly, submit)
            order_tasks = self.get_order_tasks(order, controller)
            tasks.extend(order_tasks)

        return tasks

    def get_next_task_for_ingredient(self, ing: Ingredient, controller: RobotController) -> Optional[Task]:
        """Determine the next task needed for this ingredient based on its status."""
        if ing.status == IngredientStatus.NOT_STARTED:
            # Need to buy this ingredient
            if controller.get_team_money(controller.get_team()) >= ing.cost:
                return Task(Tasks.BUY_INGREDIENT, ing, self.nav_maps.get("Shop"), ing.order, None)

        elif ing.status == IngredientStatus.BOUGHT:
            # Just bought - what's next?
            if ing.choppable:
                return Task(Tasks.CHOP, ing, self.nav_maps.get("Counter"), ing.order, None)
            elif ing.cookable:
                # Not choppable but cookable (e.g., EGG) - go straight to cook
                return Task(Tasks.COOK, ing, self.nav_maps.get("Cooker"), ing.order, None)
            else:
                # Not choppable or cookable (NOODLES, SAUCE) - stage for plating
                return Task(Tasks.STAGE, ing, self.nav_maps.get("Counter"), ing.order, None)

        elif ing.status == IngredientStatus.CHOPPED:
            if ing.cookable:
                # Chopped and needs cooking (e.g., MEAT) - need to pickup and cook
                return Task(Tasks.COOK, ing, self.nav_maps.get("Cooker"), ing.order, None)
            else:
                # Chopped but not cookable (ONION) - already on counter, mark as ready
                ing.status = IngredientStatus.READY
                return None

        elif ing.status == IngredientStatus.COOKING:
            # Food is in the cooker - cook bot should monitor and take when ready
            return Task(Tasks.TAKE_FROM_COOKER, ing, None, ing.order, None)

        elif ing.status == IngredientStatus.COOKED:
            # Cooked - stage it on counter for assembly
            return Task(Tasks.STAGE, ing, self.nav_maps.get("Counter"), ing.order, None)

        # READY and PLATED don't need individual tasks - handled at order level
        return None

    def get_order_tasks(self, order: Order, controller: RobotController) -> List[Task]:
        """Generate order-level tasks: acquire plate, assemble, submit."""
        tasks = []

        # Count ingredient states
        ready_ings = [ing for ing in order.ings if ing.status == IngredientStatus.READY]
        plated_ings = [ing for ing in order.ings if ing.status == IngredientStatus.PLATED]
        all_ready = len(ready_ings) == len(order.ings)
        all_plated = len(plated_ings) == len(order.ings)

        if all_plated:
            # All ingredients on plate - submit!
            return [Task(Tasks.SUBMIT_PLATE, None, self.nav_maps.get("Submit"), order, None)]

        if order.plate == "holding" and ready_ings:
            # We have the plate, assemble next ready ingredient
            ing = ready_ings[0]
            return [Task(Tasks.ASSEMBLE, ing, None, order, None)]

        if all_ready and order.plate is None:
            # All ingredients ready, need to acquire plate
            return [Task(Tasks.ACQUIRE_PLATE, None, self.nav_maps.get("Shop"), order, None)]

        return tasks

    # ===== Task Assignment =====

    def assign_tasks(self, tasks: List[Task]):
        """Assign tasks to appropriate bots based on task type."""
        for task in tasks:
            bot = self.get_bot_for_task(task)
            if bot and bot.task is None:
                bot.task = task
                task.bot_id = bot.id

    def get_bot_for_task(self, task: Task) -> Optional[Bot]:
        """Return the appropriate bot for this task type (prep vs cook)."""
        # Prep bot handles: BUY, CHOP, COOK (start_cook), WASH_PLATE
        # Cook bot handles: TAKE_FROM_COOKER, STAGE (after cook), ASSEMBLE, SUBMIT, ACQUIRE_PLATE
        prep_tasks = {Tasks.BUY_INGREDIENT, Tasks.CHOP, Tasks.COOK, Tasks.WASH_PLATE}

        if task.task in prep_tasks:
            return self.bots.get(self.prep_bot_id)
        elif task.task == Tasks.STAGE:
            # STAGE depends on who's holding: prep bot stages NOODLES/SAUCE, cook bot stages COOKED items
            ing = task.ingredient
            if ing and (ing.choppable or ing.cookable):
                # Went through cooking process, cook bot has it
                return self.bots.get(self.cook_bot_id)
            else:
                # Simple ingredient (NOODLES/SAUCE), prep bot has it
                return self.bots.get(self.prep_bot_id)
        else:
            return self.bots.get(self.cook_bot_id)

    # ===== Main Loop =====

    def play_turn(self, controller: RobotController):
        # Parse map on first turn
        if not self.parsed_map:
            self.parse_map(controller.get_map(controller.get_team()))

        # Initialize bots on first turn
        my_bot_ids = controller.get_team_bot_ids(controller.get_team())
        if not my_bot_ids:
            return

        for bot_id in my_bot_ids:
            if bot_id not in self.bots:
                self.bots[bot_id] = Bot(bot_id, self)

        # Assign prep/cook roles (first bot = prep, second = cook)
        if self.prep_bot_id is None and len(my_bot_ids) >= 1:
            self.prep_bot_id = my_bot_ids[0]
        if self.cook_bot_id is None and len(my_bot_ids) >= 2:
            self.cook_bot_id = my_bot_ids[1]

        # Sync orders with game state
        self.refresh_orders(controller)

        # Generate tasks based on ingredient statuses
        tasks = self.generate_tasks(controller)

        # Assign tasks to idle bots
        self.assign_tasks(tasks)

        # Each bot executes their task
        for bot in self.bots.values():
            bot.work(controller)