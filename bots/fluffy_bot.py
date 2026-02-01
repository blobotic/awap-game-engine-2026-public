import random
from collections import deque
import collections
from dataclasses import dataclass, field
from email.policy import default
from typing import Tuple, Optional, List, Dict

from game_constants import Team, TileType, FoodType, ShopCosts
from robot_controller import RobotController
from item import Pan, Plate, Food
from enum import Enum
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
    FINISHED = 2
    BOUGHT = 3
    CHOPPED = 4
    COOKING = 5
    COOKED = 6
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

        self.bought = False
        self.cook_progress = 0
        self.plated = False
        self.cooked = False
        self.ready_to_cook = False
        self.cooking = False
        self.chopped = False
        self.bought = False
        self.x = 0
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

@dataclass
class OrderQueue:
    capacity: int
    active: Dict[int, Order] = field(default_factory=dict)

    # Used as a prio function for orders
    # Todo make this useful
    def _score(self, order, turn):
        return 1

    def refresh(self, rc):
        turn = rc.get_turn()
        orders = rc.get_orders()
        order_ids = {o["order_id"]: o for o in orders if o["is_active"]}

        # Update active orders dict
        for oid, order in list(self.active.items()):
            o = order_ids.get(oid)
            if o is None:
                order.abandoned = True
                del self.active[oid]
                continue
            if o.get("completed_turn") is not None:
                order.completed = True
                del self.active[oid]
                continue
            if turn > o.get("expires_turn", 0):
                order.abandoned = True
                del self.active[oid]
                continue
            order.reward = o["reward"]
            order.expires_turn = o["expires_turn"]

        if len(self.active) >= self.capacity:
            return
        to_add = self.capacity - len(self.active)
        # New orders to work on
        candidates = []
        for o in orders:
            if not o.get("is_active"):
                continue
            if o.get("completed_turn") is not None:
                continue
            oid = o["order_id"]
            if oid in self.active:
                continue
            if turn > o.get("expires_turn", 0):
                continue
            candidates.append(o)

        candidates.sort(key=lambda o: self._score(o, turn), reverse=True)

        # Todo create order objects for candidates, add them to active orders
# ===============================================
# Tasks
# ===============================================

class Tasks(Enum):
    BUY_INGREDIENT = 1
    CHOP = 2
    COOK = 3
    GOTO_PLATE = 5
    ACQUIRE_PLATE = 6
    SUBMIT_PLATE = 7
    ASSEMBLE = 8
    WASH_PLATE = 9

class Task:
    def __init__(self, task, ingredient, metadata, order, bot_id):
        self.task = task
        self.ingredient = ingredient
        self.metadata = metadata
        self.order = order
        self.bot_id = bot_id
        self.target_x = -1
        self.target_y = -1

    def get_closest_loc(self, bot_loc):
        return self.metadata["dists"][bot_loc[0]][bot_loc[1]][0][1]
    
    def get_closest_unclaimed_loc(self, bot_loc):
        # TODO: implement
        return self.metadata["dists"][bot_loc[0]][bot_loc[1]][0][1]


    def __str__(self):
        metadata_format = f"{self.metadata}"
        return f"{self.task} for {self.ingredient.name} with {metadata_format:.20}"
    
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

    # Bot will procure a plate
    # Decides whether to buy a plate or get a clean one
    def select_plate(self):
        pass

    # Bot chooses best location to chop food
    def select_chopping_counter(self):
        pass

    def work(self, controller : RobotController):
        bot_state = controller.get_bot_state(self.id)
        bot_loc = (bot_state["x"], bot_state["y"])

        if self.task is None:
            return 
        elif self.task.task == Tasks.BUY_INGREDIENT:
            dest = self.task.get_closest_loc(bot_loc)
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived and controller.get_team_money(team=controller.get_team()) >= self.task.ingredient.cost:
                # only buy if we have money
                print("Self.task is", self.task)
                if controller.buy(self.id, self.task.ingredient.shopItem, dest[0], dest[1]):
                    print("ingredient status set to bought!!")
                    self.task.ingredient.status = IngredientStatus.BOUGHT
                    self.task = None
        elif self.task.task == Tasks.CHOP:
            dest = self.task.get_closest_unclaimed_loc(bot_loc)
            # TODO: claim loc
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived:
                # start chopping 
                if controller.chop(self.id, dest[0], dest[1]):
                    self.task.ingredient.status = IngredientStatus.CHOPPED
                    self.task = None
                    # TODO: unclaim loc
        elif self.task.task == Tasks.COOK:
            dest = self.task.get_closest_unclaimed_loc(bot_loc)
            # TODO: claim loc
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived:
                # start chopping 
                if controller.start_cook(self.id, dest[0], dest[1]):
                    self.task.ingredient.status = IngredientStatus.COOKING
                    self.task = None
                    # TODO: unclaim loc
                
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
        self.assembly_counter = None 
        self.cooker_loc = None
        self.my_bot_id = None

        self.order_queue = OrderQueue(capacity=3)
        
        self.state = 0

        # new things
        self.orders = {}
        self.bots = {}

        self.parsed_map = False

        self.plates = []

    # Inspects orders list and assigns a raw and cooked staging area
    def assign_staging(self):
        pass

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

    def find_nearest_tile(self, controller: RobotController, bot_x: int, bot_y: int, tile_name: str) -> Optional[Tuple[int, int]]:
        best_dist = 9999
        best_pos = None
        m = controller.get_map(controller.get_team())
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == tile_name:
                    dist = max(abs(bot_x - x), abs(bot_y - y))
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (x, y)
        return best_pos



    # my functions below
    def parse_map(self, m) -> None:
        """
        Analyzes the map to generate flow fields for all relevant points of interest.
        Populates self.nav_maps where self.nav_maps[category] contains:
           - 'dists': 2D array of sorted lists of (distance, (target_x, target_y)) tuples
           - 'moves': 2D array of (dx, dy) tuples (direction to move to get closer)
        """
        self.parsed_map = True
        print(f"[{self.state}] Parsing map flow fields...")
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
                                    neighbor_tile = m.tiles[nx][ny]
                                    tile_name = getattr(neighbor_tile, "tile_name", "")

                                    if tile_name == "FLOOR":
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

    def calculate_cost(self, ingredient_list):
        cost = 0
        for ing in ingredient_list:
            cost += ingredient_data[ing]["cost"]

        return cost
    

    # orders are [{'order_id': 1, 'required': ['NOODLES', 'MEAT'], 'created_turn': 0, 'expires_turn': 200, 'reward': 10000, 'penalty': 3, 'claimed_by': None, 'completed_turn': None, 'is_active': True}]
    def prioritize_ingredients(self, controller):
        orders = controller.get_orders(controller.get_team())

        ingredients = []

        for order in orders:
            priority = 1
            if not order["is_active"]:
                continue
        
            oid = order["order_id"]
            if oid not in self.orders:
                self.orders[oid] = Order(order)
            elif self.orders[oid].active:
                # we care more about orders that we already started
                priority += 10

            # calculate priority based on reward / penalty
            reward = order["reward"] - self.calculate_cost(order["required"])
            penalty = order["penalty"]
            priority += reward - penalty

            # TODO: calculate if it's possible to complete in the remaining turns
            # for now, we will just prioritize by how many turns are left
            turn = controller.get_turn()
            remaining_turns = order["expires_turn"] - turn
            priority += 1000 / remaining_turns

            # todo: modify priority by distance to bots

            for ing in self.orders[oid].ings:
                if ing.status != IngredientStatus.FINISHED:
                    ingredients.append((priority, ing))

        ingredients.sort()
        return ingredients
    
    def generate_tasks(self, controller, ingredient_list):
        task_list = []

        # TODO: generate tasks for buying plates/pans and 
        # moving them if not generated yet

        # try to generate the next task for submitting orders
        for order_id in self.orders:
            order = self.orders[order_id]

            if order.all_plated():
                task_list.append((priority, Task(Tasks.SUBMIT_PLATE)))

        # try to generate the next task for all the ingredients
        for (priority, ingredient) in ingredient_list:
            if ingredient.status == IngredientStatus.NOT_STARTED:
                # buy only if we have enough money
                # TODO: do some handling for when bots are on their way to the shop!
                if controller.get_team_money(team=controller.get_team()) >= ingredient.cost:
                    task_list.append((priority, Task(Tasks.BUY_INGREDIENT, ingredient, self.nav_maps["Shop"])))
            elif ingredient.status == IngredientStatus.BOUGHT:
                # case on the ingredient
                if ingredient.choppable:
                    task_list.append((priority, Task(Tasks.CHOP, ingredient, self.nav_maps["Counter"])))
                elif ingredient.cookable: 
                    task_list.append((priority, Task(Tasks.COOK, ingredient, self.nav_maps["Cooker"])))
                else:
                    # put it on a plate
                    if ingredient.order.plate != None:
                        task_list.append((priority, Task(Tasks.GOTO_PLATE, ingredient, ingredient.order.plate)))
                    else:
                        # the order has no assigned plate yet
                        task_list.append((priority, Task(Tasks.ACQUIRE_PLATE, ingredient, self.plates)))
            elif ingredient.status == IngredientStatus.CHOPPED:
                if ingredient.cookable:
                    task_list.append((priority, Task(Tasks.COOK, ingredient, self.cookers)))
                else:
                    # put it on a plate
                    if ingredient.order.plate != None:
                        task_list.append((priority, Task(Tasks.GOTO_PLATE, ingredient, ingredient.order.plate)))
                    else:
                        # the order has no assigned plate yet
                        task_list.append((priority, Task(Tasks.ACQUIRE_PLATE, ingredient, self.plates)))
            elif ingredient.status == IngredientStatus.COOKING:
                # if controller.item_to_public_dict(ingredient.item)["cooked_stage"] == 1:
                # check if by the time you walk there it will be cooked
                    continue
            elif ingredient.status == IngredientStatus.COOKED:
                # put it on a plate
                if ingredient.order.plate != None:
                    task_list.append((priority, Task(Tasks.GOTO_PLATE, ingredient, ingredient.order.plate)))
                else:
                    # the order has no assigned plate yet
                    task_list.append((priority, Task(Tasks.ACQUIRE_PLATE, ingredient, self.plates)))
            

        return task_list

    def assign_bot(self, bot, task_list):
        # todo: make this functional
        # naive algorithm: just assign in order
        # future: 
        # goto tasks -> distance changes priority so search for a close one
        for i, (priority, task) in enumerate(task_list):
            if priority > 0:
                bot.task = task
                task_list[i] = (-1, None)
                return 
            

    def play_turn(self, controller: RobotController):
        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots: return

        # Refresh the order queue
        self.order_queue.refresh(controller)
        active_orders = self.order_queue.list_active()
        if not active_orders:
            return

        # Bot task assignment loop, heavy pseudocode for now
        for bot in my_bots:
            if bot.task is None:
                # in next task function, wherever it is
                # Loop through its ingredients, and do the next possible task:
                # Buy, chop, cook, plate, submit etc
                bot.task = next_task()
                done = bot.work(controller)
                if done:
                    bot.task = None

        # parse map
        if not self.parsed_map:
            self.parse_map(controller.get_map(controller.get_team()))

        # update bots
        for bot_id in my_bots:
            if bot_id not in self.bots:
                self.bots[bot_id] = Bot(bot_id, self)

        # based on orders, get ingredients in order of priority of what needs to be done
        ingredient_list = self.prioritize_ingredients(controller)
        task_list = self.generate_tasks(controller, ingredient_list)

        print(f"task list is {task_list}")
        
        # assign idle bots to do ingredients / tasks
        print("    self.bots is", self.bots)
        for bot_id in self.bots:
            bot = self.bots[bot_id]
            if bot.task is None:
                self.assign_bot(bot, task_list)

            # all bots do what they're assigned to do
            bot.work(controller)