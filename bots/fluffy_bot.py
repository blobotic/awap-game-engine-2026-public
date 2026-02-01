import random
from collections import deque
import collections
from typing import Tuple, Optional, List

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
    BOUGHT_PLATE = 8
    WASHING = 9

class Ingredient:
    def __init__(self, name, order, index):
        self.name = name 
        self.index = index
        self.id = ingredient_data[name]["id"]
        self.choppable = ingredient_data[name]["choppable"]
        self.cookable = ingredient_data[name]["cookable"]
        self.cost = ingredient_data[name]["cost"]
        self.shopItem = ingredient_data[name]["shopItem"]
        
        self.status = IngredientStatus.NOT_STARTED
        self.order = order
        self.preplate_status = IngredientStatus.NOT_STARTED
        self.working = None # which bot is working on this ingredient currently
        self.loc = None

    def __eq__(self, other):
        if not isinstance(other, Ingredient):
            return NotImplemented
        return self.order.id == other.order.id

    def __lt__(self, other):
        if not isinstance(other, Ingredient):
            return NotImplemented
        return self.order.id < other.order.id



# ===============================================
# Plate
# ===============================================
class Plate:
    def __init__(self):
        self.claimed_by = None 
        self.loc = None
        self.shopItem = None
        

    def is_free(self):
        return self.claimed_by == None 
    
    def is_dirty(self):
        print("self.shopItem is", self.shopItem)
        return False


# ===============================================
# Order
# ===============================================

class Order:
    def __init__(self, order, botplayer):
        self.id = order["order_id"] 
        self.ings = []
        self.botplayer = botplayer

        i = 0
        for ing in order["required"]:
            self.ings.append(Ingredient(ing, self, i))
            i += 1

        self.active = False
        self.plate = None
        self.submitting = False

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
    COOK = 3
    PLATE = 5
    ACQUIRE_PLATE = 6
    SUBMIT_PLATE = 7
    WASH_PLATE = 8
    MOVE_PLATE_TO_COUNTER = 9

class Task:
    def __init__(self, task, ingredient, metadata):
        self.task = task
        self.ingredient = ingredient
        self.metadata = metadata 

    def get_closest_loc(self, bot_loc):
        return self.metadata["dists"][bot_loc[0]][bot_loc[1]][0][1]
    
    def get_closest_unclaimed_loc(self, bot_loc):
        # TODO: implement
        for el in self.metadata["dists"][bot_loc[0]][bot_loc[1]]:
            print("el[1] is", el[1])
            if el[1] not in self.ingredient.order.botplayer.used_counters:
                return el[1]
        return (-1, -1)

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

    def holding(self, controller):
        bot_state = controller.get_bot_state(self.id)
        print("bot_state is", bot_state)
        return bot_state["holding"]

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
                    self.task.ingredient.item = controller.get_bot_state(self.id)["holding"]
                    self.task.ingredient.status = IngredientStatus.BOUGHT
                    self.task.ingredient.working = None
                    self.task = None
        elif self.task.task == Tasks.CHOP:
            dest = self.task.get_closest_unclaimed_loc(bot_loc)
            # TODO: claim loc
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived:
                # first try to put it down
                if self.holding(controller):
                    controller.place(self.id, dest[0], dest[1])
                # start chopping 
                elif controller.chop(self.id, dest[0], dest[1]):
                    print("chopping!!!")
                    self.task.ingredient.status = IngredientStatus.CHOPPED
                    self.task.ingredient.loc = (dest[0], dest[1])
                    self.task.ingredient.working = None
                    self.task = None
                    # TODO: unclaim loc
        elif self.task.task == Tasks.COOK:
            # cooker loc
            dest = self.task.get_closest_unclaimed_loc(bot_loc)

            # first try to pick up the ingredient
            cooker_tile = controller.get_tile(controller.get_team(), dest[0], dest[1])
            cooker_has_food = cooker_tile and hasattr(cooker_tile, "item") and cooker_tile.item is not None and cooker_tile.item.food is not None
            print("cooker tile is", cooker_tile.item)
            print("cooker tile is", cooker_tile.item.food)
            if not cooker_has_food and not self.holding(controller):
                # go to the ingredient
                ing_dest = self.task.ingredient.loc
                arrived = self.botplayer.move_towards(controller, self.id, ing_dest[0], ing_dest[1])
                if not arrived:
                    return 
                # then try to pick the ingredient up 
                if controller.pickup(self.id, ing_dest[0], ing_dest[1]):
                    return
                else:
                    raise "Ingredient location is not good"

            # then try to go to the cooker
            # TODO: claim loc
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived:
                # start cooking 
                if controller.start_cook(self.id, dest[0], dest[1]):
                    self.task.ingredient.status = IngredientStatus.COOKING
                    self.task.ingredient.loc = (dest[0], dest[1])
                    self.task.ingredient.working = None
                    self.task = None
                    # TODO: unclaim loc
        elif self.task.task == Tasks.ACQUIRE_PLATE:
            # TODO: change to be more functional
            # buy a plate
            dest = self.task.get_closest_loc(bot_loc)
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived and controller.get_team_money(team=controller.get_team()) >= ShopCosts.PLATE.buy_cost:
                print("trying to buy plate!!!")
                # only buy if we have money
                if controller.buy(self.id, ShopCosts.PLATE, dest[0], dest[1]):
                    print("bought plate!")
                    # make new plate and give it to the order
                    plate = Plate()
                    self.task.ingredient.order.plate = plate
                    self.botplayer.plates.append(plate)

                    self.task.ingredient.status = IngredientStatus.BOUGHT_PLATE
                    self.task.ingredient.working = None
                    self.task = None
        elif self.task.task == Tasks.WASH_PLATE:
            dest = self.task.get_closest_loc(bot_loc)
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived:
                # start washing the plate
                if controller.wash_sink(self.id, dest[0], dest[1]):
                    self.task.ingredient.status = IngredientStatus.WASHING
                    self.task.ingredient.working = None
                    self.task = None 

                    # give the plate to the order

        elif self.task.task == Tasks.MOVE_PLATE_TO_COUNTER:
            dest = self.task.get_closest_unclaimed_loc(bot_loc)
            print("MOVE_PLATE_TO_COUNTER dest is ", dest)
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived:
                # place on counter
                if controller.place(self.id, dest[0], dest[1]):
                    self.task.ingredient.status = self.task.ingredient.preplate_status
                    self.task.ingredient.order.plate.loc = (dest[0], dest[1])
                    self.botplayer.used_counters[(dest[0], dest[1])] = True
                    self.task.ingredient.working = None
                    self.task = None

        elif self.task.task == Tasks.PLATE:
            # metadata dictates if we need to pick something up
            if self.task.metadata != None and not self.holding(controller):
                # go to location
                dest = self.task.metadata
                arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])
                if not arrived:
                    return
                meta_tile = controller.get_tile(controller.get_team(), dest[0], dest[1])
                is_pan = hasattr(meta_tile, "item") and isinstance(meta_tile.item, Pan)
                print("meta_tile is", meta_tile, "and meta_tile.item is", meta_tile.item)
                if is_pan and controller.take_from_pan(self.id, dest[0], dest[1]):
                    print("took from pan!")
                    return 
                if controller.pickup(self.id, dest[0], dest[1]):
                    return 
                else:
                    raise Exception(f"Plate metadata {self.task.metadata} should be a valid location")

            # move to plate
            dest = self.task.ingredient.order.plate.loc
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived:
                # place on plate 
                controller.add_food_to_plate(self.id, dest[0], dest[1])
                self.task.ingredient.status = IngredientStatus.PLATED
                self.task.ingredient.working = None
                self.task = None
        elif self.task.task == Tasks.SUBMIT_PLATE:
            self.task.ingredient.order.submitting = True

            if not self.holding(controller):
                # go to plate
                dest = self.task.ingredient.order.plate.loc 
                arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])
                if not arrived:
                    return 
                
                # pick up plate
                controller.pickup(self.id, dest[0], dest[1])
                return

            # go to submit
            dest = self.task.get_closest_loc(bot_loc)
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])
            if not arrived:
                return

            # submit
            controller.submit(self.id, dest[0], dest[1])
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
        self.assembly_counter = None 
        self.cooker_loc = None
        self.my_bot_id = None
        
        self.state = 0

        # new things
        self.orders = {}
        self.bots = {}

        self.parsed_map = False

        self.plates = []
        self.used_counters = {}

    # Inspects orders list and assigns a raw and cooked staging area
    def assign_staging(self):
        pass

    def get_bfs_path(self, controller: RobotController, start: Tuple[int, int], target_predicate, bot_id: int = None) -> Optional[Tuple[int, int]]:
        queue = deque([(start, [])]) 
        visited = set([start])
        w, h = self.map.width, self.map.height
        
        # Get positions of all other bots to avoid collisions
        occupied_positions = set()
        if bot_id is not None:
            my_bots = controller.get_team_bot_ids(team=controller.get_team())
            for other_bot_id in my_bots:
                if other_bot_id != bot_id:
                    other_bot_state = controller.get_bot_state(other_bot_id)
                    if other_bot_state:
                        occupied_positions.add((other_bot_state['x'], other_bot_state['y']))

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
                        # Check if tile is walkable AND not occupied by another bot
                        if controller.get_map(team=controller.get_team()).is_tile_walkable(nx, ny) and (nx, ny) not in occupied_positions:
                            visited.add((nx, ny))
                            queue.append(((nx, ny), path + [(dx, dy)]))
        return None

    def move_towards(self, controller: RobotController, bot_id: int, target_x: int, target_y: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        bx, by = bot_state['x'], bot_state['y']
        def is_adjacent_to_target(x, y, tile):
            return max(abs(x - target_x), abs(y - target_y)) <= 1
        if is_adjacent_to_target(bx, by, None): return True
        
        # Pass bot_id to get_bfs_path so it can avoid other bots
        step = self.get_bfs_path(controller, (bx, by), is_adjacent_to_target, bot_id)
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

                                    temp_dist[nx][ny] = current_dist + 1
                                    move_matrix[nx][ny][(tx, ty)] = (curr_x - nx, curr_y - ny)
                                    if tile_name == "FLOOR":
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
    

    # orders are in the format of: [{'order_id': 1, 'required': ['NOODLES', 'MEAT'], 'created_turn': 0, 'expires_turn': 200, 'reward': 10000, 'penalty': 3, 'claimed_by': None, 'completed_turn': None, 'is_active': True}]
    # takes all the ingredients by all orders and prioritize them and return the list of the ingredients
    def prioritize_ingredients(self, controller):
        orders = controller.get_orders(controller.get_team())

        ingredients = []

        for order in orders:
            priority = 1
            if not order["is_active"]:
                continue
        
            oid = order["order_id"]
            if oid not in self.orders:
                self.orders[oid] = Order(order, self)
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
    
    # Bot will procure a plate
    # Decides whether to buy a plate or get a clean one
    # Returns true if there is a free plate assigned to the order, else FF we need to do work
    def get_plate_task(self, ingredient):
        ingredient.preplate_status = ingredient.status
        if ingredient.order.plate != None:
            return Task(Tasks.PLATE, ingredient, None)

        for plate in self.plates:
            if plate.is_free() and not plate.is_dirty():
                # TODO: we can optimize which plate we assign theoretically
                ingredient.order.plate = plate 
                plate.order = ingredient.order
                return Task(Tasks.PLATE, ingredient, plate)  

        for plate in self.plates:
            if plate.is_dirty():
                return Task(Tasks.WASH_PLATE, ingredient, self.nav_maps["Sink"])

        return Task(Tasks.ACQUIRE_PLATE, ingredient, self.nav_maps["Shop"])

    # generate the task list by the priority given from prioritize_ingredients
    def generate_tasks(self, controller, ingredient_list):
        task_list = []

        # TODO: generate tasks for buying plates/pans and 
        # moving them if not generated yet

        # try to generate the next task for submitting orders
        for order_id in self.orders:
            order = self.orders[order_id]

            if order.all_plated() and not order.submitting:
                task_list.append((10000, Task(Tasks.SUBMIT_PLATE, order.ings[0], self.nav_maps["Submit"])))

        making_plates = {}

        # try to generate the next task for all the ingredients
        for (priority, ingredient) in ingredient_list:
            if ingredient.working != None:
                continue    
            if ingredient.status == IngredientStatus.NOT_STARTED:
                # buy only if we have enough money
                # TODO: do some handling for when bots are on their way to the shop!
                # if we can't give the order a plate, first try to get a plate
                plate_task = self.get_plate_task(ingredient)
                if plate_task.task != Tasks.PLATE and ingredient.order.id not in making_plates:
                    task_list.append((priority, plate_task))
                    making_plates[order.id] = True
                elif controller.get_team_money(team=controller.get_team()) >= ingredient.cost:
                    task_list.append((priority, Task(Tasks.BUY_INGREDIENT, ingredient, self.nav_maps["Shop"])))
            elif ingredient.status == IngredientStatus.BOUGHT:
                # case on the ingredient
                if ingredient.choppable:
                    task_list.append((priority, Task(Tasks.CHOP, ingredient, self.nav_maps["Counter"])))
                elif ingredient.cookable: 
                    task_list.append((priority, Task(Tasks.COOK, ingredient, self.nav_maps["Cooker"])))
                else:
                    # put it on a plate
                    task_list.append((priority, self.get_plate_task(ingredient)))
            elif ingredient.status == IngredientStatus.CHOPPED:
                if ingredient.cookable:
                    task_list.append((priority, Task(Tasks.COOK, ingredient, self.nav_maps["Cooker"])))
                else:
                    # put it on a plate
                    task_list.append((priority, self.get_plate_task(ingredient)))
            elif ingredient.status == IngredientStatus.COOKING:
                # check if by the time you walk there it will be cooked
                cooker_tile = controller.get_tile(controller.get_team(), ingredient.loc[0], ingredient.loc[1])
                if cooker_tile.item.food.cooked_stage == 1:
                    task_list.append((priority, Task(Tasks.PLATE, ingredient, ingredient.loc)))
            elif ingredient.status == IngredientStatus.BOUGHT_PLATE:
                task_list.append((priority, Task(Tasks.MOVE_PLATE_TO_COUNTER, ingredient, self.nav_maps["Counter"])))
            elif ingredient.status == IngredientStatus.COOKED:
                # put it on a plate
                task_list.append((priority, self.get_plate_task(ingredient)))
            elif ingredient.status == IngredientStatus.PLATED:
                # this ingredient is done!!
                continue
            else:
                print("ingredients.status is ", ingredient.status)
                raise(NotImplemented)
            

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
        print("            self.bots is", self.bots)
        for bot_id in self.bots:
            bot = self.bots[bot_id]
            if bot.task is None:
                self.assign_bot(bot, task_list)
                if bot.task is not None:
                    bot.task.ingredient.working = bot

            # all bots do what they're assigned to do
            bot.work(controller)