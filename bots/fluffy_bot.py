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

ingredient_data = {"EGG": {"id": 0, "choppable": False, "cookable": True, "cost": 20},
                   "ONION": {"id": 1, "choppable": True, "cookable": False, "cost": 30},
                   "MEAT": {"id": 2, "choppable": True, "cookable": True, "cost": 80},
                   "NOODLES": {"id": 3, "choppable": False, "cookable": False, "cost": 40},
                   "SAUCE": {"id": 4, "choppable": False, "cookable": False, "cost": 10}}

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
        self.ings = []

        i = 0
        for ing in order["required"]:
            self.ings.append(Ingredient(ing, self, i))
            i += 1

        self.active = False

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
    GOTO_PLATE = 5
    ACQUIRE_PLATE = 6
    SUBMIT_PLATE = 7

class Task:
    def __init__(self, task, ingredient, metadata):
        self.task = task
        self.ingredient = ingredient
        self.metadata = metadata 

    def get_closest_loc(self, bot_loc):
        # TODO: implement
        return self.metadata[0]
    
    def get_closest_loc(self, bot_loc):
        # TODO: implement
        return self.metadata[0]


    def __str__(self):
        return f"{self.task} for {self.ingredient.name} with {self.metadata}"
    
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
                if controller.buy(self.id, self.task.metadata, dest[0], dest[1]):
                    self.task.ingredient.status = IngredientStatus.BOUGHT
        elif self.task.task == Tasks.CHOP:
            dest = self.task.get_closest_unclaimed_loc(bot_loc)
            # TODO: claim loc
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived:
                # start chopping 
                if controller.chop(self.id, dest[0], dest[1]):
                    self.task.ingredient.status = IngredientStatus.CHOPPED
                    # TODO: unclaim loc
        elif self.task.task == Tasks.COOK:
            dest = self.task.get_closest_unclaimed_loc(bot_loc)
            # TODO: claim loc
            arrived = self.botplayer.move_towards(controller, self.id, dest[0], dest[1])

            if arrived:
                # start chopping 
                if controller.start_cook(self.id, dest[0], dest[1]):
                    self.task.ingredient.status = IngredientStatus.COOKING
                    # TODO: unclaim loc
                
        else:
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
           - 'dists': 2D array of integers (distance to target)
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

            # Initialize grids
            dist_matrix = [[float('inf') for _ in range(height)] for _ in range(width)]
            move_matrix = [[None for _ in range(height)] for _ in range(width)]

            queue = deque()

            # Seed the BFS with all target locations (distance 0)
            for (tx, ty) in targets:
                dist_matrix[tx][ty] = 0
                move_matrix[tx][ty] = (0, 0)
                queue.append((tx, ty))

            while queue:
                curr_x, curr_y = queue.popleft()
                current_dist = dist_matrix[curr_x][curr_y]

                # Check all 8 neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue

                        nx, ny = curr_x + dx, curr_y + dy

                        if 0 <= nx < width and 0 <= ny < height:
                            if dist_matrix[nx][ny] == float('inf'):
                                neighbor_tile = m.tiles[nx][ny]
                                tile_name = getattr(neighbor_tile, "tile_name", "")

                                if tile_name == "FLOOR":
                                    dist_matrix[nx][ny] = current_dist + 1
                                    move_matrix[nx][ny] = (curr_x - nx, curr_y - ny)
                                    queue.append((nx, ny))

            self.nav_maps[category] = {
                "dists": dist_matrix,
                "moves": move_matrix,
                "targets": targets
            }

        self.parsed_map = True

        # --- OUTPUT RESULTS ---
        print("\n=== MAP PARSING COMPLETE ===")
        print(f"Categories found: {list(self.nav_maps.keys())}")

        # Iterate through ALL categories to print them
        for category, data in self.nav_maps.items():
            print(f"\nDistance Map for '{category}' (Visualized):")
            dms = data["dists"]

            # Print transposed (y is rows)
            for y in range(height-1, -1, -1):
                row_str = ""
                for x in range(width):
                    val = dms[x][y]
                    if val == float('inf'):
                        row_str += " . "
                    elif val == 0:
                        # Grab first letter for tag, e.g., C for Counter

                        tag = category[0].upper()
                        row_str += f"[{tag}]"
                    else:
                        row_str += f"{val:2} "
                print(row_str)
        print("============================\n")

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
                        task_list.append((priority, Task(Tasks.ACQUIRE_PLATE), ingredient, self.plates))
            elif ingredient.status == IngredientStatus.CHOPPED:
                if ingredient.cookable:
                    task_list.append((priority, Task(Tasks.COOK, ingredient, self.cookers)))
                else:
                    # put it on a plate
                    if ingredient.order.plate != None:
                        task_list.append((priority, Task(Tasks.GOTO_PLATE, ingredient, ingredient.order.plate)))
                    else:
                        # the order has no assigned plate yet
                        task_list.append((priority, Task(Tasks.ACQUIRE_PLATE), ingredient, self.plates))
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
                    task_list.append((priority, Task(Tasks.ACQUIRE_PLATE), ingredient, self.plates))
            

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
        print("self.bots is", self.bots)
        for bot_id in self.bots:
            bot = self.bots[bot_id]
            if bot.task is None:
                self.assign_bot(bot, task_list)

            # all bots do what they're assigned to do
            bot.work(controller)

    
        # self.my_bot_id = my_bots[0]
        # bot_id = self.my_bot_id
        
        # bot_info = controller.get_bot_state(bot_id)
        # bx, by = bot_info['x'], bot_info['y']

        # if self.assembly_counter is None:
        #     self.assembly_counter = self.find_nearest_tile(controller, bx, by, "COUNTER")
        # if self.cooker_loc is None:
        #     self.cooker_loc = self.find_nearest_tile(controller, bx, by, "COOKER")

        # if not self.assembly_counter or not self.cooker_loc: return

        # cx, cy = self.assembly_counter
        # kx, ky = self.cooker_loc

        # if self.state in [2, 8, 10] and bot_info.get('holding'):
        #     self.state = 16

        # #state 0: init + checking the pan
        # if self.state == 0:
        #     tile = controller.get_tile(controller.get_team(), kx, ky)
        #     if tile and isinstance(tile.item, Pan):
        #         self.state = 2
        #     else:
        #         self.state = 1

        # #state 1: buy pan
        # elif self.state == 1:
        #     holding = bot_info.get('holding')
        #     if holding: # assume it's the pan
        #         if self.move_towards(controller, bot_id, kx, ky):
        #             if controller.place(bot_id, kx, ky):
        #                 self.state = 2
        #     else:
        #         shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
        #         if not shop_pos: return
        #         sx, sy = shop_pos
        #         if self.move_towards(controller, bot_id, sx, sy):
        #             if controller.get_team_money(team=controller.get_team()) >= ShopCosts.PAN.buy_cost:
        #                 controller.buy(bot_id, ShopCosts.PAN, sx, sy)

        # #state 2: buy meat
        # elif self.state == 2:
        #     shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
        #     sx, sy = shop_pos
        #     if self.move_towards(controller, bot_id, sx, sy):
        #         if controller.get_team_money(team=controller.get_team()) >= FoodType.MEAT.buy_cost:
        #             if controller.buy(bot_id, FoodType.MEAT, sx, sy):
        #                 self.state = 3

        # #state 3: put meat on counter
        # elif self.state == 3:
        #     if self.move_towards(controller, bot_id, cx, cy):
        #         if controller.place(bot_id, cx, cy):
        #             self.state = 4

        # #state 4: chop meat
        # elif self.state == 4:
        #     if self.move_towards(controller, bot_id, cx, cy):
        #         if controller.chop(bot_id, cx, cy):
        #             self.state = 5

        # #state 5: pickup meat
        # elif self.state == 5:
        #     if self.move_towards(controller, bot_id, cx, cy):
        #         if controller.pickup(bot_id, cx, cy):
        #             self.state = 6

        # #state 6: put meat on counter
        # elif self.state == 6:
        #     if self.move_towards(controller, bot_id, kx, ky):
        #         # Using the NEW logic where place() starts cooking automatically
        #         if controller.place(bot_id, kx, ky):
        #             self.state = 8 # Skip state 7

        # #state 7: start the cook, but is cooking so we just go
        # elif self.state == 7:
        #     self.state = 8

        # #state 8: buy the plate
        # elif self.state == 8:
        #     shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
        #     sx, sy = shop_pos
        #     if self.move_towards(controller, bot_id, sx, sy):
        #         if controller.get_team_money(team=controller.get_team()) >= ShopCosts.PLATE.buy_cost:
        #             if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
        #                 self.state = 9

        # #state 9: put the plate on the counter
        # elif self.state == 9:
        #     if self.move_towards(controller, bot_id, cx, cy):
        #         if controller.place(bot_id, cx, cy):
        #             self.state = 10

        # #state 10: buy noodle
        # elif self.state == 10:
        #     shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
        #     sx, sy = shop_pos
        #     if self.move_towards(controller, bot_id, sx, sy):
        #         if controller.get_team_money(team=controller.get_team()) >= FoodType.NOODLES.buy_cost:
        #             if controller.buy(bot_id, FoodType.NOODLES, sx, sy):
        #                 self.state = 11

        # #state 11: add noodles to plate
        # elif self.state == 11:
        #     if self.move_towards(controller, bot_id, cx, cy):
        #         if controller.add_food_to_plate(bot_id, cx, cy):
        #             self.state = 12

        # #state 12: wait and take meat
        # elif self.state == 12:
        #     if self.move_towards(controller, bot_id, kx, ky):
        #         tile = controller.get_tile(controller.get_team(), kx, ky)
        #         if tile and isinstance(tile.item, Pan) and tile.item.food:
        #             food = tile.item.food
        #             if food.cooked_stage == 1:
        #                 if controller.take_from_pan(bot_id, kx, ky):
        #                     self.state = 13
        #             elif food.cooked_stage == 2:

        #                 #trash
        #                 if controller.take_from_pan(bot_id, kx, ky):
        #                     self.state = 16 
        #         else:
        #             if bot_info.get('holding'):
        #                 #trash
        #                 self.state = 16
        #             else:
        #                 #restart
        #                 self.state = 2 

        # #state 13: add meat to plate
        # elif self.state == 13:
        #     if self.move_towards(controller, bot_id, cx, cy):
        #         if controller.add_food_to_plate(bot_id, cx, cy):
        #             self.state = 14

        # #state 14: pick up the plate
        # elif self.state == 14:
        #     if self.move_towards(controller, bot_id, cx, cy):
        #         if controller.pickup(bot_id, cx, cy):
        #             self.state = 15

        # #state 15: submit
        # elif self.state == 15:
        #     submit_pos = self.find_nearest_tile(controller, bx, by, "SUBMIT")
        #     ux, uy = submit_pos
        #     if self.move_towards(controller, bot_id, ux, uy):
        #         if controller.submit(bot_id, ux, uy):
        #             self.state = 0

        # #state 16: trash
        # elif self.state == 16:
        #     trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
        #     if not trash_pos: return
        #     tx, ty = trash_pos
        #     if self.move_towards(controller, bot_id, tx, ty):
        #         if controller.trash(bot_id, tx, ty):
        #             self.state = 2 #restart
        # for i in range(1, len(my_bots)):
        #     self.my_bot_id = my_bots[i]
        #     bot_id = self.my_bot_id
            
        #     bot_info = controller.get_bot_state(bot_id)
        #     bx, by = bot_info['x'], bot_info['y']

        #     dx = random.choice([-1, 1])
        #     dy = random.choice([-1, 1])
        #     nx,ny = bx + dx, by + dy
        #     if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
        #         controller.move(bot_id, dx, dy)
        #         return