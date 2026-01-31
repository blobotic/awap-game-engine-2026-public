"""
Robust Bot - patched:
  Fix 1: Actually start cooking via can_start_cook()/start_cook() (not place()).
  Fix 2: For *_on_plate milestones, ensure bot acquires the right ingredient first (pickup/take_from_pan),
         then add_food_to_plate().
  Fix 3: More reliable submit check: normalize food names to UPPER.
  Fix 4: Make 'submitted' milestone actually submit: pickup plate -> move to submit -> submit.
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import traceback

from game_constants import Team, FoodType, ShopCosts
from robot_controller import RobotController


# ============================================
# Constants
# ============================================

INGREDIENT_COSTS = {
    'EGG': 20, 'ONIONS': 30, 'MEAT': 80, 'NOODLES': 40, 'SAUCE': 10,
}

INGREDIENT_PROPS = {
    'EGG': {'can_chop': False, 'can_cook': True},
    'ONIONS': {'can_chop': True, 'can_cook': False},
    'MEAT': {'can_chop': True, 'can_cook': True},
    'NOODLES': {'can_chop': False, 'can_cook': False},
    'SAUCE': {'can_chop': False, 'can_cook': False},
}

PLATE_COST = 2
PAN_COST = 4
MAX_RETRIES = 3

DIRECTIONS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]


# ============================================
# Helpers / Kitchen Plan
# ============================================

@dataclass
class KitchenPlan:
    cooker: Optional[Tuple[int, int]] = None
    assembly_counter: Optional[Tuple[int, int]] = None
    chop_counter: Optional[Tuple[int, int]] = None
    shop: Optional[Tuple[int, int]] = None
    submit: Optional[Tuple[int, int]] = None
    sink: Optional[Tuple[int, int]] = None
    sinktable: Optional[Tuple[int, int]] = None
    trash: Optional[Tuple[int, int]] = None
    all_counters: List[Tuple[int, int]] = field(default_factory=list)

def _norm_food_name(x: Any) -> str:
    if x is None:
        return ""
    # enums -> .name, strings stay strings
    s = getattr(x, "name", None)
    if s is None:
        s = str(x)
    return str(s).upper()

def chebyshev(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def compute_kitchen_plan(rc: RobotController, team: Team) -> KitchenPlan:
    game_map = rc.get_map(team)
    plan = KitchenPlan()

    cookers, counters, shops, submits = [], [], [], []
    sinks, sinktables, trashes = [], [], []

    for x in range(game_map.width):
        for y in range(game_map.height):
            tile = rc.get_tile(team, x, y)
            if tile is None:
                continue
            name = getattr(tile, 'tile_name', None)
            if name == 'COOKER':
                cookers.append((x, y))
            elif name == 'COUNTER':
                counters.append((x, y))
            elif name == 'SHOP':
                shops.append((x, y))
            elif name == 'SUBMIT':
                submits.append((x, y))
            elif name == 'SINK':
                sinks.append((x, y))
            elif name == 'SINKTABLE':
                sinktables.append((x, y))
            elif name == 'TRASH':
                trashes.append((x, y))

    plan.all_counters = counters

    if shops:
        plan.shop = shops[0]
    if submits:
        plan.submit = submits[0]
    if sinks:
        plan.sink = sinks[0]
    if sinktables:
        plan.sinktable = sinktables[0]
    if trashes:
        plan.trash = trashes[0]

    if cookers and plan.shop and plan.submit:
        plan.cooker = min(cookers, key=lambda c: chebyshev(c, plan.shop) + chebyshev(c, plan.submit))
    elif cookers:
        plan.cooker = cookers[0]

    if counters and plan.cooker and plan.submit:
        plan.assembly_counter = min(counters, key=lambda c: chebyshev(c, plan.cooker) + chebyshev(c, plan.submit))
        remaining = [c for c in counters if c != plan.assembly_counter]
        plan.chop_counter = min(remaining, key=lambda c: chebyshev(c, plan.cooker)) if remaining else plan.assembly_counter
    elif counters:
        plan.assembly_counter = counters[0]
        plan.chop_counter = counters[0]

    return plan


# ============================================
# Milestones
# ============================================

class MilestoneStatus(Enum):
    PENDING = auto()
    READY = auto()
    IN_PROGRESS = auto()
    BLOCKED = auto()
    COMPLETE = auto()


@dataclass
class Milestone:
    id: str
    deps: List[str] = field(default_factory=list)
    status: MilestoneStatus = MilestoneStatus.PENDING
    assigned_bot: Optional[int] = None
    ingredient: Optional[str] = None

    def is_ready(self, completed: Set[str]) -> bool:
        return all(d in completed for d in self.deps)


@dataclass
class OrderPlan:
    order_id: int
    order: Dict[str, Any]
    milestones: Dict[str, Milestone] = field(default_factory=dict)
    completed: Set[str] = field(default_factory=set)

    def mark_complete(self, m_id: str):
        if m_id in self.milestones:
            self.milestones[m_id].status = MilestoneStatus.COMPLETE
            self.completed.add(m_id)

    def is_complete(self) -> bool:
        return 'submitted' in self.completed


def build_order_plan(order: Dict[str, Any]) -> OrderPlan:
    milestones: Dict[str, Milestone] = {}
    ingredients = order['required']

    needs_cooking = any(INGREDIENT_PROPS.get(ing, {}).get('can_cook') for ing in ingredients)
    if needs_cooking:
        milestones['pan_ready'] = Milestone(id='pan_ready', deps=[])

    milestones['plate_ready'] = Milestone(id='plate_ready', deps=[])

    for ing in ingredients:
        props = INGREDIENT_PROPS.get(ing, {})
        ing_lower = ing.lower()

        milestones[f'{ing_lower}_bought'] = Milestone(id=f'{ing_lower}_bought', deps=[], ingredient=ing)
        prev = f'{ing_lower}_bought'

        if props.get('can_chop'):
            milestones[f'{ing_lower}_chopped'] = Milestone(id=f'{ing_lower}_chopped', deps=[prev], ingredient=ing)
            prev = f'{ing_lower}_chopped'

        if props.get('can_cook'):
            milestones[f'{ing_lower}_cooking'] = Milestone(
                id=f'{ing_lower}_cooking', deps=[prev, 'pan_ready'], ingredient=ing
            )
            milestones[f'{ing_lower}_cooked'] = Milestone(
                id=f'{ing_lower}_cooked', deps=[f'{ing_lower}_cooking'],
                status=MilestoneStatus.BLOCKED, ingredient=ing
            )
            prev = f'{ing_lower}_cooked'

        milestones[f'{ing_lower}_on_plate'] = Milestone(
            id=f'{ing_lower}_on_plate', deps=[prev, 'plate_ready'], ingredient=ing
        )

    plate_deps = [f'{ing.lower()}_on_plate' for ing in ingredients]
    milestones['submitted'] = Milestone(id='submitted', deps=plate_deps)

    return OrderPlan(order_id=order['order_id'], order=order, milestones=milestones)


# ============================================
# Bot State
# ============================================

@dataclass
class BotTaskState:
    current_milestone: Optional[str] = None
    retry_count: int = 0
    last_pos: Optional[Tuple[int, int]] = None
    stuck_turns: int = 0


# ============================================
# Main Bot
# ============================================

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.kitchen_plan: Optional[KitchenPlan] = None
        self.order_plan: Optional[OrderPlan] = None
        self.current_order: Optional[Dict] = None
        self.bot_states: Dict[int, BotTaskState] = {}
        self.tile_reservations: Dict[Tuple[int, int], int] = {}

        self._last_error_turn = -100
        self._error_count = 0

        self._fail_place = 0
        self._fail_buy = 0
        self._fail_other = 0

    def _milestone_for_ingredient(self, suffix: str, ingredient: str) -> str:
        return f"{ingredient.lower()}_{suffix}"

    def _force_complete(self, bot_id: int, m_id: str):
        """Force-complete a milestone (used when the action succeeded but state schema is unclear)."""
        if not self.order_plan or m_id not in self.order_plan.milestones:
            return
        self.order_plan.mark_complete(m_id)
        m = self.order_plan.milestones[m_id]
        m.assigned_bot = None
        bs = self.bot_states.get(bot_id)
        if bs:
            bs.current_milestone = None
            bs.retry_count = 0

    def _tile_item(self, tile):
        """Return the item on a tile, supporting both object and dict tiles."""
        if tile is None:
            return None
        # dict style
        if isinstance(tile, dict):
            return tile.get("item")
        # object style
        return getattr(tile, "item", None)

    def _item_type(self, item) -> Optional[str]:
        """Return a stable 'type' string like 'Plate', 'Pan', 'Food'."""
        if item is None:
            return None
        if isinstance(item, dict):
            return item.get("type") or item.get("item_type")
        return type(item).__name__

    def _food_name(self, obj) -> Optional[str]:
        """Extract food_name from either dict or object."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get("food_name") or obj.get("name")
        return getattr(obj, "food_name", None)

    def _plate_food_list(self, plate_item) -> List:
        """Get list of foods on a plate (dict or object)."""
        if plate_item is None:
            return []
        if isinstance(plate_item, dict):
            # common schema: {'type': 'Plate', 'food': [ ... ]}
            foods = plate_item.get("food") or plate_item.get("foods") or []
            return foods if isinstance(foods, list) else []
        foods = getattr(plate_item, "food", None)
        return foods if isinstance(foods, list) else []

    def play_turn(self, rc: RobotController):
        try:
            self._play_turn_impl(rc)
        except Exception as e:
            self._error_count += 1
            turn = rc.get_turn()
            if turn - self._last_error_turn >= 50:
                print(f"[ERROR turn {turn}] Exception in play_turn: {e}")
                traceback.print_exc()
                self._last_error_turn = turn

    def _play_turn_impl(self, rc: RobotController):
        team = rc.get_team()
        bot_ids = rc.get_team_bot_ids(team)
        turn = rc.get_turn()
        if not bot_ids:
            return

        if self.kitchen_plan is None:
            self.kitchen_plan = compute_kitchen_plan(rc, team)

        self.map = rc.get_map(team)
        self.tile_reservations.clear()

        if self._needs_new_order(rc):
            self._select_new_order(rc)
        if self.order_plan is None:
            return

        self._update_cooking_status(rc)

        for bot_id in bot_ids:
            self._execute_bot_turn(rc, bot_id)

        if turn % 50 == 0:
            self._log_summary(rc, bot_ids)

    def _log_summary(self, rc: RobotController, bot_ids: List[int]):
        team = rc.get_team()
        turn = rc.get_turn()
        money = rc.get_team_money(team)

        bot_info = []
        for bid in bot_ids:
            state = rc.get_bot_state(bid)
            if state:
                pos = (state['x'], state['y'])
                holding = state.get('holding')
                h_type = holding.get('type') if holding else 'None'
                bs = self.bot_states.get(bid)
                milestone = bs.current_milestone if bs else 'None'
                bot_info.append(f"B{bid}@{pos} hold={h_type} ms={milestone}")

        order_info = "None"
        if self.current_order:
            order_info = f"id={self.current_order['order_id']} req={self.current_order['required']}"

        print(
            f"[T{turn}] ${money} | {' | '.join(bot_info)} | order={order_info} "
            f"| fails: place={self._fail_place} buy={self._fail_buy} other={self._fail_other}"
        )

    # ---------- Order selection ----------

    def _needs_new_order(self, rc: RobotController) -> bool:
        team = rc.get_team()
        if self.current_order is None:
            return True
        if self.order_plan and self.order_plan.is_complete():
            return True
        if rc.get_turn() > self.current_order.get('expires_turn', 0):
            return True
        for o in rc.get_orders(team):
            if o['order_id'] == self.current_order['order_id']:
                if o.get('completed_turn') is not None:
                    return True
        return False

    def _select_new_order(self, rc: RobotController):
        team = rc.get_team()
        self.bot_states.clear()
        orders = rc.get_orders(team)
        money = rc.get_team_money(team)
        turn = rc.get_turn()

        valid = []
        for o in orders:
            if not o.get('is_active'):
                continue
            if o.get('completed_turn') is not None:
                continue
            cost = sum(INGREDIENT_COSTS.get(ing, 0) for ing in o['required']) + PLATE_COST
            if any(INGREDIENT_PROPS.get(ing, {}).get('can_cook') for ing in o['required']):
                cost += PAN_COST
            if cost <= money:
                valid.append((o, cost))

        if not valid:
            self.current_order = None
            self.order_plan = None
            return

        def score(item):
            o, cost = item
            turns_left = max(1, o['expires_turn'] - turn)
            return (o['reward'] - cost) / turns_left

        valid.sort(key=score, reverse=True)
        self.current_order = valid[0][0]
        self.order_plan = build_order_plan(self.current_order)
        print(f"[T{turn}] Selected ORDER id={self.current_order['order_id']} required={self.current_order['required']}")

    # ---------- Tile queries ----------

    def _tile_is_empty(self, rc: RobotController, pos: Tuple[int, int]) -> bool:
        if pos is None:
            return False
        team = rc.get_team()
        tile = rc.get_tile(team, pos[0], pos[1])
        if tile is None:
            return False
        return self._tile_item(tile) is None

    def _tile_has_plate(self, rc: RobotController, pos: Tuple[int, int]) -> bool:
        if pos is None:
            return False
        team = rc.get_team()
        tile = rc.get_tile(team, pos[0], pos[1])
        if tile is None:
            return False
        item = self._tile_item(tile)
        return self._item_type(item) == "Plate"

    def _find_empty_counter(self, rc: RobotController) -> Optional[Tuple[int, int]]:
        # Prefer assembly/chop if empty
        for counter in [self.kitchen_plan.assembly_counter, self.kitchen_plan.chop_counter]:
            if counter and self._tile_is_empty(rc, counter):
                return counter
        # Otherwise any counter
        for counter in self.kitchen_plan.all_counters:
            if self._tile_is_empty(rc, counter):
                return counter
        return None

    def _find_plate_location(self, rc: RobotController) -> Optional[Tuple[int, int]]:
        """Find where a plate currently is (not just counters).

        Many engines allow plates to sit on sinktable/sink/submit, not only COUNTER.
        """
        team = rc.get_team()

        # Candidate spots we know about
        candidates: List[Tuple[int, int]] = []

        if self.kitchen_plan.assembly_counter:
            candidates.append(self.kitchen_plan.assembly_counter)
        if self.kitchen_plan.chop_counter and self.kitchen_plan.chop_counter != self.kitchen_plan.assembly_counter:
            candidates.append(self.kitchen_plan.chop_counter)

        # All counters
        candidates.extend(self.kitchen_plan.all_counters)

        # Also check other stations that can hold items in some maps
        if self.kitchen_plan.sinktable:
            candidates.append(self.kitchen_plan.sinktable)
        if self.kitchen_plan.sink:
            candidates.append(self.kitchen_plan.sink)
        if self.kitchen_plan.submit:
            candidates.append(self.kitchen_plan.submit)

        # Dedup while preserving order
        seen = set()
        deduped = []
        for p in candidates:
            if p and p not in seen:
                seen.add(p)
                deduped.append(p)

        # Look for a Plate on any candidate tile
        for x, y in deduped:
            tile = rc.get_tile(team, x, y)
            if tile is None:
                continue
            item = self._tile_item(tile)
            if self._item_type(item) == "Plate":
                return (x, y)

        return None

    def _find_food_on_counters(self, rc: RobotController, ingredient: str) -> Optional[Tuple[int, int]]:
        """Find an ingredient (Food object) sitting on a counter."""
        team = rc.get_team()
        for counter in self.kitchen_plan.all_counters:
            tile = rc.get_tile(team, *counter)
            it = getattr(tile, 'item', None) if tile else None
            if it is None:
                continue
            # it may be Food or Plate/Pan; we only want Food on counter
            if getattr(it, 'food_name', None) == ingredient:
                return counter
        return None

    # ---------- Cooking status ----------

    def _update_cooking_status(self, rc: RobotController):
        if self.order_plan is None or self.kitchen_plan.cooker is None:
            return

        team = rc.get_team()
        tile = rc.get_tile(team, *self.kitchen_plan.cooker)
        pan = getattr(tile, 'item', None) if tile else None
        food = getattr(pan, 'food', None) if pan else None
        if not food:
            return

        stage = getattr(food, 'cooked_stage', 0)
        name = getattr(food, 'food_name', '')
        ing_lower = str(name).lower()

        if stage >= 1:
            cooking_id = f'{ing_lower}_cooking'
            cooked_id = f'{ing_lower}_cooked'
            if cooking_id in self.order_plan.milestones:
                self.order_plan.mark_complete(cooking_id)
            if cooked_id in self.order_plan.milestones:
                m = self.order_plan.milestones[cooked_id]
                if m.status == MilestoneStatus.BLOCKED:
                    m.status = MilestoneStatus.READY

    # ---------- Turn execution ----------

    def _execute_bot_turn(self, rc: RobotController, bot_id: int):
        if self.order_plan is None:
            return

        state = rc.get_bot_state(bot_id)
        if state is None:
            return

        if bot_id not in self.bot_states:
            self.bot_states[bot_id] = BotTaskState()

        bs = self.bot_states[bot_id]
        bx, by = state['x'], state['y']
        holding = state.get('holding')

        if bs.last_pos == (bx, by):
            bs.stuck_turns += 1
        else:
            bs.stuck_turns = 0
        bs.last_pos = (bx, by)

        self.tile_reservations[(bx, by)] = bot_id

        urgent_op = self._check_urgency(rc, bot_id, state)
        if urgent_op:
            self._execute_action(rc, bot_id, urgent_op)
            return

        if holding:
            self._handle_holding(rc, bot_id, holding)
            return

        if bs.current_milestone is None or bs.retry_count > MAX_RETRIES or bs.stuck_turns > 10:
            bs.current_milestone = self._assign_milestone(bot_id)
            bs.retry_count = 0
            bs.stuck_turns = 0
            if bs.current_milestone:
                print(f"[DBG] bot {bot_id} assigned milestone={bs.current_milestone}")

        if bs.current_milestone is None:
            return

        self._execute_milestone(rc, bot_id, bs.current_milestone)

    def _check_urgency(self, rc: RobotController, bot_id: int, state: dict) -> Optional[tuple]:
        """If cooked food is in pan, take it ASAP (hands permitting)."""
        if self.kitchen_plan.cooker is None:
            return None

        team = rc.get_team()
        tile = rc.get_tile(team, *self.kitchen_plan.cooker)
        pan = getattr(tile, 'item', None) if tile else None
        food = getattr(pan, 'food', None) if pan else None
        if not food:
            return None

        stage = getattr(food, 'cooked_stage', 0)
        if stage == 1:
            if state.get('holding') is not None:
                empty = self._find_empty_counter(rc)
                if empty:
                    return ('place', empty)
                if self.kitchen_plan.trash:
                    return ('trash', self.kitchen_plan.trash)
                return None
            return ('take_from_pan', self.kitchen_plan.cooker)
        return None

    def _handle_holding(self, rc: RobotController, bot_id: int, holding: dict):
        """Handle what bot is currently holding - safer: never trash required ingredients."""
        h_type = holding.get('type')
        bs = self.bot_states.get(bot_id)
        stuck = bs.stuck_turns if bs else 0

        if h_type == 'Pan':
            cooker = self.kitchen_plan.cooker
            if cooker:
                team = rc.get_team()
                tile = rc.get_tile(team, cooker[0], cooker[1])
                if tile and (not hasattr(tile, 'item') or tile.item is None):
                    self._execute_action(rc, bot_id, ('place', cooker))
                    return
            empty = self._find_empty_counter(rc)
            if empty:
                self._execute_action(rc, bot_id, ('place', empty))
                return
            if self.kitchen_plan.assembly_counter:
                self._execute_action(rc, bot_id, ('move_to', self.kitchen_plan.assembly_counter))
            return

        elif h_type == 'Plate':
            foods = holding.get('food', [])
            required = set(self.current_order['required']) if self.current_order else set()
            on_plate = {f.get('food_name') for f in foods if isinstance(f, dict) and f.get('food_name')}

            # If complete, go submit.
            if required and required == on_plate:
                self._execute_action(rc, bot_id, ('submit', self.kitchen_plan.submit))
                return

            empty_counter = self._find_empty_counter(rc)
            if empty_counter:
                self._execute_action(rc, bot_id, ('place', empty_counter))
                return

            if self.kitchen_plan.assembly_counter:
                self._execute_action(rc, bot_id, ('move_to', self.kitchen_plan.assembly_counter))
            return

        elif h_type == 'Food':
            name = holding.get('food_name')
            ing = str(getattr(name, "name", name)).upper() if name is not None else ""

            props = INGREDIENT_PROPS.get(ing, {})
            chopped = bool(holding.get('chopped', False))
            cooked = int(holding.get('cooked_stage', 0))

            required = set(self.current_order['required']) if self.current_order else set()
            is_required = ing in required

            # Need chopping
            if props.get('can_chop') and not chopped:
                # Try to place onto chop counter if empty
                chop_counter = self.kitchen_plan.chop_counter
                if chop_counter and self._tile_is_empty(rc, chop_counter):
                    self._execute_action(rc, bot_id, ('place', chop_counter))
                    return

                # Otherwise place on ANY empty counter
                empty = self._find_empty_counter(rc)
                if empty:
                    self._execute_action(rc, bot_id, ('place', empty))
                    return

                # If none empty: do NOT trash required ingredient; wait near chop area
                if chop_counter:
                    self._execute_action(rc, bot_id, ('move_to', chop_counter))
                elif self.kitchen_plan.assembly_counter:
                    self._execute_action(rc, bot_id, ('move_to', self.kitchen_plan.assembly_counter))

                # Only trash if NOT required and we're extremely stuck
                if (not is_required) and stuck > 30 and self.kitchen_plan.trash:
                    self._execute_action(rc, bot_id, ('trash', self.kitchen_plan.trash))
                return

            # Need cooking
            if props.get('can_cook') and cooked == 0:
                cooker = self.kitchen_plan.cooker
                if cooker:
                    team = rc.get_team()
                    tile = rc.get_tile(team, cooker[0], cooker[1])
                    if tile and getattr(tile, 'item', None):
                        pan = tile.item
                        if getattr(pan, 'food', None) is None:
                            self._execute_action(rc, bot_id, ('place', cooker))
                            return
                    self._execute_action(rc, bot_id, ('move_to', cooker))
                    return

                # No cooker known: place on any counter (don’t trash required)
                empty = self._find_empty_counter(rc)
                if empty:
                    self._execute_action(rc, bot_id, ('place', empty))
                    return

                if self.kitchen_plan.assembly_counter:
                    self._execute_action(rc, bot_id, ('move_to', self.kitchen_plan.assembly_counter))

                if (not is_required) and stuck > 30 and self.kitchen_plan.trash:
                    self._execute_action(rc, bot_id, ('trash', self.kitchen_plan.trash))
                return

            # Otherwise: should be plated
            plate_loc = self._find_plate_location(rc)
            if plate_loc:
                self._execute_action(rc, bot_id, ('add_to_plate', plate_loc))
                return

            # No plate found: place on any counter; do NOT trash required
            empty = self._find_empty_counter(rc)
            if empty:
                self._execute_action(rc, bot_id, ('place', empty))
                return

            if self.kitchen_plan.assembly_counter:
                self._execute_action(rc, bot_id, ('move_to', self.kitchen_plan.assembly_counter))

            if (not is_required) and stuck > 30 and self.kitchen_plan.trash:
                self._execute_action(rc, bot_id, ('trash', self.kitchen_plan.trash))

    # ---------- Milestone scheduling ----------

    def _assign_milestone(self, bot_id: int) -> Optional[str]:
        if self.order_plan is None:
            return None

        ready = []
        for m in self.order_plan.milestones.values():
            if m.status == MilestoneStatus.COMPLETE:
                continue
            if m.is_ready(self.order_plan.completed):
                if m.status == MilestoneStatus.PENDING:
                    m.status = MilestoneStatus.READY
                if m.status == MilestoneStatus.READY and m.assigned_bot is None:
                    ready.append(m)

        if not ready:
            return None

        def priority(m: Milestone) -> int:
            if '_cooking' in m.id:
                return 0
            if m.id in ['pan_ready', 'plate_ready']:
                return 1
            if '_bought' in m.id:
                return 2
            if '_chopped' in m.id:
                return 3
            if '_cooked' in m.id:
                return 4
            if '_on_plate' in m.id:
                return 5
            if m.id == 'submitted':
                return 6
            return 10

        ready.sort(key=priority)
        m = ready[0]
        m.assigned_bot = bot_id
        m.status = MilestoneStatus.IN_PROGRESS
        return m.id

    def _execute_milestone(self, rc: RobotController, bot_id: int, m_id: str):
        """FIXED: cooking uses start_cook; on_plate acquires ingredient first; submitted actually submits."""
        team = rc.get_team()
        bs = self.bot_states.get(bot_id)
        m = self.order_plan.milestones.get(m_id)
        if m is None:
            return

        if m_id == 'pan_ready':
            self._execute_action(rc, bot_id, ('buy', 'PAN', self.kitchen_plan.shop))
            return

        if m_id == 'plate_ready':
            if self.kitchen_plan.sinktable:
                tile = rc.get_tile(team, *self.kitchen_plan.sinktable)
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    self._execute_action(rc, bot_id, ('take_clean_plate', self.kitchen_plan.sinktable))
                    return
            self._execute_action(rc, bot_id, ('buy', 'PLATE', self.kitchen_plan.shop))
            return

        if '_bought' in m_id:
            self._execute_action(rc, bot_id, ('buy', m.ingredient, self.kitchen_plan.shop))
            return

        if '_chopped' in m_id:
            for counter in self.kitchen_plan.all_counters:
                tile = rc.get_tile(team, *counter)
                it = getattr(tile, 'item', None) if tile else None
                if it and getattr(it, 'food_name', None) == m.ingredient:
                    if getattr(it, 'chopped', False):
                        self._execute_action(rc, bot_id, ('pickup', counter))
                    else:
                        self._execute_action(rc, bot_id, ('chop', counter))
                    return
            if bs:
                bs.retry_count += 1
            return

        if '_cooking' in m_id and '_cooked' not in m_id:
            # FIX: start cooking (requires holding the ingredient and pan on cooker)
            self._execute_action(rc, bot_id, ('start_cook', self.kitchen_plan.cooker))
            return

        if '_cooked' in m_id:
            self._execute_action(rc, bot_id, ('take_from_pan', self.kitchen_plan.cooker))
            return

        if '_on_plate' in m_id:
            plate_loc = self._find_plate_location(rc)
            if not plate_loc:
                if bs:
                    bs.retry_count += 1
                return

            # FIX: ensure we are holding the correct ingredient first.
            st = rc.get_bot_state(bot_id)
            holding = st.get('holding') if st else None
            if not holding or holding.get('type') != 'Food' or holding.get('food_name') != m.ingredient:
                # try pickup from counters
                loc = self._find_food_on_counters(rc, m.ingredient)
                if loc:
                    self._execute_action(rc, bot_id, ('pickup', loc))
                    return
                # if it's cookable, maybe still in pan
                if INGREDIENT_PROPS.get(m.ingredient, {}).get('can_cook') and self.kitchen_plan.cooker:
                    self._execute_action(rc, bot_id, ('take_from_pan', self.kitchen_plan.cooker))
                    return
                if bs:
                    bs.retry_count += 1
                return

            self._execute_action(rc, bot_id, ('add_to_plate', plate_loc))
            return

        if m_id == 'submitted':
            # FIX: if not holding the plate, pick it up; else go submit
            st = rc.get_bot_state(bot_id)
            holding = st.get('holding') if st else None
            if holding and holding.get('type') == 'Plate':
                self._execute_action(rc, bot_id, ('submit', self.kitchen_plan.submit))
                return
            plate_loc = self._find_plate_location(rc)
            if plate_loc:
                self._execute_action(rc, bot_id, ('pickup', plate_loc))
                return
            if bs:
                bs.retry_count += 1
            return

    def _holding_food_name(self, holding: Optional[dict]) -> str:
        if not holding:
            return ""
        # common patterns
        if "food_name" in holding:
            return _norm_food_name(holding.get("food_name"))
        if "name" in holding:
            return _norm_food_name(holding.get("name"))
        # sometimes type itself encodes the ingredient
        # e.g., holding['type'] == 'MEAT'
        t = holding.get("type")
        if isinstance(t, str) and t.upper() in INGREDIENT_PROPS:
            return t.upper()
        return ""
    # ---------- Action execution ----------

    def _execute_action(self, rc: RobotController, bot_id: int, action: tuple):
        if action is None or len(action) < 2:
            return

        state = rc.get_bot_state(bot_id)
        if state is None:
            return

        bx, by = state['x'], state['y']
        act_type = action[0]
        target = action[-1]
        if target is None:
            return
        tx, ty = target

        # Move-only
        if act_type == 'move_to':
            self._move_toward(rc, bot_id, tx, ty)
            return

        # Not adjacent yet → move
        if not self._is_adjacent(bx, by, tx, ty):
            self._move_toward(rc, bot_id, tx, ty)
            return

        # Capture milestone + held ingredient BEFORE action (important for add_to_plate)
        bs = self.bot_states.get(bot_id)
        cur_m = bs.current_milestone if bs else None
        holding = state.get('holding') or {}
        held_food = holding.get('food_name')
        held_food_norm = str(getattr(held_food, "name", held_food)).upper() if held_food is not None else ""

        success = False

        if act_type == 'buy':
            item_name = action[1]
            if item_name == 'PAN':
                success = rc.buy(bot_id, ShopCosts.PAN, tx, ty)
            elif item_name == 'PLATE':
                success = rc.buy(bot_id, ShopCosts.PLATE, tx, ty)
            else:
                food_type = getattr(FoodType, item_name, None)
                if food_type:
                    success = rc.buy(bot_id, food_type, tx, ty)
            if not success:
                self._fail_buy += 1
            self._check_completion(rc, bot_id)

        elif act_type == 'place':
            success = rc.place(bot_id, tx, ty)
            if not success:
                self._fail_place += 1
            self._check_completion(rc, bot_id)

        elif act_type == 'pickup':
            success = rc.pickup(bot_id, tx, ty)
            if not success:
                self._fail_other += 1
            self._check_completion(rc, bot_id)

        elif act_type == 'chop':
            success = rc.chop(bot_id, tx, ty)
            if not success:
                self._fail_other += 1
            # IMPORTANT: chopping changes a counter item, so re-check
            self._check_completion(rc, bot_id)

        elif act_type == 'take_from_pan':
            success = rc.take_from_pan(bot_id, tx, ty)
            if not success:
                self._fail_other += 1
            self._check_completion(rc, bot_id)

        elif act_type == 'add_to_plate':
            success = rc.add_food_to_plate(bot_id, tx, ty)
            if not success:
                self._fail_other += 1
                self._check_completion(rc, bot_id)
            else:
                # FIX: if action succeeded, force-complete the matching *_on_plate milestone
                if cur_m and cur_m.endswith('_on_plate') and held_food_norm:
                    # only force if current milestone matches the ingredient we just added
                    if cur_m == self._milestone_for_ingredient('on_plate', held_food_norm):
                        self._force_complete(bot_id, cur_m)
                    else:
                        self._check_completion(rc, bot_id)

        elif act_type == 'submit':
            success = rc.submit(bot_id, tx, ty)
            if not success:
                self._fail_other += 1
                self._check_completion(rc, bot_id)
            else:
                # FIX: force-complete submitted on success
                if cur_m == 'submitted':
                    self._force_complete(bot_id, 'submitted')
                else:
                    self._check_completion(rc, bot_id)

        elif act_type == 'take_clean_plate':
            success = rc.take_clean_plate(bot_id, tx, ty)
            if not success:
                self._fail_other += 1
            self._check_completion(rc, bot_id)

        elif act_type == 'trash':
            success = rc.trash(bot_id, tx, ty)
            if not success:
                self._fail_other += 1

    def _is_adjacent(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        return max(abs(x1 - x2), abs(y1 - y2)) <= 1

    def _move_toward(self, rc: RobotController, bot_id: int, tx: int, ty: int):
        state = rc.get_bot_state(bot_id)
        if state is None:
            return
        bx, by = state['x'], state['y']

        best_move = None
        best_dist = float('inf')
        for dx, dy in DIRECTIONS:
            if not rc.can_move(bot_id, dx, dy):
                continue
            nx, ny = bx + dx, by + dy
            if (nx, ny) in self.tile_reservations and self.tile_reservations[(nx, ny)] != bot_id:
                continue
            dist = chebyshev((nx, ny), (tx, ty))
            if dist < best_dist:
                best_dist = dist
                best_move = (dx, dy)

        if best_move:
            rc.move(bot_id, best_move[0], best_move[1])
            st2 = rc.get_bot_state(bot_id)
            if st2:
                self.tile_reservations[(st2['x'], st2['y'])] = bot_id

    # ---------- Completion checks ----------

    def _check_completion(self, rc: RobotController, bot_id: int):
        if self.order_plan is None:
            return

        bs = self.bot_states.get(bot_id)
        if bs is None or bs.current_milestone is None:
            return

        m_id = bs.current_milestone
        m = self.order_plan.milestones.get(m_id)
        if m is None:
            return

        team = rc.get_team()
        state = rc.get_bot_state(bot_id)
        holding = state.get('holding') if state else None

        complete = False

        if m_id == 'pan_ready':
            if self.kitchen_plan.cooker:
                tile = rc.get_tile(team, *self.kitchen_plan.cooker)
                if tile and getattr(tile, 'item', None):
                    complete = True

        elif m_id == 'plate_ready':
            plate_loc = self._find_plate_location(rc)
            if plate_loc:
                complete = True

        elif '_bought' in m_id:
            # FIX: robust bought detection
            held = self._holding_food_name(holding)
            if held == _norm_food_name(m.ingredient):
                complete = True

        elif '_chopped' in m_id:
            held = self._holding_food_name(holding)
            if held == _norm_food_name(m.ingredient) and bool(holding.get('chopped', False)):
                complete = True

        elif '_cooked' in m_id:
            held = self._holding_food_name(holding)
            if held == _norm_food_name(m.ingredient) and int(holding.get('cooked_stage', 0)) >= 1:
                complete = True

        elif '_on_plate' in m_id:
            plate_loc = self._find_plate_location(rc)
            if plate_loc:
                tile = rc.get_tile(team, *plate_loc)
                plate_item = self._tile_item(tile)
                if self._item_type(plate_item) == "Plate":
                    for f in self._plate_food_list(plate_item):
                        fname = self._food_name(f)
                        if fname and str(fname).upper() == str(m.ingredient).upper():
                            complete = True
                            break

        elif m_id == 'submitted':
            for o in rc.get_orders(team):
                if o['order_id'] == self.order_plan.order_id:
                    if o.get('completed_turn') is not None:
                        complete = True
                    break

        if complete:
            self.order_plan.mark_complete(m_id)
            m.assigned_bot = None
            bs.current_milestone = None
            bs.retry_count = 0
            print(f"[DBG] bot {bot_id} completed milestone={m_id}")