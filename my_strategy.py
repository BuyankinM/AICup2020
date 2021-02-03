from model import *
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict, namedtuple
from numba import njit, void, int8, int16, int64, float64, boolean
from heapq import heapify, heappop, heappush
import time


class Timer(object):
    def __init__(self, verbose=True, t=""):
        self.verbose = verbose
        self.t = t

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print(f"{self.t} = {self.msecs}")


def get_slice_to_control(x_pos, y_pos, size):
    min_x = max(0, x_pos - size)
    min_y = max(0, y_pos - size)
    max_x = min(80, x_pos + size + 1)
    max_y = min(80, y_pos + size + 1)
    return max_x, max_y, min_x, min_y


def find_best_manh_for_arrays(arr_1, arr_2):
    res_manh = cdist(arr_1, arr_2, "cityblock")
    ind_min_manh = np.unravel_index(np.argmin(res_manh), res_manh.shape)
    return ind_min_manh


def get_unit_area(radius, size):
    tr = [(x, y) for x in range(1, radius) for y in range(1, radius) if x + y <= radius]
    s1 = size - 1

    mas = [(x + s1, y + s1) for (x, y) in tr] \
          + [(x + s1, -y) for (x, y) in tr] \
          + [(-x, -y) for (x, y) in tr] \
          + [(-x, y + s1) for (x, y) in tr] \
          + [(x, y) if z else (y, x) for x in range(-radius, 0) for y in range(size) for z in range(2)] \
          + [(x, y) if z else (y, x) for x in range(size) for y in range(size, size + radius) for z in range(2)] \
          + [(x, y) for x in range(size) for y in range(size)]
    light_mas = np.array(mas, dtype=np.int8)

    return light_mas


@njit(int8[:, :](int16[:, :, :], int8[:, :], int8, int8, boolean))
def correct_coords_fast(all_map, default_coords, x_pos, y_pos, check_free=True):
    # correct to point
    actual_coord = default_coords + np.array([x_pos, y_pos])

    # check bounds
    actual_coord = actual_coord[(actual_coord[:, 0] >= 0) & (actual_coord[:, 1] >= 0)
                                & (actual_coord[:, 0] < 80) & (actual_coord[:, 1] < 80)]

    if check_free:
        mask = np.full(actual_coord.shape[0], False)
        for i, (x, y) in enumerate(actual_coord):
            mask[i] = (all_map[x, y, 0] == 0)
        actual_coord = actual_coord[mask]

    return actual_coord


@njit(void(int16[:, :, :], float64[:, :]))
def fill_res_map(all_map, res_go_map):
    hq = []
    visited = np.full((80, 80), False)

    forest_ind = np.argwhere(res_go_map == 0)
    if not forest_ind.shape[0]:
        return

    for x, y in forest_ind:
        hq.append((0, x, y))
        visited[x, y] = True

    xy_ar = np.zeros(2, dtype=np.int64)
    ways = np.array([(+1, 0), (0, +1), (-1, 0), (0, -1)], dtype=np.int64)

    while hq:
        d, x, y = heappop(hq)
        xy_ar[0] = x
        xy_ar[1] = y

        new_ways = ways + xy_ar
        for x_new, y_new in new_ways:

            if 0 <= x_new <= 79 and 0 <= y_new <= 79:

                if visited[x_new, y_new]:
                    continue
                else:
                    visited[x_new, y_new] = True
                    val_map = all_map[x_new, y_new, 0]
                    if val_map not in (0, -1) \
                            or res_go_map[x_new, y_new] == -1:
                        continue

                d_new = d + (1 if val_map == 0 else 2)
                if res_go_map[x_new, y_new] > d_new:
                    res_go_map[x_new, y_new] = d_new

                heappush(hq, (d_new, x_new, y_new))


@njit(void(int16[:, :, :], float64[:, :]))
def fill_enemy_map(all_map, enemy_go_map):
    hq = []
    visited = np.full((80, 80), False)

    enemy_ind = np.argwhere(enemy_go_map == 0)
    for x, y in enemy_ind:
        hq.append((0, x, y))
        visited[x, y] = True

    xy_ar = np.zeros(2, dtype=np.int64)
    ways = np.array([(+1, 0), (0, +1), (-1, 0), (0, -1)], dtype=np.int64)

    while hq:
        d, x, y = heappop(hq)
        xy_ar[0] = x
        xy_ar[1] = y

        new_ways = ways + xy_ar
        for x_new, y_new in new_ways:

            if 0 <= x_new <= 79 and 0 <= y_new <= 79:

                if visited[x_new, y_new]:
                    continue
                else:
                    visited[x_new, y_new] = True
                    val_map = all_map[x_new, y_new, 0]
                    if val_map not in (0, -1, 90, 8) \
                            or enemy_go_map[x_new, y_new] == -1:
                        continue

                if val_map == 0:
                    w = 1
                elif val_map == -1:
                    w = 2
                elif val_map == 8:
                    w = 2
                elif val_map == 90:
                    w = all_map[x_new, y_new, 4] // 5

                d_new = d + w
                if enemy_go_map[x_new, y_new] > d_new:
                    enemy_go_map[x_new, y_new] = d_new

                heappush(hq, (d_new, x_new, y_new))


@njit(void(int16[:, :, :], float64[:, :]))
def fill_friends_map(all_map, friends_go_map):
    hq = []
    visited = np.full((80, 80), False)

    enemy_ind = np.argwhere(friends_go_map == 0)
    for x, y in enemy_ind:
        hq.append((0.0, x, y))
        visited[x, y] = True

    xy_ar = np.zeros(2, dtype=np.int64)
    ways = np.array([(+1, 0), (0, +1), (-1, 0), (0, -1)], dtype=np.int64)

    while hq:
        d, x, y = heappop(hq)
        xy_ar[0] = x
        xy_ar[1] = y

        new_ways = ways + xy_ar
        for x_new, y_new in new_ways:

            if 0 <= x_new <= 79 and 0 <= y_new <= 79:

                if visited[x_new, y_new]:
                    continue
                else:
                    visited[x_new, y_new] = True
                    val_map = all_map[x_new, y_new, 0]
                    if val_map not in (0, -1, 90, 8, 6) \
                            or friends_go_map[x_new, y_new] == -1:
                        continue

                if val_map <= 0:
                    w = 1
                elif val_map <= 8:
                    w = 0.5
                elif val_map == 90:
                    w = all_map[x_new, y_new, 4] // 10

                d_new = d + w
                if friends_go_map[x_new, y_new] > d_new:
                    friends_go_map[x_new, y_new] = d_new

                heappush(hq, (d_new, x_new, y_new))


@njit(void(int16[:, :, :], float64[:, :], int8[:, :], int8[:, :], int8[:, :]))
def fill_control_res_poles(all_map, res_go_map, range_control, melee_control, turret_control):
    set_id = set()

    for x in range(80):
        for y in range(80):
            if all_map[x, y, 0] == 90:
                res_go_map[x, y] = 0
                continue
            elif all_map[x, y, 0] == 80:
                default_coords = range_control
            elif all_map[x, y, 0] == 60:
                default_coords = melee_control
            elif all_map[x, y, 0] == 100:

                id = all_map[x, y, 1]
                if id in set_id:
                    continue
                else:
                    set_id.add(id)

                default_coords = turret_control
            else:
                continue

            actual_coord = correct_coords_fast(all_map, default_coords, x, y, False)

            for xp, yp in actual_coord:
                res_go_map[xp, yp] = -1


@njit(void(int16[:, :, :], float64[:, :], int8[:, :], int8[:, :], int8[:, :], int8[:, :]))
def fill_enemy_poles(all_map, enemy_go_map, range_attack, turret_attack, melee_control, turret_control):
    set_id = set()

    for x in range(80):
        for y in range(80):
            pole = 0

            if all_map[x, y, 0] == 80 \
                    or all_map[x, y, 0] == 60:

                default_coords = range_attack

            elif all_map[x, y, 0] == 40:

                default_coords = melee_control

            elif all_map[x, y, 0] == 100:

                id = all_map[x, y, 1]
                if id in set_id:
                    continue
                else:
                    set_id.add(id)

                if all_map[x, y, 2] == 1:
                    # is broken
                    default_coords = turret_attack
                else:
                    # check ready for attack
                    num_ru = 0
                    control_coord = correct_coords_fast(all_map, turret_control, x, y, False)
                    for xc, yc in control_coord:
                        if all_map[xc, yc, 0] == 80:
                            num_ru += 1

                    if num_ru >= 10:
                        # kill it!!!
                        default_coords = turret_attack
                    else:
                        # stop here!
                        default_coords = turret_control
                        pole = -1

            else:
                continue

            actual_coord = correct_coords_fast(all_map, default_coords, x, y, False)

            for xp, yp in actual_coord:
                enemy_go_map[xp, yp] = pole


@njit(int16[:, :](int16[:, :, :], int8[:, :], int64, int64))
def get_precision_attack_map(all_map, range_attack, x_pos, y_pos):
    attack_coord = correct_coords_fast(all_map, range_attack, x_pos, y_pos, False)

    n = attack_coord.shape[0]
    m = all_map.shape[2]
    attack_map = np.zeros((n, m), dtype=np.int16)

    for i in range(n):
        x, y = attack_coord[i]
        for j in range(m):
            attack_map[i, j] = all_map[x, y, j]

    # get health mask
    health_mask = attack_map[:, 4] > 0

    # get enemy mask (MELEE_UNIT, RANGED_UNIT)
    attack_mask = health_mask & ((attack_map[:, 0] == 80) | (attack_map[:, 0] == 60))

    if not attack_mask.any():
        # get enemy mask (BUILDER_UNIT)
        attack_mask = health_mask & (attack_map[:, 0] == 40)

    return attack_map[attack_mask]


@njit(float64[:](float64[:, :], int8[:, :]))
def get_values_by_coord(val_map, coords):

    values = np.zeros(coords.shape[0], dtype=np.float64)

    for i, (x, y) in enumerate(coords):
        values[i] = val_map[x, y]

    return values


@njit(int16[:](int16[:, :, :], float64[:, :], float64[:, :], float64[:, :], int8[:, :], int8[:, :], int64, int64, boolean))
def get_best_way_fast(all_map,
                      res_go_map,
                      enemy_go_map,
                      friend_go_map,
                      near_attack,
                      range_control,
                      x,
                      y,
                      res_type=True):

    if res_type:
        go_map = res_go_map
    else:
        go_map = enemy_go_map

    unit_coord = correct_coords_fast(all_map, near_attack, x, y, False)
    near_ways = get_values_by_coord(go_map, unit_coord)

    free_mask = near_ways > 0  # not 0 and -1
    near_ways_free = near_ways[free_mask]
    f = near_ways_free.shape[0]

    if len(near_ways_free):
        if res_type:
            ind = np.argmin(near_ways_free)
        else:
            need_to_run = False

            near_friends = get_values_by_coord(friend_go_map, unit_coord)
            near_friends_free = near_friends[free_mask]

            # check area for friends
            x_min, y_min = max(0, x-1), max(0, y-1)
            if np.sum(all_map[x_min:x+2, y_min: y+2, 0] == 8) == 1:
                # check area for enemies
                num_enemies = 0
                control_coord = correct_coords_fast(all_map, range_control, x, y, False)
                for xc, yc in control_coord:
                    if all_map[xc, yc, 0] in (60, 80):
                        num_enemies += 1

                if num_enemies > 1:
                    need_to_run = True

            if not need_to_run:
                # find best way to enemy and friend both
                hq = [(np.float64(1), np.float64(1), np.int64(1)) for _ in range(0)]
                for i in range(f):
                    heappush(hq, (near_ways_free[i], near_friends_free[i], i))
                de, df, ind = heappop(hq)
            else:
                ind = np.argmin(near_friends_free)

        x_new, y_new = unit_coord[free_mask][ind]
        go_map[x_new, y_new] = np.inf
        d = near_ways_free[ind]
        go_map[x, y] = d + 1
    elif res_type:
        x_new, y_new = 0, 0
    else:
        x_new, y_new = 70, 70

    return np.array((x_new, y_new), dtype=np.int16)


class MyStrategy:

    def __init__(self):
        self.time_dict = {"a": 0,
                          "p": 0,
                          "mc": 0,
                          "ud": 0,
                          "res": 0,
                          "fr": 0,
                          "en": 0,
                          "rep": 0,
                          "h": 0,
                          "base": 0,
                          "bu": 0,
                          "ru": 0}
        self.t_main = 0
        self.t_per = 0
        self.per_time = 50

        self.status_base = "WORK"
        self.cur_tick = 0
        self.max_population = 0
        self.my_resource = 0
        self.is_dark_near = False
        self.is_final = False

        self.type_resource = (EntityType.RESOURCE + 1) * 10
        self.type_builder = EntityType.BUILDER_UNIT + 1
        self.type_range = EntityType.RANGED_UNIT + 1

        self.type_enemy_range = (EntityType.RANGED_UNIT + 1) * 10
        self.type_enemy_melee = (EntityType.MELEE_UNIT + 1) * 10
        self.type_enemy_builder = (EntityType.BUILDER_UNIT + 1) * 10

        self.builders = {}
        self.range_units = {}
        self.max_builders = 50
        self.max_melees = 4
        self.coef_builders_to_houses = 0.2

        self.buildings_light = {}
        self.all_map = np.zeros((80, 80, 5), dtype=np.int16)  # type, id, small health, type of work, health

        self.res_go_map = np.full((80, 80), np.inf)
        self.enemy_go_map = np.full((80, 80), np.inf)
        self.friend_go_map = np.full((80, 80), np.inf)

        self.zero_area = np.ones((80, 80), dtype=np.int8)
        self.res_area = np.zeros((80, 80), dtype=np.int8)

        near_base_coord = np.array([(x, y) if z else (y, x)
                                    for z in range(2)
                                    for x in [-1, 5]
                                    for y in range(5)], dtype=np.int8)

        near_house_coord = np.array([(x, y) if z else (y, x)
                                     for z in range(2)
                                     for x in [-1, 3]
                                     for y in range(3)], dtype=np.int8)

        near_turret_coord = np.array([(x, y) if z else (y, x)
                                      for z in range(2)
                                      for x in [-1, 2]
                                      for y in range(2)], dtype=np.int8)

        house_coord = np.array(([[0, 0]] + [[x, 1] if z else [1, x]
                                            for x in range(5, 33, 3)
                                            for z in range(2)
                                            if not (z == 1 and x == 0)]
                                + [[11, 11], [16, 11], [11, 16], [21, 11], [11, 21]]
                                + [[25, 11], [11, 25]]
                                + [[17, 6], [22, 6], [6, 17], [6, 22]]
                                + [[5, 2], [2, 5]]
                                ), dtype=np.int8)

        self.buildings = {EntityType.RANGED_BASE: {"coords": np.array([[10, 5], [5, 10], [5, 15]], dtype=np.int8),
                                                   "near_coords": near_base_coord,
                                                   "size": 5,
                                                   "cost": 500},
                          EntityType.HOUSE: {"coords": house_coord,
                                             "near_coords": near_house_coord,
                                             "size": 3,
                                             "cost": 50},
                          EntityType.BUILDER_BASE: {"coords": np.array([[5, 5]], dtype=np.int8),
                                                    "near_coords": near_base_coord,
                                                    "size": 5,
                                                    "cost": 500},
                          EntityType.MELEE_BASE: {"coords": np.array([[5, 15]], dtype=np.int8),
                                                  "near_coords": near_base_coord,
                                                  "size": 5,
                                                  "cost": 500},
                          EntityType.TURRET: {"coords": np.array([[15, 15], [6, 27], [27, 6]], dtype=np.int8),
                                              "near_coords": near_turret_coord,
                                              "size": 2,
                                              "cost": 50}
                          }

        self.near_attack = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8)

        self.range_attack = get_unit_area(5, 1)
        self.range_control = get_unit_area(6, 1)

        self.melee_attack = get_unit_area(1, 1)
        self.melee_control = get_unit_area(2, 1)

        self.turret_attack = get_unit_area(4, 2)
        self.turret_control = get_unit_area(5, 2)

        self.base_attack = get_unit_area(4, 5)
        self.house_attack = get_unit_area(4, 3)

    def debug_update(self, player_view, debug_interface):

        debug_interface.send(Clear())

        # debug_interface.send(SetAutoFlush(False))
        # color = Color(255.0, 0.0, 0.0, 1.0)
        # offset = Vec2Float(0, 0)
        #
        # for entity in player_view.entities:
        #     properties = player_view.entity_properties[entity.entity_type]
        #     if properties.can_move:
        #         cv = ColoredVertex(Vec2Float(entity.position.x + 0.85, entity.position.y + 0.15), offset, color)
        #         text = PlacedText(cv, str(entity.id), 1, 12.0)
        #         debug_interface.send(Add(text))
        #
        # debug_interface.send(Flush())

        debug_interface.get_state()

    def find_nearest_manhattan(self, x, y, type_num):
        rows, cols = np.where(self.all_map[:, :, 0] == type_num)

        if len(rows):
            manh_arr = np.abs(rows - x) + np.abs(cols - y)
            min_idx = manh_arr.argmin()
            x, y, manh_dist = rows[min_idx], cols[min_idx], manh_arr[min_idx]
        else:
            x, y, manh_dist = 70, 70, 70

        return x, y, manh_dist

    def collect_data_for_attack(self, attack_map, precision_attack, counter_unit, ru_id):
        for info in attack_map:
            id_en = info[1]
            if id_en not in precision_attack:
                precision_attack[id_en] = {"health": info[4], "my_ens": []}

            precision_attack[id_en]["my_ens"].append(ru_id)
            counter_unit[ru_id] += 1

    def get_action(self, player_view, debug_interface):

        start_all = time.time()

        result = Action({})
        my_id = player_view.my_id

        list_melee_units = []
        list_range_units = []
        list_builders_units = []
        list_broken_buildings = []
        list_turrets = []

        my_buildings = defaultdict(list)

        dict_broken_buildings = defaultdict(int)
        dict_coord_bilder_to_houses = {}

        num_builders_to_houses = 0
        num_repairs_to_houses = 0

        self.cur_tick = player_view.current_tick

        self.all_map.fill(0)
        self.res_go_map.fill(np.inf)
        self.enemy_go_map.fill(np.inf)
        if self.is_dark_near:
            self.friend_go_map.fill(np.inf)

        self.max_population = 0
        self.my_resource = [val.resource for val in player_view.players if val.id == my_id][0]

        resource_here = False
        enemy_here = False

        dark_map = None
        watch_in_dark = self.cur_tick == 1 \
                        or self.is_dark_near and self.cur_tick < 900
        if watch_in_dark:
            dark_map = np.ones((80, 80), dtype=np.int8)

        if self.cur_tick == 1 and len(player_view.players) == 2:
            self.is_final = True
            self.max_builders = 80

        start = time.time()
        for entity in player_view.entities:

            ent_type = entity.entity_type
            properties = player_view.entity_properties[ent_type]

            its_my = entity.player_id == my_id
            is_broken = entity.health < properties.max_health

            pos = entity.position
            size = properties.size
            sight_range = properties.sight_range
            type_num = (ent_type + 1) * (1 if its_my else 10)

            self.all_map[pos.x: pos.x + size, pos.y: pos.y + size, 0] = type_num
            self.all_map[pos.x: pos.x + size, pos.y: pos.y + size, 1] = entity.id
            self.all_map[pos.x: pos.x + size, pos.y: pos.y + size, 4] = entity.health
            if is_broken:
                self.all_map[pos.x: pos.x + size, pos.y: pos.y + size, 2] = 1

            if not its_my:

                enemy_here = enemy_here or ent_type != EntityType.RESOURCE
                resource_here = resource_here or ent_type == EntityType.RESOURCE

                if size >= 3:
                    self.enemy_go_map[pos.x: pos.x + size, pos.y: pos.y + size] = 0

                continue

            if watch_in_dark:
                self.update_dark_map(dark_map, size, sight_range, pos.x, pos.y)

            if not properties.can_move:
                # buildings
                if is_broken:
                    list_broken_buildings.append(((pos.x, pos.y), entity.id))
                    if not self.is_dark_near:
                        dict_broken_buildings[(pos.x, pos.y)] += size // 2 + 1  # 2 for house, 3 for base
                    else:
                        dict_broken_buildings[(pos.x, pos.y)] += size if size <= 3 else 10

                if properties.population_provide:
                    my_buildings[ent_type].append(entity)
                    self.max_population += properties.population_provide if not is_broken else 0

            if ent_type == EntityType.TURRET:

                list_turrets.append(entity)

            elif ent_type == EntityType.BUILDER_UNIT:

                list_builders_units.append(entity)

                if entity.id in self.builders:

                    type_oper, coord_oper, coord_to_go = self.builders[entity.id]

                    if type_oper in self.buildings:
                        # check house
                        x_oper, y_oper = coord_oper
                        if self.all_map[x_oper, y_oper, 0] != (type_oper + 1):
                            num_builders_to_houses += 1
                            self.all_map[pos.x, pos.y, 3] = 1
                            dict_coord_bilder_to_houses[coord_oper] = True

                            # correct resources to buildings plan
                            self.my_resource -= min(self.buildings[type_oper]["cost"], self.my_resource)

                    elif type_oper == "Repair":

                        num_repairs_to_houses += 1
                        self.all_map[pos.x, pos.y, 3] = 2
                        dict_broken_buildings[coord_oper] -= 1

            elif ent_type == EntityType.RANGED_UNIT:

                list_range_units.append(entity)
                self.friend_go_map[pos.x, pos.y] = 0

            elif ent_type == EntityType.MELEE_UNIT:

                list_melee_units.append(entity)
                self.friend_go_map[pos.x, pos.y] = 0

        self.time_dict["mc"] += (time.time() - start) * 1000

        if self.cur_tick == 1:
            self.is_dark_near = EntityType.RANGED_BASE not in my_buildings

        start = time.time()

        if watch_in_dark:
            first_layer = self.all_map[:, :, 0]
            dark_mask = (dark_map == 1)
            first_layer[dark_mask] = -1

            self.zero_area[(first_layer <= 10) & (first_layer >= 0)] = 0
            first_layer[(dark_mask) & (self.zero_area == 0)] = 0

            self.res_area[self.zero_area == 0] = 0
            self.res_area[first_layer == self.type_resource] = 1
            self.res_go_map[(dark_mask) & (self.res_area == 1)] = 0

        self.time_dict["ud"] += (time.time() - start) * 1000
        start = time.time()

        # RESOURCE MAP
        if not resource_here:
            self.res_go_map[75, 75] = 0
            self.res_go_map[5, 75] = 0
            self.res_go_map[75, 5] = 0

        fill_control_res_poles(self.all_map,
                               self.res_go_map,
                               self.range_control,
                               self.melee_control,
                               self.turret_control)
        fill_res_map(self.all_map, self.res_go_map)

        self.time_dict["res"] += (time.time() - start) * 1000
        start = time.time()

        # FRIEND MAP
        if self.is_dark_near \
                and (len(list_range_units) + len(list_melee_units)) > 2:
            fill_friends_map(self.all_map, self.friend_go_map)

        self.time_dict["fr"] += (time.time() - start) * 1000
        start = time.time()

        # ENEMY MAP
        if not enemy_here:
            Virt_Pos = namedtuple("Virt_Pos", "x y")

            if self.cur_tick <= 500 \
                    and np.sum(self.all_map[30:50, 30:50, 0] == self.type_range) < 10:
                self.draw_enemy_pole(self.range_control, Virt_Pos(40, 40), 0)

            if np.sum(self.zero_area[70:, 70:]) > 5:
                self.draw_enemy_pole(self.range_attack, Virt_Pos(75, 75), 0)
            else:
                self.draw_enemy_pole(self.melee_attack, Virt_Pos(5, 75), 0)
                self.draw_enemy_pole(self.melee_attack, Virt_Pos(75, 5), 0)

        fill_enemy_poles(self.all_map,
                         self.enemy_go_map,
                         self.range_attack,
                         self.turret_attack,
                         self.melee_control,
                         self.turret_control)

        if list_range_units:
            self.enemy_go_map[self.all_map[:, :, 0] == self.type_resource] = np.inf
            fill_enemy_map(self.all_map, self.enemy_go_map)

        self.time_dict["en"] += (time.time() - start) * 1000

        len_bu = len(list_builders_units)
        all_units = len_bu + len(list_range_units) + len(list_melee_units)

        self.check_base_status()

        start = time.time()

        # REPAIR
        result_broken_buildings = [(coord, bu_id, dict_broken_buildings[coord]) for (coord, bu_id) in
                                   list_broken_buildings if dict_broken_buildings[coord] >= 1]
        if result_broken_buildings and (num_repairs_to_houses // 2) < len(result_broken_buildings):
            self.repair_buildings(result_broken_buildings, result)

        self.time_dict["rep"] += (time.time() - start) * 1000
        start = time.time()

        # NEW BUILDINGS
        max_builders_for_houses = round(len_bu * self.coef_builders_to_houses)
        max_builders_for_houses = max(0, max_builders_for_houses - len(dict_coord_bilder_to_houses))
        max_houses_to_build = max_builders_for_houses if (all_units >= self.max_population) else 0
        if max_houses_to_build:
            self.build_houses(my_buildings, max_houses_to_build, dict_coord_bilder_to_houses, result)

        self.time_dict["h"] += (time.time() - start) * 1000
        start = time.time()

        # BUILDER BASE
        need_to_build = len_bu < self.max_builders \
                        and self.my_resource // 10 >= 1
        self.builder_base_action(my_buildings, need_to_build, result)

        # RANGE BASE
        self.range_base_action(my_buildings, result)

        # MELEE BASE
        need_to_build = len(list_melee_units) < self.max_melees \
                        and len(list_range_units) > 10 \
                        and self.my_resource // 20 >= 1
        self.melee_base_action(my_buildings, need_to_build, result)

        self.time_dict["base"] += (time.time() - start) * 1000

        start = time.time()

        if list_builders_units:
            self.builder_unit_action(list_builders_units, result)

        self.time_dict["bu"] += (time.time() - start) * 1000

        start = time.time()

        if list_range_units:
            self.range_unit_action(list_range_units, result)

        self.time_dict["ru"] += (time.time() - start) * 1000

        if list_melee_units:
            self.melee_unit_action(list_melee_units, result)

        if list_turrets:
            self.turret_unit_action(list_turrets, result)

        end_all = time.time()
        self.time_dict["a"] += (end_all - start_all) * 1000
        self.time_dict["p"] += (end_all - start_all) * 1000

        if self.cur_tick and not self.cur_tick % self.per_time:
            res_t = f"{self.cur_tick:3d} = a:{self.time_dict['a']:7.1f}"

            for k in self.time_dict:
                if k == "a":
                    continue
                val = self.time_dict[k]
                res_t += f", {k}:{val:7.1f}"
                self.time_dict[k] = 0

            print(res_t)

        return result

    def build_houses(self, my_buildings, max_houses_to_build, dict_coord_bilder_to_houses, result):

        if self.status_base == "WAR" \
                and self.cur_tick % 5:
            return

        free_builders_arr = np.argwhere((self.all_map[:, :, 0] == self.type_builder) & (self.all_map[:, :, 3] == 0))
        num_free_builders = free_builders_arr.shape[0]
        if not num_free_builders:
            return

        cur_resource = self.my_resource

        num_houses = len(my_buildings[EntityType.HOUSE])
        num_ranged_bases = len(my_buildings[EntityType.RANGED_BASE])

        for type_building, building_property in self.buildings.items():

            list_buildings = my_buildings[type_building]

            if self.is_dark_near:
                # need for ranged base!
                if not num_ranged_bases \
                        and type_building != EntityType.RANGED_BASE \
                        and (type_building != EntityType.HOUSE
                             or num_houses > 4):
                    continue

            building_coords = building_property["coords"]
            near_house_coords = building_property["near_coords"]
            size = building_property["size"]
            cost = building_property["cost"]

            if self.is_final:
                if type_building == EntityType.MELEE_BASE \
                        or (type_building == EntityType.TURRET and num_ranged_bases < 2) \
                        or type_building == EntityType.RANGED_BASE and num_ranged_bases > 1:
                    continue
            elif cost == 500 and list_buildings:
                continue

            max_houses_by_resource = cur_resource // cost
            if not max_houses_by_resource:
                continue

            if type_building != EntityType.HOUSE:
                max_houses_to_build = 1

            max_houses_to_build = min(max_houses_to_build, max_houses_by_resource)

            if len(list_buildings) == len(building_coords) \
                    or not max_houses_to_build:
                continue

            for x_house, y_house in building_coords:

                if (x_house, y_house) in dict_coord_bilder_to_houses:
                    continue

                free_place_for_house = not self.all_map[x_house:x_house + size, y_house:y_house + size, 0].any()
                if not free_place_for_house:
                    continue

                near_house_coord = correct_coords_fast(self.all_map, near_house_coords, x_house, y_house, True)

                if len(near_house_coord):

                    # find nearest builder
                    ind_build, ind_coord = find_best_manh_for_arrays(free_builders_arr, near_house_coord)

                    x_builder, y_builder = free_builders_arr[ind_build]
                    if x_builder == 999:
                        break

                    builder_id = self.all_map[x_builder, y_builder, 1]

                    # drop builder from next calc
                    free_builders_arr[ind_build, 0] = 999
                    free_builders_arr[ind_build, 1] = 999

                    x_move, y_move = near_house_coord[ind_coord]

                    result.entity_actions[builder_id] = EntityAction(
                        MoveAction(Vec2Int(x_move, y_move), True, True),
                        BuildAction(type_building, Vec2Int(x_house, y_house)),
                        None,
                        None)

                    # set current operation
                    self.builders[builder_id] = (type_building, (x_house, y_house), (x_move, y_move))
                    self.all_map[x_builder, y_builder, 3] = 1

                    cur_resource -= cost
                    num_free_builders -= 1
                    max_houses_to_build -= 1
                    if not max_houses_to_build \
                            or not num_free_builders \
                            or cur_resource <= 0:
                        break

    def correct_build_repair_operation(self, builder_id, builder_coord, result):

        type_oper_building, (x_building, y_building), (x_move, y_move) = self.builders[builder_id]

        if type_oper_building == "Repair":
            type_building = self.all_map[x_building, y_building, 0] - 1
        else:
            type_building = type_oper_building

        near_house_coords = self.buildings[type_building]["near_coords"]
        x_builder, y_builder = builder_coord
        in_place = (x_builder == x_move) and (y_builder == y_move)

        if not in_place and self.all_map[x_move, y_move, 0] != 0:
            # lets check free place to go again
            near_house_coord = correct_coords_fast(self.all_map, near_house_coords, x_building, y_building, True)

            if not len(near_house_coord):
                return

            builder_xy_arr = np.array(builder_coord).reshape(1, 2)
            ind_coord, _ = find_best_manh_for_arrays(near_house_coord, builder_xy_arr)
            x_move, y_move = near_house_coord[ind_coord]

            # update current operation
            self.builders[builder_id] = (type_oper_building, (x_building, y_building), (x_move, y_move))

        move_action = MoveAction(Vec2Int(x_move, y_move), True, True)

        if type_oper_building == "Repair":
            building_id = self.all_map[x_building, y_building, 1]

            result.entity_actions[builder_id] = EntityAction(
                move_action,
                None,
                None,
                RepairAction(building_id))
        else:
            result.entity_actions[builder_id] = EntityAction(
                move_action,
                BuildAction(type_building, Vec2Int(x_building, y_building)),
                None,
                None)

    def repair_buildings(self, list_broken_buildings, result):

        free_builders_arr = np.argwhere((self.all_map[:, :, 0] == self.type_builder) & (self.all_map[:, :, 3] == 0))
        num_free_builders = free_builders_arr.shape[0]
        if not num_free_builders:
            return

        for (x_building, y_building), building_id, units_per_house in list_broken_buildings:

            building_type = self.all_map[x_building, y_building, 0] - 1
            near_house_coords = self.buildings[building_type]["near_coords"]
            building_point = np.array([[x_building, y_building]])

            near_house_coord = correct_coords_fast(self.all_map, near_house_coords, x_building, y_building, True)

            for _ in range(units_per_house):

                x_move, y_move = None, None

                if len(near_house_coord):
                    # find nearest builder and near coord
                    ind_build, ind_coord = find_best_manh_for_arrays(free_builders_arr, near_house_coord)
                    x_move, y_move = near_house_coord[ind_coord]

                    # drop coord from next calc
                    near_house_coord[ind_coord, 0] = -999
                    near_house_coord[ind_coord, 1] = -999

                if x_move is None or x_move == -999:
                    # find nearest builder to building point
                    ind_build, _ = find_best_manh_for_arrays(free_builders_arr, building_point)
                    x_move, y_move = x_building, y_building

                x_builder, y_builder = free_builders_arr[ind_build]
                builder_id = self.all_map[x_builder, y_builder, 1]

                # drop builder from next calc
                free_builders_arr[ind_build, 0] = 999
                free_builders_arr[ind_build, 1] = 999

                result.entity_actions[builder_id] = EntityAction(
                    MoveAction(Vec2Int(x_move, y_move), True, True),
                    None,
                    None,
                    RepairAction(building_id))

                num_free_builders -= 1
                if not num_free_builders:
                    return

                self.builders[builder_id] = ("Repair", (x_building, y_building), (x_move, y_move))
                self.all_map[x_builder, y_builder, 3] = 2

    def builder_base_action(self, my_buildings, need_to_build, result):

        if self.status_base == "WAR" \
                and self.cur_tick % 3:
            need_to_build = False

        build_action = None
        unit_base_coord = self.buildings[EntityType.BUILDER_BASE]["near_coords"]

        for builder_base in my_buildings[EntityType.BUILDER_BASE]:
            if need_to_build:
                build_arr = unit_base_coord + np.array([builder_base.position.x, builder_base.position.y])
                # check free
                build_arr = build_arr[self.all_map[build_arr.T[0], build_arr.T[1], 0] == 0]
                if not build_arr.shape[0]:
                    continue

                moves = self.res_go_map[build_arr.T[0], build_arr.T[1]]
                ind_min_move = np.argmin(moves)

                if moves[ind_min_move] != np.inf:
                    x, y = build_arr[ind_min_move]
                else:
                    x, y = build_arr[np.sum(build_arr, axis=1).argmax()]

                build_action = BuildAction(EntityType.BUILDER_UNIT, Vec2Int(x, y))

            result.entity_actions[builder_base.id] = EntityAction(
                None,
                build_action,
                None,
                None)

    def range_base_action(self, my_buildings, result):

        build_action = None
        unit_base_coord = self.buildings[EntityType.BUILDER_BASE]["near_coords"]

        for range_base in my_buildings[EntityType.RANGED_BASE]:
            build_arr = unit_base_coord + np.array([range_base.position.x, range_base.position.y])

            # check free
            build_arr = build_arr[self.all_map[build_arr.T[0], build_arr.T[1], 0] == 0]
            if not build_arr.shape[0]:
                self.status_base = "WAR"
                continue

            moves = self.enemy_go_map[build_arr.T[0], build_arr.T[1]]
            ind_min_move = np.argmin(moves)

            if moves[ind_min_move] != np.inf:
                x, y = build_arr[ind_min_move]
            else:
                x, y = build_arr[np.sum(build_arr, axis=1).argmax()]

            build_action = BuildAction(
                EntityType.RANGED_UNIT,
                Vec2Int(x, y))

            result.entity_actions[range_base.id] = EntityAction(
                None,
                build_action,
                None,
                None)

    def melee_base_action(self, my_buildings, need_to_build, result):

        build_action = None
        for melee_base in my_buildings[EntityType.MELEE_BASE]:

            if need_to_build:
                build_action = BuildAction(
                    EntityType.MELEE_UNIT,
                    Vec2Int(melee_base.position.x + 5, melee_base.position.y + 3))

            result.entity_actions[melee_base.id] = EntityAction(
                None,
                build_action,
                None,
                None)

    def builder_unit_action(self, list_builders_units, result):

        for builder in list_builders_units:

            x_bu, y_bu = builder.position.x, builder.position.y

            i_need_to_run = False

            if builder.id in self.builders:
                # check operation
                type_oper, (x_oper, y_oper), (x_move, y_move) = self.builders[builder.id]

                if type_oper == self.type_resource:
                    builder_attack_coord = correct_coords_fast(self.all_map, self.near_attack, x_bu, y_bu, False)
                    attack_map = self.all_map[builder_attack_coord[:, 0], builder_attack_coord[:, 1]]

                    res_map = self.res_go_map[builder_attack_coord[:, 0], builder_attack_coord[:, 1]]
                    i_need_to_run = (res_map == -1).any()

                    if (attack_map[:, 0] == type_oper).any() \
                            and not i_need_to_run:
                        # dig next
                        continue

                elif type_oper in self.buildings:
                    # check house
                    if self.all_map[x_oper, y_oper, 0] != (type_oper + 1):
                        if self.all_map[x_oper, y_oper, 0] == 0:
                            # on the way to work
                            self.correct_build_repair_operation(builder.id, (x_bu, y_bu), result)
                            continue
                        else:
                            # tuman mistake. need to work
                            pass
                    else:
                        # check health near
                        if self.all_map[x_oper, y_oper, 2] > 0:
                            result.entity_actions[builder.id] = EntityAction(
                                MoveAction(Vec2Int(x_oper, y_oper), True, True),
                                None,
                                None,
                                RepairAction(self.all_map[x_oper, y_oper, 1]))

                            self.builders[builder.id] = ("Repair", (x_oper, y_oper), (x_oper, y_oper))
                            self.all_map[x_bu, y_bu, 3] = 2
                            continue

                elif type_oper == "Repair":
                    # check health
                    if self.all_map[x_oper, y_oper, 2] > 0:
                        self.correct_build_repair_operation(builder.id, (x_bu, y_bu), result)
                        continue

            x_res, y_res = get_best_way_fast(self.all_map,
                                             self.res_go_map,
                                             self.enemy_go_map,
                                             self.friend_go_map,
                                             self.near_attack,
                                             self.range_control,
                                             x_bu,
                                             y_bu,
                                             True)

            if i_need_to_run:
                attack = None
            else:
                attack = AttackAction(None, AutoAttack(1, [EntityType.RESOURCE, EntityType.BUILDER_UNIT]))

            result.entity_actions[builder.id] = EntityAction(
                MoveAction(Vec2Int(x_res, y_res), True, True),
                None,
                attack,
                None)

            self.builders[builder.id] = (self.type_resource, (x_res, y_res), (x_res, y_res))

    def range_unit_action(self, list_range_units, result):

        # check group to go
        group_area = self.all_map[:22, :22]
        group_mask = group_area[:, :, 0] == (EntityType.RANGED_UNIT + 1)
        go_go = np.sum(group_mask) >= 15
        ids_to_go = group_area[group_mask, 1]

        x, y = 70, 70
        auto_attack_len = 5
        precision_attack = {}
        counter_unit = defaultdict(int)

        for range_unit in list_range_units:
            x_pos, y_pos = range_unit.position.x, range_unit.position.y

            id_enemy = None

            attack_map = get_precision_attack_map(self.all_map, self.range_attack, x_pos, y_pos)
            if attack_map.any():
                self.collect_data_for_attack(attack_map, precision_attack, counter_unit, range_unit.id)
                continue

            oper = "Group"
            if self.status_base == "WAR":
                oper = "Attack"
            elif range_unit.id in self.range_units:
                oper, _ = self.range_units[range_unit.id]
                if oper == "Group" \
                        and go_go \
                        and range_unit.id in ids_to_go:
                    oper = "Attack"

            if oper == "Group":
                auto_attack_len = 30
                x_go, y_go = x_pos, y_pos
                move = None
            else:
                x_go, y_go = get_best_way_fast(self.all_map,
                                               self.res_go_map,
                                               self.enemy_go_map,
                                               self.friend_go_map,
                                               self.near_attack,
                                               self.range_control,
                                               x_pos,
                                               y_pos,
                                               False)
                move = MoveAction(Vec2Int(x_go, y_go), True, True)

            if id_enemy is None:
                attack = AttackAction(None, AutoAttack(auto_attack_len, []))
            else:
                attack = AttackAction(id_enemy, None)

            result.entity_actions[range_unit.id] = EntityAction(
                move,
                None,
                attack,
                None)

            self.range_units[range_unit.id] = (oper, (x_go, y_go))

        if precision_attack:
            precision_attack_sort = sorted(precision_attack.items(),
                                           key=lambda data: (5 * len(data[1]["my_ens"]) != data[1]["health"],
                                                             data[1]["health"],
                                                             len(data[1]["my_ens"])))

            used_ru = set()
            unused_ru = set()

            default_info = ("Attack", (x, y))
            default_move = MoveAction(Vec2Int(x, y), True, True)

            for id_enemy, data in precision_attack_sort:
                health = data["health"]
                my_ens_sort = sorted(data["my_ens"], key=lambda x: counter_unit[x])
                for id_ru in my_ens_sort:
                    if id_ru in used_ru:
                        continue
                    if health:
                        result.entity_actions[id_ru] = EntityAction(None, None, AttackAction(id_enemy, None), None)
                        self.range_units[id_ru] = default_info

                        used_ru.add(id_ru)
                        unused_ru.discard(id_ru)
                        counter_unit[id_ru] -= 1

                        health -= 5
                    else:
                        unused_ru.add(id_ru)

            for id_ru in unused_ru:
                result.entity_actions[id_ru] = EntityAction(
                    default_move,
                    None,
                    AttackAction(None, AutoAttack(auto_attack_len, [])),
                    None)
                self.range_units[id_ru] = default_info

    def melee_unit_action(self, list_melee_units, result):

        for melee_unit in list_melee_units:
            result.entity_actions[melee_unit.id] = EntityAction(
                MoveAction(Vec2Int(14, 14), True, False),
                None,
                AttackAction(None, AutoAttack(20, [])),
                None)

    def turret_unit_action(self, list_turrets, result):

        for turret in list_turrets:
            x_pos, y_pos = turret.position.x, turret.position.y

            ind_enemy = None

            attack_coord = correct_coords_fast(self.all_map, self.turret_attack, x_pos, y_pos, False)

            # get map
            attack_map = self.all_map[attack_coord[:, 0], attack_coord[:, 1]]

            # get enemy mask
            attack_mask = np.isin(attack_map[:, 0], (self.type_enemy_range, self.type_enemy_melee)) \
                          & (attack_map[:, 4] > 0)

            if attack_mask.any():
                ind_enemy = np.argmin(attack_map[attack_mask, 4])
                id_enemy = attack_map[attack_mask][ind_enemy, 1]

            if ind_enemy is None:
                attack = AttackAction(None, AutoAttack(10, []))
            else:
                attack = AttackAction(id_enemy, None)

            result.entity_actions[turret.id] = EntityAction(None, None, attack, None)

    def check_base_status(self):
        d = 45
        if (self.all_map[:d, :d, 0] == self.type_enemy_range).any() \
                or (self.all_map[:d, :d, 0] == self.type_enemy_melee).any():

            self.status_base = "WAR"
        else:
            self.status_base = "WORK"

    def update_dark_map(self, dark_map, size, sight_range, x, y):

        key = (size, sight_range)

        if key not in self.buildings_light:
            light_mas = get_unit_area(sight_range, size)
            self.buildings_light[key] = light_mas
        else:
            light_mas = self.buildings_light[key]

        light_mas_cur = correct_coords_fast(self.all_map, light_mas, x, y, False)

        # update light
        dark_map[light_mas_cur.T[0], light_mas_cur.T[1]] = 0

    def draw_enemy_pole(self, coord, pos, pole):
        unit_coord = correct_coords_fast(self.all_map, coord, pos.x, pos.y, False)
        self.enemy_go_map[unit_coord[:, 0], unit_coord[:, 1]] = pole
