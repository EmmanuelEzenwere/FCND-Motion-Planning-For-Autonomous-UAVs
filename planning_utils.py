from enum import Enum
from udacidrone.frame_utils import local_to_global
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue
import numpy as np
import networkx as nx


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:

            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


def create_graph(data, k, start_node=None, goal_node=None, random_points=500):
    """
        creates a graph representation connecting given nodes along with weights of connecting edges.
        :param data: numpy array with 6 cols specifying the size of an obstacle.
        :param k: int, number of branches/ edges from each node to neighbouring nodes in the graph.
                  An important condition to observe is that k <= total number of nodes in the graph.
        :param start_node: tuple(x,y,z). Optional start location on the graph.
        :param goal_node: tuple(x, y, z). Optional goal location on the graph
        :param random_points: Sampling 500 random points as a default value.
        :return g: networkx weighted graph object.

    """
    # to generate random 3d points that do not collide with obstacles at positions specified in the given data.
    sampler = Sampler(data)
    polygons = sampler.polygons

    # list of points (tuples) eg. [(x0,y0,z0),...(xn, yn, zn)]
    print("sampling points...")
    nodes = sampler.sample(random_points)
    print("sample points.")
    print("random points func:", random_points)

    if start_node is not None:
        nodes[0] = start_node
    if goal_node is not None:
        nodes[-1] = goal_node

    g = nx.Graph()
    print("creating tree...")
    tree = KDTree(nodes)
    print("created tree")
    for n1 in nodes:
        # for each node connect try to connect to k nearest nodes
        idxs = tree.query([n1], k, return_distance=False)[0]
        for idx in idxs:
            n2 = nodes[idx]
            if n2 == n1:
                continue

            if can_connect(n1, n2, polygons):
                # include cost of moving from point n1 to n2 to be the distance between n1 and n2.
                # during A* Search Algorithm, this becomes the cost of moving along the edge connecting point n1 to n2.
                g.add_edge(n1, n2, weight=heuristic(n1, n2))
    return g


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)
    NORTHEAST = (-1, 1, np.around(np.sqrt(2), decimals=3))
    NORTHWEST = (-1, -1, np.around(np.sqrt(2), decimals=3))
    SOUTHWEST = (1, -1, np.around(np.sqrt(2), decimals=3))
    SOUTHEAST = (1, 1, np.around(np.sqrt(2), decimals=3))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return self.value[0], self.value[1]


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions_ = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions_.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions_.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions_.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions_.remove(Action.EAST)
    if x - 1 < 0 or y - 1 < 0 or grid[x - 1, y - 1] == 1:
        valid_actions_.remove(Action.NORTHWEST)
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1:
        valid_actions_.remove(Action.NORTHEAST)
    if x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] == 1:
        valid_actions_.remove(Action.SOUTHWEST)
    if x + 1 > n or y + 1 > m or grid[x + 1, y + 1] == 1:
        valid_actions_.remove(Action.SOUTHEAST)

    return valid_actions_


def a_star_grid(grid, h, start, goal):
    print("A* grid search running...")
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)
    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]

        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            # print("first actions:", valid_actions(grid, current_node))
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


def a_star_graph(graph, h, start, goal):
    """Modified A* to work with NetworkX graphs."""
    print("A* graph search running")
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:

        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')

    return path[::-1], path_cost


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def random_goal_search(grid, global_home, north_offset, east_offset):
    """
     Random Goal Search randomly selects an un-occupied point on the grid and converts to global_position given a
     global_home position and off-set values during co-ordinate translation.

    :param grid: numpy array occupancy grid of any size.
    :param global_home: reference global position (lon,lat,alt) on grid.
    :param north_offset: translation along the north during co-ordinate transformation to grid.
    :param east_offset: translation along the east during co-ordinate transformation to grid.
    :return global_position: np.array (lon, lat, alt) of random un-occupied point on grid.
    """
    # global_position_ref
    rows, cols = np.where(grid == 0)
    free_points = [(rows[i], cols[i]) for i in range(rows.shape[0])]

    # obtain indices of free states on the grid.
    random_index = np.random.randint(len(free_points))
    grid_pos = free_points[random_index]

    # relative distances to grid reference point: lat0, lon0
    local_north = grid_pos[0] + north_offset
    local_east = grid_pos[1] + east_offset

    local_position = (local_north, local_east, 0)
    global_position = local_to_global(local_position, global_home)

    return global_position


def point(p):
    if len(p) == 3:
        point_ = np.array([p[0], p[1], p[2]]).reshape(1, -1)
    else:
        point_ = np.array([p[0], p[1], 1.]).reshape(1, -1)

    return point_


def collinearity_check(p1, p2, p3, epsilon=1e-4):
    m = np.concatenate((p1, p2, p3), axis=0)
    area = np.linalg.det(m)
    return abs(area) < epsilon


def prune_path(path):
    assert len(path) != 1, "Error, path contains only one point."

    p1 = path[0]
    p2 = path[1]

    pruned_path = [p1, p2]

    for p in path[2:]:

        if collinearity_check(point(p1), point(p2), point(p)):

            pruned_path = pruned_path[0:-1]
            pruned_path.append(p)

        elif heuristic(p, p2) < 1e-6:
            pruned_path = pruned_path[0:-1]
            pruned_path.append(p)

        else:
            pruned_path.append(p)

        p2 = pruned_path[-1]
        p1 = pruned_path[-2]

    return pruned_path


class Poly:

    def __init__(self, coords, height):
        self._polygon = Polygon(coords)
        self._height = height

    @property
    def height(self):
        return self._height

    @property
    def coords(self):
        return list(self._polygon.exterior.coords)[:-1]

    @property
    def area(self):
        return self._polygon.area

    @property
    def center(self):
        return self._polygon.centroid.x, self._polygon.centroid.y

    def contains(self, point):
        point = Point(point)
        return self._polygon.contains(point)

    def crosses(self, other):
        return self._polygon.crosses(other)


def extract_polygons(data):
    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
        corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]),
                   (obstacle[1], obstacle[2])]

        # height of the polygon
        height = alt + d_alt

        p = Poly(corners, height)
        polygons.append(p)

    return polygons


def can_connect(p1, p2, polygons):
    """
    Tests to see if a straight line can connect the two given points without intersecting polygons in a grid.
    instance.
    :param p1: tuple(x1, y1, h1)
    :param p2: tuple(x2, y2, h2)
    :param polygons: list of shapely polygon objects.
    :return:
    """
    l = LineString([p1, p2])
    for p in polygons:
        if p.crosses(l) and p.height >= min(p1[2], p2[2]):
            return False
    return True


class Sampler:

    def __init__(self, data):
        """
            class to generate random 3d points that do not collide with obstacles at positions specified in the given
            data array.
            :param data: numpy array with 6 cols specifying the size of an obstacle.

        """
        # represent the obstacles as polygons.
        self._polygons = extract_polygons(data)
        self._xmin = np.min(data[:, 0] - data[:, 3])
        self._xmax = np.max(data[:, 0] + data[:, 3])

        self._ymin = np.min(data[:, 1] - data[:, 4])
        self._ymax = np.max(data[:, 1] + data[:, 4])
        # limit zmin to min flying altitude.
        self._zmin = 5
        # limit z-axis
        self._zmax = 20
        # Record maximum polygon dimension in the xy plane
        # multiply by 2 since given sizes are half widths
        # This is still rather clunky but will allow us to
        # cut down the number of polygons we compare with by a lot.

        # self._max_poly_xy sets the safety distance from obstacles.
        self._max_poly_xy = 2 * np.max((data[:, 3], data[:, 4]))
        centers = np.array([p.center for p in self._polygons])
        self._tree = KDTree(centers, metric='euclidean')

    def sample(self, num_samples):
        """Implemented with a k-d tree for efficiency."""
        xvals = np.random.uniform(self._xmin, self._xmax, num_samples)
        yvals = np.random.uniform(self._ymin, self._ymax, num_samples)
        zvals = np.random.uniform(self._zmin, self._zmax, num_samples)
        samples = list(zip(xvals, yvals, zvals))

        pts = []
        for s in samples:
            in_collision = False
            # obtain obstacle polygon centers that are within a radius of self._max_poly_xy from sample nodes.
            idxs = list(self._tree.query_radius(np.array([s[0], s[1]]).reshape(1, -1), r=self._max_poly_xy)[0])

            # If the sample node falls within the 2d space of the obstacle polygon we want to make sure that the node
            # is actually above the obstacle otherwise we discard the node to avoid collision.
            if len(idxs) > 0:
                for ind in idxs:
                    p = self._polygons[int(ind)]
                    if p.contains(s) and p.height >= s[2]:
                        in_collision = True
            if not in_collision:
                pts.append(s)

        return pts

    def polygons(self):
        return self._polygons

    @property
    def polygons(self):
        return self._polygons

