import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
from udacidrone.connection import MavlinkConnection
from planning_utils import a_star_grid, a_star_graph, heuristic, create_grid, create_graph, random_goal_search, \
    prune_path

from udacidrone import Drone
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")

        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        # self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values.
        filename = 'colliders.csv'
        # sub_data has the format 'lat0 37.792480, lon0 -122.397450\n'.
        sub_data = open(filename).readline()
        # re-format "sub_data" string to remove strings "lat0", "long0" and "\n".
        lat0, lon0 = [float(i) for i in sub_data.replace('\n', '').replace('lat0 ', '').replace(' lon0 ', '').split(',')]

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)
        print("\nglobal home position (lat, lon) : " + str((lat0, lon0)))

        # TODO: retrieve current global position
        current_global_position = self.global_position

        # TODO: convert to current local position using global_to_local().
        # Computing current local position relative to global home.
        local_north_position, local_east_position, local_down_position = global_to_local(current_global_position,
                                                                                         self.global_home)

        print("\nlocal (north, east, down) :" + str((local_north_position, local_east_position, local_down_position)))
        print("global home position: ", self.global_home)
        print("current global position :", self.global_position)
        print("current local position : ", self.local_position)

        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=3)

        # **************************************** A* grid search ******************************************************
        # For A* grid search uncomment the block of code below

        print("generating search grid...")
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("grid created.")
        # The north and east offsets are the north and east values by which the actual layout points have been
        # translated to fit on the grid.
        print("\nNorth offset = {0}, east offset = {1}".format(north_offset, east_offset))

        # TODO: convert start position to current position rather than map center
        # changing grid start position to become current local position.
        grid_north_pos = int(np.ceil(local_north_position - north_offset))
        grid_east_pos = int(np.ceil(local_east_position - east_offset))

        grid_start = (grid_north_pos, grid_east_pos)
        print("\ndrone mission start grid position:", grid_start)

        # Set goal as some arbitrary position on the grid
        # TODO: adapt to set goal as latitude / longitude position and convert
        global_goal_pos = random_goal_search(grid, self.global_home, north_offset, east_offset)
        local_goal_pos = global_to_local(global_goal_pos, self.global_home)
        grid_north_pos = int(np.ceil(local_goal_pos[0] - north_offset))
        # avoided taking np.ceil because global_to_local gives an overestimate by +1 of local_goal_pos[1] from
        # local_goal_pos[1] value computed by random_goal_search. Look up the def random_goal_search function in
        # planning_utils.py.
        grid_east_pos = int(local_goal_pos[1] - east_offset)

        grid_goal = (grid_north_pos, grid_east_pos)
        print("drone mission goal grid position:", grid_goal)

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)

        path, cost = a_star_grid(grid, heuristic, grid_start, grid_goal)
        print("\npath (length, cost):", (len(path), cost))

        # TODO: prune path to minimize number of waypoints
        path = prune_path(path)
        print("number of pruned waypoints : ", len(path))
        # Convert path to waypoints
        waypoints_ = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in path]

        # print("grid waypoints:", waypoints_)
        # **************************************************************************************************************
        print("\n")
        # TODO (if you're feeling ambitious): Try a different approach altogether!
        # ************************************* Probabilistic RoadMap **************************************************

        # For a Probabilistic RoadMap implementation uncomment the block of code below

        # Important Note:
        # The probabilistic road map is very computational intensive and takes more time than the Mavlink connection
        # will allow. A solution is to first compute waypoints then set these waypoints to the waypoints variable in
        # another run instance commenting the Probabilistic RoadMap block.

        # # TODO: convert start position to current position rather than map center
        # graph_start = (local_north_position, local_east_position, TARGET_ALTITUDE)
        # print("drone mission graph start : ", graph_start)
        # # # TODO: adapt to set goal as latitude / longitude position and convert
        # # # generates a random un-occupied global goal position on the grid.
        # #
        # print("generating search grid...")
        # # # Define a grid for a particular altitude and safety margin around obstacles
        # grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        # print("grid created.")
        # #
        # global_goal_pos = random_goal_search(grid, self.global_home, north_offset, east_offset)
        # local_goal_pos = global_to_local(global_goal_pos, self.global_home)
        # graph_goal = (local_goal_pos[0], local_goal_pos[1], 5)  # local_goal_pos[2])
        # print("drone mission graph goal :,", graph_goal)
        #
        # print("generating search graph...")
        # # The choice for number of edges between nodes k = 10 is arbitrary.
        # graph = create_graph(data, k=20, start_node=graph_start, goal_node=graph_goal, random_points=600)
        # print("search graph generated.")
        # print("number of nodes: ", len(graph.nodes))
        #
        # path, cost = a_star_graph(graph, heuristic, graph_start, graph_goal)
        # print("\ngraph path (length, cost):", (len(path), cost))
        #
        # # TODO: prune path to minimize number of waypoints
        # path = prune_path(path)
        # #
        # waypoints_ = [[int(p[0]), int(p[1]), int(p[2]), 0] for p in path]
        # print("graph waypoints:", waypoints_)
        #
        # **************************************************************************************************************

        # To align drone's heading with next waypoints
        p0 = waypoints_[0]
        waypoints = [p0]
        for p in waypoints_[1:]:
            heading_ = np.arctan2((p[1] - p0[1]), (p[0] - p0[0]))
            waypoints.append((p[0], p[1], p[2], heading_))
            p0 = p

        # Set self.waypoints
        self.waypoints = waypoints

        # TODO: send waypoints to sim
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        # *****************************
        # print("path:", self.plan_path())
        # *****************************

        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
