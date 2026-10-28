"""
Hybrid Controller: Integrates Multi-Proposal Global Routing with Local Driving Control.
Supports selecting from multiple alternative routes (A*, Dijkstra, Highway, Custom Waypoint).
"""

from typing import List, Tuple, Dict, Any, Optional
import math
from src.global_routing.graph_manager import RoadNetworkManager
from src.global_routing.router import GlobalRouter
from src.local_env.environment import LocalDrivingEnv


class HybridNavigationController:
    def __init__(
        self,
        network_manager: RoadNetworkManager,
        start_node: str,
        goal_node: str,
        via_node: Optional[str] = None,
        re_route_interval_steps: int = 40,
        congestion_threshold: float = 0.75
    ):
        self.net = network_manager
        self.router = GlobalRouter(network_manager)
        self.start_node = start_node
        self.goal_node = goal_node
        self.via_node = via_node
        self.re_route_interval_steps = re_route_interval_steps
        self.congestion_threshold = congestion_threshold

        self.route_proposals: List[Dict[str, Any]] = []
        self.active_proposal_index = 0
        self.start_point_info: Optional[Dict[str, Any]] = None
        self.goal_point_info: Optional[Dict[str, Any]] = None

        self.current_path_nodes: List[str] = []
        self.current_waypoints: List[Tuple[float, float]] = []
        self.current_instructions: List[Dict[str, Any]] = []
        self.total_route_length_m = 0.0
        self.step_counter = 0

        # Compute initial proposals
        self.compute_all_route_proposals(self.start_node, self.goal_node, self.via_node)

        # Initialize Local Driving Environment with road network awareness (traffic lights, intersections)
        self.env = LocalDrivingEnv(waypoints=self.current_waypoints, num_npcs=3, network_manager=self.net)

    def compute_all_route_proposals(
        self,
        start: str,
        goal: str,
        via: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Calculate multiple candidate routes."""
        self.start_node = start
        self.goal_node = goal
        self.via_node = via

        self.route_proposals = self.router.find_all_route_proposals(start, goal, via)
        if not self.route_proposals:
            self.route_proposals = [self.router.find_path_astar(start, goal)]

        # Default select the first proposal (A* Traffic-aware)
        self.select_route_proposal(0)
        return self.route_proposals

    def select_route_proposal(self, index: int) -> Dict[str, Any]:
        """User picks one of the candidate routes (A*, Dijkstra, Highway, or Custom)."""
        if not self.route_proposals:
            return {}

        self.active_proposal_index = max(0, min(len(self.route_proposals) - 1, index))
        selected = self.route_proposals[self.active_proposal_index]

        self.current_path_nodes = selected["nodes"]
        self.current_instructions = selected.get("instructions", [])
        self.total_route_length_m = selected.get("total_length", 0.0)
        self.current_waypoints = GlobalRouter.interpolate_waypoints(selected["waypoints"], step_size=7.0)

        if hasattr(self, "env"):
            self.env.set_waypoints(self.current_waypoints, path_nodes=self.current_path_nodes)
            self.env.reset()

        print(f"[ROUTE SELECTION] Da chon: {selected.get('name')} | Quang duong: {self.total_route_length_m:.1f}m")
        return selected

    def set_route(
        self,
        new_start_node: str,
        new_goal_node: str,
        new_via_node: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Updates trip origin and destination and generates all alternative routes."""
        self.start_point_info = None
        self.goal_point_info = None
        proposals = self.compute_all_route_proposals(new_start_node, new_goal_node, new_via_node)
        self.env.set_waypoints(self.current_waypoints)
        self.env.reset()
        return proposals

    def set_point_route(
        self,
        start_pt_info: Dict[str, Any],
        goal_pt_info: Dict[str, Any],
        via_node: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Sets route between arbitrary along-road points on either carriageway direction.
        Automatically handles U-turns at intersections/traffic lights when destination is on opposite lane.
        """
        self.start_point_info = start_pt_info
        self.goal_point_info = goal_pt_info
        self.start_node = start_pt_info.get("u", "")
        self.goal_node = goal_pt_info.get("v", "")
        self.via_node = via_node

        self.route_proposals = self.router.find_all_route_proposals_for_points(
            start_pt_info,
            goal_pt_info,
            via_node
        )
        if not self.route_proposals:
            # Fallback to standard node routing
            self.route_proposals = self.router.find_all_route_proposals(self.start_node, self.goal_node, via_node)

        self.select_route_proposal(0)
        return self.route_proposals

    def get_nearest_graph_node(self, x: float, y: float) -> str:
        best_node = None
        min_dist = float("inf")
        for node, data in self.net.graph.nodes(data=True):
            nx_pos, ny_pos = data["pos"]
            d = math.hypot(nx_pos - x, ny_pos - y)
            if d < min_dist:
                min_dist = d
                best_node = node
        return best_node

    def check_dynamic_reroute_needed(self) -> bool:
        if len(self.current_path_nodes) < 2:
            return False

        for i in range(len(self.current_path_nodes) - 1):
            u = self.current_path_nodes[i]
            v = self.current_path_nodes[i + 1]
            if self.net.graph.has_edge(u, v):
                density = self.net.graph[u][v].get("density", 0.0)
                if density >= self.congestion_threshold:
                    return True
        return False

    def update_cycle(self) -> Dict[str, Any]:
        self.step_counter += 1
        rerouted = False

        if self.step_counter % self.re_route_interval_steps == 0:
            if self.check_dynamic_reroute_needed():
                ego_x = self.env.ego_vehicle.x
                ego_y = self.env.ego_vehicle.y
                nearest_node = self.get_nearest_graph_node(ego_x, ego_y)

                if nearest_node and nearest_node != self.goal_node:
                    old_path = list(self.current_path_nodes)
                    self.compute_all_route_proposals(nearest_node, self.goal_node, self.via_node)
                    self.env.set_waypoints(self.current_waypoints)
                    rerouted = True
                    print(f"[RE-ROUTE] Phát hiện kẹt xe! Đã chuyển hướng từ {old_path} -> {self.current_path_nodes}")

    def setup_random_training_route(self, deterministic_seed: Optional[int] = None) -> Tuple[str, str]:
        """
        Picks a new Start and Destination pair from the complete set of all 506 valid
        combinatorial OD routes across the metropolitan city network.
        Guarantees a distinct route from the current trip to expose the agent to all road types.
        """
        import random
        import networkx as nx

        all_nodes = list(self.net.graph.nodes())
        current_pair = (self.start_node, self.goal_node)

        # Tạo danh sách các cặp (Start -> Goal) hợp lệ có đường đi
        valid_pairs = [
            (u, v) for u in all_nodes for v in all_nodes
            if u != v and (u, v) != current_pair and nx.has_path(self.net.graph, u, v)
        ]

        if valid_pairs:
            best_pair = random.choice(valid_pairs)
        else:
            best_pair = ("Port_Harbor", "Financial_Tower")

        start, goal = best_pair
        self.set_route(start, goal, None)
        if hasattr(self, "env"):
            self.env.reset(seed=deterministic_seed)
        return start, goal

