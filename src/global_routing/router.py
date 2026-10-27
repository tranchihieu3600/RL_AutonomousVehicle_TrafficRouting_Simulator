"""
Routing engine implementing multi-criteria pathfinding:
1. Route A (A* Smart Routing): Né kẹt xe thời gian thực (Traffic-Aware Fast ETA).
2. Route B (Dijkstra Shortest Distance): Đường ngắn nhất theo số mét di chuyển thuần túy.
3. Route C (High-speed Expressway Route): Tuyến ưu tiên đường cao tốc / đại lộ lớn (Bypass Boulevard).
4. Custom Intermediate Waypoint Routing (Tương tự Google Maps: Chọn điểm dừng trung gian).
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import networkx as nx
from src.global_routing.graph_manager import RoadNetworkManager


class GlobalRouter:
    """Computes multiple route proposals (A*, Dijkstra, Highway Bypass) with turn instructions."""

    def __init__(self, network_manager: RoadNetworkManager):
        self.net = network_manager
        self.graph = network_manager.graph

    def _euclidean_heuristic(self, u: str, goal: str) -> float:
        pos_u = self.graph.nodes[u]["pos"]
        pos_goal = self.graph.nodes[goal]["pos"]
        dist = math.hypot(pos_goal[0] - pos_u[0], pos_goal[1] - pos_u[1])
        max_speed = 22.22  # Highway speed ~80 km/h
        return dist / max_speed

    def generate_turn_instructions(self, path_nodes: List[str]) -> List[Dict[str, Any]]:
        if len(path_nodes) < 2:
            return []

        instructions = []
        for i in range(len(path_nodes) - 1):
            curr_id = path_nodes[i]
            next_id = path_nodes[i + 1]

            pos_curr = self.graph.nodes[curr_id]["pos"]
            pos_next = self.graph.nodes[next_id]["pos"]
            edge = self.graph[curr_id][next_id]
            dist_m = edge.get("length", math.hypot(pos_next[0] - pos_curr[0], pos_next[1] - pos_curr[1]))

            curr_heading = math.atan2(pos_next[1] - pos_curr[1], pos_next[0] - pos_curr[0])

            if i == 0:
                action_text = "Xuất phát"
                icon = "🚩"
            else:
                prev_id = path_nodes[i - 1]
                pos_prev = self.graph.nodes[prev_id]["pos"]
                prev_heading = math.atan2(pos_curr[1] - pos_prev[1], pos_curr[0] - pos_prev[0])
                turn_angle = (curr_heading - prev_heading + math.pi) % (2 * math.pi) - math.pi

                if abs(turn_angle) < 0.35:
                    action_text = "Đi thẳng"
                    icon = "⬆️"
                elif turn_angle > 0.35:
                    action_text = "Rẽ trái"
                    icon = "⬅️"
                else:
                    action_text = "Rẽ phải"
                    icon = "➡️"

            node_name = self.graph.nodes[next_id].get("name", next_id)
            instructions.append({
                "from_node": curr_id,
                "to_node": next_id,
                "to_name": node_name,
                "action": action_text,
                "icon": icon,
                "distance_m": dist_m,
                "heading": curr_heading
            })

        return instructions

    def build_waypoints_with_roundabouts(
        self,
        path_nodes: List[str],
        roundabout_radius: float = 38.0,
        corner_fillet_radius: float = 18.0
    ) -> List[Tuple[float, float]]:
        """
        Builds smooth continuous waypoints along the route:
        1. Roundabouts (is_roundabout=True): Replaces center node with a smooth counter-clockwise
           circular arc hugging the roundabout ring.
        2. Regular Intersections: Generates quadratic Bezier fillets to round out sharp corners (e.g. 90-degree turns),
           preventing vehicles from overshooting or getting stuck circling at intersections.
        """
        if not path_nodes:
            return []
        if len(path_nodes) <= 2:
            return [self.net.get_node_pos(nid) for nid in path_nodes]

        waypoints: List[Tuple[float, float]] = []
        for i in range(len(path_nodes)):
            curr_id = path_nodes[i]
            curr_pos = self.net.get_node_pos(curr_id)
            node_data = self.graph.nodes[curr_id]
            is_rb = node_data.get("is_roundabout", False)

            # Intermediate node processing
            if 0 < i < len(path_nodes) - 1:
                prev_pos = self.net.get_node_pos(path_nodes[i - 1])
                next_pos = self.net.get_node_pos(path_nodes[i + 1])

                if is_rb:
                    # 1. Roundabout arc (Vòng xuyến)
                    theta_in = math.atan2(prev_pos[1] - curr_pos[1], prev_pos[0] - curr_pos[0])
                    theta_out = math.atan2(next_pos[1] - curr_pos[1], next_pos[0] - curr_pos[0])

                    # Lưu thông ngược chiều kim đồng hồ (Counter-Clockwise)
                    d_theta = (theta_out - theta_in) % (2.0 * math.pi)
                    if d_theta < 0.25:
                        d_theta = 2.0 * math.pi

                    num_arc_pts = max(6, int(round(d_theta * 4.0)))
                    for k in range(num_arc_pts + 1):
                        frac = k / float(num_arc_pts)
                        ang = theta_in + frac * d_theta
                        arc_x = curr_pos[0] + roundabout_radius * math.cos(ang)
                        arc_y = curr_pos[1] + roundabout_radius * math.sin(ang)
                        waypoints.append((arc_x, arc_y))
                else:
                    # 2. Regular Intersection Corner Fillet (Uốn cong ngã rẽ góc nhọn/vuông)
                    v1_x = curr_pos[0] - prev_pos[0]
                    v1_y = curr_pos[1] - prev_pos[1]
                    len1 = max(1e-3, math.hypot(v1_x, v1_y))
                    u1_x, u1_y = v1_x / len1, v1_y / len1

                    v2_x = next_pos[0] - curr_pos[0]
                    v2_y = next_pos[1] - curr_pos[1]
                    len2 = max(1e-3, math.hypot(v2_x, v2_y))
                    u2_x, u2_y = v2_x / len2, v2_y / len2

                    # Calculate turn angle between incoming and outgoing directions
                    dot = max(-1.0, min(1.0, u1_x * u2_x + u1_y * u2_y))
                    # If road turns significantly (angle > 15 degrees, i.e., dot < 0.96)
                    if dot < 0.96:
                        r = min(corner_fillet_radius, len1 * 0.4, len2 * 0.4)
                        p_start = (curr_pos[0] - u1_x * r, curr_pos[1] - u1_y * r)
                        p_end = (curr_pos[0] + u2_x * r, curr_pos[1] + u2_y * r)

                        # Quadratic Bezier curve rounding the corner
                        num_fillet_pts = 5
                        for k in range(num_fillet_pts + 1):
                            t = k / float(num_fillet_pts)
                            bx = (1 - t)**2 * p_start[0] + 2 * (1 - t) * t * curr_pos[0] + t**2 * p_end[0]
                            by = (1 - t)**2 * p_start[1] + 2 * (1 - t) * t * curr_pos[1] + t**2 * p_end[1]
                            waypoints.append((bx, by))
                    else:
                        waypoints.append(curr_pos)
            else:
                waypoints.append(curr_pos)

        return waypoints

    def _package_path_result(
        self,
        path_nodes: List[str],
        name: str,
        tag: str,
        color_hex: str
    ) -> Dict[str, Any]:
        """Calculates distance, ETA, and turns for a given node sequence."""
        total_cost = 0.0
        total_length = 0.0
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            edge = self.graph[u][v]
            total_cost += edge.get("cost", 0.0)
            total_length += edge.get("length", 0.0)

        # Lộ trình uốn cong theo vòng xuyến (không đâm xuyên tâm đảo cỏ)
        waypoints = self.build_waypoints_with_roundabouts(path_nodes)
        instructions = self.generate_turn_instructions(path_nodes)

        # Estimate travel time based on road speeds and densities
        estimated_seconds = max(10, int(total_cost))

        return {
            "name": name,
            "tag": tag,
            "color": color_hex,
            "nodes": path_nodes,
            "total_cost": total_cost,
            "total_length": total_length,
            "estimated_seconds": estimated_seconds,
            "waypoints": waypoints,
            "instructions": instructions
        }

    def find_all_route_proposals(
        self,
        start_node: str,
        goal_node: str,
        via_node: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates multiple alternative routes similar to Google Maps:
        1. Tuyến 1: A* Dynamic Traffic Routing (Nhanh nhất - Né kẹt xe).
        2. Tuyến 2: Dijkstra Shortest Distance (Khoảng cách ngắn nhất).
        3. Tuyến 3: Highway / Expressway Priority (Ưu tiên đại lộ rộng).
        4. Tuyến 4: Nếu có điểm ghé qua (Via/Waystop), tính lộ trình Start -> Via -> Goal.
        """
        proposals: List[Dict[str, Any]] = []

        if start_node not in self.graph or goal_node not in self.graph:
            return proposals

        # 1. Proposal 1: A* Dynamic Cost (Nhanh nhất, né kẹt xe)
        try:
            nodes_astar = nx.astar_path(
                self.graph,
                source=start_node,
                target=goal_node,
                heuristic=self._euclidean_heuristic,
                weight="cost"
            )
            res_astar = self._package_path_result(
                nodes_astar,
                name="Thuật toán A*",
                tag="A*",
                color_hex="#00E676"  # Vibrant Emerald Green
            )
            proposals.append(res_astar)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # 2. Proposal 2: Dijkstra Shortest Distance (Khoảng cách ngắn nhất)
        try:
            nodes_dijkstra = nx.shortest_path(self.graph, source=start_node, target=goal_node, weight="length")
            # Only add if distinct path from A*
            res_dijkstra = self._package_path_result(
                nodes_dijkstra,
                name="Thuật toán Dijkstra",
                tag="DIJKSTRA",
                color_hex="#29B6F6"  # Light Blue
            )
            if not proposals or nodes_dijkstra != proposals[0]["nodes"]:
                proposals.append(res_dijkstra)
            else:
                res_dijkstra["name"] = "Thuật toán Dijkstra"
                proposals.append(res_dijkstra)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # 3. Proposal 3: Highway / Boulevard Bypass (Ưu tiên đường lớn)
        try:
            # Custom weight favoring highway lanes
            def highway_weight(u, v, d):
                base_len = d.get("length", 100.0)
                lanes = d.get("lanes", 1)
                is_prohibited = d.get("is_prohibited", False)
                if is_prohibited:
                    return 1e9
                return base_len / (lanes * 1.5)

            nodes_hwy = nx.shortest_path(self.graph, source=start_node, target=goal_node, weight=highway_weight)
            existing_nodes = [p["nodes"] for p in proposals]
            if nodes_hwy not in existing_nodes:
                res_hwy = self._package_path_result(
                    nodes_hwy,
                    name="Tuyến Đại Lộ / Cao Tốc (Đường rộng thoáng)",
                    tag="ĐƯỜNG LỚN",
                    color_hex="#AB47BC"  # Purple
                )
                proposals.append(res_hwy)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # 4. If user specified custom intermediate waypoint (Via stop)
        if via_node and via_node in self.graph and via_node not in [start_node, goal_node]:
            try:
                p1 = nx.astar_path(self.graph, start_node, via_node, weight="cost")
                p2 = nx.astar_path(self.graph, via_node, goal_node, weight="cost")
                combined_nodes = p1[:-1] + p2
                res_via = self._package_path_result(
                    combined_nodes,
                    name=f"Tuyến Tùy Chọn (Ghé qua: {self.graph.nodes[via_node].get('name', via_node)})",
                    tag="TÙY CHỌN",
                    color_hex="#FFA726"  # Orange
                )
                proposals.append(res_via)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        return proposals

    def find_path_astar(self, start_node: str, goal_node: str) -> Dict[str, Any]:
        """Default primary route using A* dynamic cost."""
        proposals = self.find_all_route_proposals(start_node, goal_node)
        return proposals[0] if proposals else {
            "nodes": [], "total_cost": float("inf"), "total_length": 0.0, "waypoints": [], "instructions": []
        }

    @staticmethod
    def interpolate_waypoints(waypoints: List[Tuple[float, float]], step_size: float = 6.0) -> List[Tuple[float, float]]:
        if len(waypoints) < 2:
            return waypoints

        dense_points: List[Tuple[float, float]] = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.hypot(dx, dy)

            num_steps = max(1, int(dist // step_size))
            for s in range(1, num_steps + 1):
                t = s / num_steps
                dense_points.append((p1[0] + dx * t, p1[1] + dy * t))

        return dense_points
