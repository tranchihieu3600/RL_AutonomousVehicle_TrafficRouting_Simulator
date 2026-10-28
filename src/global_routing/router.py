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

    def find_nearest_road_edge_point(
        self,
        click_x: float,
        click_y: float,
        max_dist: float = 35.0
    ) -> Optional[Dict[str, Any]]:
        """
        Projects an arbitrary 2D click point (x, y) onto the nearest road segment in the network.
        Identifies the exact directed carriageway edge (u -> v), longitudinal progress t in [0, 1],
        and lateral distance/side relative to the median strip.
        """
        best_candidate = None
        min_proj_dist = float("inf")

        for u, v, data in self.graph.edges(data=True):
            if u not in self.graph.nodes or v not in self.graph.nodes:
                continue
            pos_u = self.graph.nodes[u]["pos"]
            pos_v = self.graph.nodes[v]["pos"]

            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-3:
                continue

            ux = dx / seg_len
            uy = dy / seg_len

            # Vector from pos_u to click
            vx = click_x - pos_u[0]
            vy = click_y - pos_u[1]

            # Longitudinal projection
            t = (vx * ux + vy * uy) / seg_len
            # Clamp t slightly inside segment (not right at junction center)
            t_clamped = max(0.05, min(0.95, t))

            # Centerline point on road
            cx = pos_u[0] + ux * (t_clamped * seg_len)
            cy = pos_u[1] + uy * (t_clamped * seg_len)

            # Normal vector pointing rightwards relative to u -> v: (uy, -ux)
            nx_r = uy
            ny_r = -ux

            # Lateral signed offset from centerline along normal
            # For road u -> v, forward carriageway is at right offset ~ +10.0m:
            # (click_x - cx) * nx_r + (click_y - cy) * ny_r
            lat_offset = (click_x - cx) * nx_r + (click_y - cy) * ny_r

            # Distance from click to this edge's carriageway center (+10.0m)
            dist_to_carriageway = math.hypot(
                click_x - (cx + nx_r * 10.0),
                click_y - (cy + ny_r * 10.0)
            )

            # Only consider if click is on the forward side of the road (lat_offset > 0)
            # or if it's the closest within max_dist
            if lat_offset >= -2.0 and dist_to_carriageway < min_proj_dist and dist_to_carriageway <= max_dist:
                min_proj_dist = dist_to_carriageway
                road_heading = math.atan2(dy, dx)
                actual_pt = (cx + nx_r * 10.0, cy + ny_r * 10.0)

                u_name = self.graph.nodes[u].get("name", u)
                v_name = self.graph.nodes[v].get("name", v)
                road_label = f"{u_name} ➔ {v_name}"

                best_candidate = {
                    "u": u,
                    "v": v,
                    "t": t_clamped,
                    "centerline_pt": (cx, cy),
                    "proj_point": actual_pt,
                    "heading": road_heading,
                    "dist": dist_to_carriageway,
                    "road_label": road_label,
                    "edge_data": data
                }

        return best_candidate

    def _build_uturn_curve(
        self,
        junc_id: str,
        incoming_from: str,
        outgoing_to: str,
        radius: float = 24.0
    ) -> List[Tuple[float, float]]:
        """
        Builds a smooth U-turn maneuver looping through an intersection node.
        Enters from `incoming_from`, sweeps around the intersection island/center,
        and aligns with `outgoing_to` direction.
        """
        junc_pos = self.graph.nodes[junc_id]["pos"]
        prev_pos = self.graph.nodes[incoming_from]["pos"]
        next_pos = self.graph.nodes[outgoing_to]["pos"]

        v_in_x = junc_pos[0] - prev_pos[0]
        v_in_y = junc_pos[1] - prev_pos[1]
        l_in = max(1e-3, math.hypot(v_in_x, v_in_y))
        u_in_x, u_in_y = v_in_x / l_in, v_in_y / l_in

        # Tangent for U-turn loop (turning left across center): normal = (-u_in_y, u_in_x)
        p_stop = (junc_pos[0] - u_in_x * 12.0, junc_pos[1] - u_in_y * 12.0)
        p_apex1 = (junc_pos[0] - u_in_y * radius, junc_pos[1] + u_in_x * radius)
        p_apex2 = (junc_pos[0] + u_in_x * 8.0 - u_in_y * (radius * 0.7), junc_pos[1] + u_in_y * 8.0 + u_in_x * (radius * 0.7))
        p_exit = (junc_pos[0] - u_in_x * 15.0 - u_in_y * 10.0, junc_pos[1] - u_in_y * 15.0 + u_in_x * 10.0)

        # Smooth Bezier arc for U-turn
        loop_pts = []
        controls = [p_stop, p_apex1, p_apex2, p_exit]
        num_loop = 12
        for k in range(num_loop + 1):
            t = k / float(num_loop)
            # Cubic bezier through controls
            bx = (1 - t)**3 * controls[0][0] + 3 * (1 - t)**2 * t * controls[1][0] + 3 * (1 - t) * t**2 * controls[2][0] + t**3 * controls[3][0]
            by = (1 - t)**3 * controls[0][1] + 3 * (1 - t)**2 * t * controls[1][1] + 3 * (1 - t) * t**2 * controls[2][1] + t**3 * controls[3][1]
            loop_pts.append((bx, by))
        return loop_pts

    def build_point_to_point_waypoints(
        self,
        start_pt_info: Dict[str, Any],
        goal_pt_info: Dict[str, Any],
        base_path_nodes: List[str]
    ) -> List[Tuple[float, float]]:
        """
        Builds seamless continuous waypoints connecting exact along-road start and goal points:
        - Slices start road segment from start_t to 1.0 (approaching next intersection).
        - Executes junction maneuvers (turns, traffic light approaches, roundabouts, or U-turns).
        - Slices destination road segment from 0.0 to goal_t (stopping at destination).
        """
        s_u, s_v, s_t = start_pt_info["u"], start_pt_info["v"], start_pt_info["t"]
        g_u, g_v, g_t = goal_pt_info["u"], goal_pt_info["v"], goal_pt_info["t"]

        pos_su = self.graph.nodes[s_u]["pos"]
        pos_sv = self.graph.nodes[s_v]["pos"]
        pos_gu = self.graph.nodes[g_u]["pos"]
        pos_gv = self.graph.nodes[g_v]["pos"]

        # Case 1: Same edge and same direction with goal ahead
        if s_u == g_u and s_v == g_v and s_t < g_t:
            # Simple straight line forward along the edge
            return [
                (pos_su[0] + (pos_sv[0] - pos_su[0]) * s_t, pos_su[1] + (pos_sv[1] - pos_su[1]) * s_t),
                (pos_su[0] + (pos_sv[0] - pos_su[0]) * g_t, pos_su[1] + (pos_sv[1] - pos_su[1]) * g_t)
            ]

        # Case 2: Multi-edge path through graph
        # Base coarse waypoints from node graph
        coarse_wps = self.build_waypoints_with_roundabouts(base_path_nodes)
        if not coarse_wps:
            return [start_pt_info["proj_point"], goal_pt_info["proj_point"]]

        final_waypoints: List[Tuple[float, float]] = []

        # Start point along start edge:
        start_wp = (
            pos_su[0] + (pos_sv[0] - pos_su[0]) * s_t,
            pos_su[1] + (pos_sv[1] - pos_su[1]) * s_t
        )
        final_waypoints.append(start_wp)

        # Intermediate points along the rest of start edge (from s_t to 1.0)
        num_s_pts = max(1, int(round((1.0 - s_t) * 5.0)))
        for k in range(1, num_s_pts):
            t_k = s_t + (k / float(num_s_pts)) * (1.0 - s_t)
            final_waypoints.append((
                pos_su[0] + (pos_sv[0] - pos_su[0]) * t_k,
                pos_su[1] + (pos_sv[1] - pos_su[1]) * t_k
            ))

        # Check for U-turn at first node: if path is [s_u, s_v, s_u]
        is_direct_uturn = (len(base_path_nodes) >= 3 and base_path_nodes[0] == s_u and base_path_nodes[1] == s_v and base_path_nodes[2] == s_u)

        if is_direct_uturn:
            uturn_arc = self._build_uturn_curve(junc_id=s_v, incoming_from=s_u, outgoing_to=s_u)
            final_waypoints.extend(uturn_arc)
        else:
            # Append node waypoints starting after node s_v
            # Find closest matching index in coarse_wps near node s_v
            final_waypoints.extend(coarse_wps[1:-1])

        # Intermediate points along goal edge (from 0.0 to g_t)
        num_g_pts = max(1, int(round(g_t * 5.0)))
        for k in range(num_g_pts):
            t_k = (k / float(num_g_pts)) * g_t
            final_waypoints.append((
                pos_gu[0] + (pos_gv[0] - pos_gu[0]) * t_k,
                pos_gu[1] + (pos_gv[1] - pos_gu[1]) * t_k
            ))

        goal_wp = (
            pos_gu[0] + (pos_gv[0] - pos_gu[0]) * g_t,
            pos_gu[1] + (pos_gv[1] - pos_gu[1]) * g_t
        )
        final_waypoints.append(goal_wp)

        return final_waypoints

    def find_all_route_proposals_for_points(
        self,
        start_pt_info: Dict[str, Any],
        goal_pt_info: Dict[str, Any],
        via_node: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates route proposals between arbitrary along-road points.
        Handles U-turn routing when the destination is on the opposite carriageway
        of the same road or requires reversing direction at a signalized intersection or roundabout.
        """
        s_u, s_v = start_pt_info["u"], start_pt_info["v"]
        g_u, g_v = goal_pt_info["u"], goal_pt_info["v"]

        # Case A: Same road segment and same direction, destination is ahead
        if s_u == g_u and s_v == g_v and start_pt_info["t"] <= goal_pt_info["t"]:
            path_nodes = [s_u, s_v]
            wps = self.build_point_to_point_waypoints(start_pt_info, goal_pt_info, path_nodes)
            total_len = math.hypot(
                goal_pt_info["proj_point"][0] - start_pt_info["proj_point"][0],
                goal_pt_info["proj_point"][1] - start_pt_info["proj_point"][1]
            )
            return [{
                "name": "Đi Thẳng Cùng Làn Đến Đích",
                "tag": "TRỰC TIẾP",
                "color": "#00E676",
                "nodes": path_nodes,
                "total_cost": total_len / 13.89,
                "total_length": total_len,
                "estimated_seconds": max(5, int(total_len / 10.0)),
                "waypoints": wps,
                "instructions": [{
                    "from_node": s_u,
                    "to_node": s_v,
                    "to_name": goal_pt_info.get("road_label", "Điểm đến"),
                    "action": "Đi thẳng đến điểm đến bên phải",
                    "icon": "⬆️",
                    "distance_m": total_len,
                    "heading": start_pt_info["heading"]
                }]
            }]

        # Case B: Destination is on the opposite carriageway of the SAME road (s_u == g_v and s_v == g_u)
        # Or destination requires traveling through s_v, executing a U-turn / loop-around at s_v
        if s_u == g_v and s_v == g_u:
            # Vehicle MUST go forward along s_u -> s_v, reach junction s_v (traffic light/roundabout),
            # execute a U-turn at s_v into edge s_v -> s_u, and proceed to destination.
            # Base node path is [s_u, s_v, s_u]
            path_nodes = [s_u, s_v, s_u]
            wps = self.build_point_to_point_waypoints(start_pt_info, goal_pt_info, path_nodes)
            edge_fwd_len = self.graph[s_u][s_v].get("length", 100.0)
            total_len = edge_fwd_len * (1.0 - start_pt_info["t"]) + 35.0 + edge_fwd_len * goal_pt_info["t"]
            junc_name = self.graph.nodes[s_v].get("name", s_v)

            instructions = [
                {
                    "from_node": s_u,
                    "to_node": s_v,
                    "to_name": junc_name,
                    "action": "Tiến thẳng đến giao lộ có đèn tín hiệu",
                    "icon": "⬆️",
                    "distance_m": edge_fwd_len * (1.0 - start_pt_info["t"]),
                    "heading": start_pt_info["heading"]
                },
                {
                    "from_node": s_v,
                    "to_node": s_u,
                    "to_name": goal_pt_info.get("road_label", "Điểm đến bên làn đối diện"),
                    "action": f"Vòng quay đầu (U-Turn) tại {junc_name} sang làn ngược lại",
                    "icon": "↶",
                    "distance_m": edge_fwd_len * goal_pt_info["t"],
                    "heading": goal_pt_info["heading"]
                }
            ]

            return [{
                "name": f"Vòng Đầu Tại {junc_name} Sang Làn Đối Diện",
                "tag": "QUAY ĐẦU",
                "color": "#00E676",
                "nodes": path_nodes,
                "total_cost": total_len / 10.0,
                "total_length": total_len,
                "estimated_seconds": max(10, int(total_len / 9.0)),
                "waypoints": wps,
                "instructions": instructions
            }]

        # Case C: General multi-road routing: from intersection s_v to intersection g_u
        graph_start_node = s_v
        graph_goal_node = g_u

        # If s_v == g_u, vehicle just goes from s_u -> s_v -> g_v
        if graph_start_node == graph_goal_node:
            base_proposals = [{
                "nodes": [s_u, s_v, g_v],
                "name": "Lộ trình qua giao lộ",
                "tag": "NHANH NHẤT",
                "color": "#00E676"
            }]
        else:
            base_proposals = self.find_all_route_proposals(graph_start_node, graph_goal_node, via_node)

        packaged_proposals = []
        for prop in base_proposals:
            p_nodes = prop["nodes"]
            # Prepend s_u if not already there
            full_nodes = [s_u] + [n for n in p_nodes if n != s_u]
            # Append g_v if not already ending at g_v
            if not full_nodes or full_nodes[-1] != g_v:
                full_nodes.append(g_v)

            wps = self.build_point_to_point_waypoints(start_pt_info, goal_pt_info, full_nodes)
            instructions = self.generate_turn_instructions(full_nodes)

            total_len = prop.get("total_length", 100.0)
            packaged_proposals.append({
                "name": prop.get("name", "Lộ trình đề xuất"),
                "tag": prop.get("tag", "ĐỀ XUẤT"),
                "color": prop.get("color", "#00E676"),
                "nodes": full_nodes,
                "total_cost": prop.get("total_cost", 10.0),
                "total_length": total_len,
                "estimated_seconds": prop.get("estimated_seconds", 30),
                "waypoints": wps,
                "instructions": instructions
            })

        return packaged_proposals


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
        roundabout_radius: float = 34.0,
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
                    # 1. Roundabout arc (Vòng xuyến) lưu thông ngược chiều kim đồng hồ (Counter-Clockwise)
                    # Thiết kế đường cong chuyển tiếp Bezier mượt mà tại cả Cửa Vào (Entry) và Cửa Ra (Exit),
                    # triệt tiêu hoàn toàn góc gập 90 độ, giúp xe vào/ra vòng xuyến êm ái, bám chắc làn đường.
                    v_in_x = curr_pos[0] - prev_pos[0]
                    v_in_y = curr_pos[1] - prev_pos[1]
                    l_in = max(1e-3, math.hypot(v_in_x, v_in_y))
                    u_in_x, u_in_y = v_in_x / l_in, v_in_y / l_in

                    v_out_x = next_pos[0] - curr_pos[0]
                    v_out_y = next_pos[1] - curr_pos[1]
                    l_out = max(1e-3, math.hypot(v_out_x, v_out_y))
                    u_out_x, u_out_y = v_out_x / l_out, v_out_y / l_out

                    theta_in = math.atan2(-u_in_y, -u_in_x)
                    theta_out = math.atan2(u_out_y, u_out_x)
                    d_theta = (theta_out - theta_in) % (2.0 * math.pi)
                    if d_theta < 0.2:
                        d_theta = 2.0 * math.pi

                    l_blend = min(22.0, l_in * 0.35, l_out * 0.35)
                    blend_ang = min(0.35, d_theta * 0.2)

                    # 1.1 Đoạn uốn cong đón xe vào vòng xuyến (Entrance Bezier Fillet)
                    p_app = (curr_pos[0] - u_in_x * (roundabout_radius + l_blend), curr_pos[1] - u_in_y * (roundabout_radius + l_blend))
                    p_ctrl_in = (curr_pos[0] - u_in_x * roundabout_radius, curr_pos[1] - u_in_y * roundabout_radius)
                    ang_in = theta_in + blend_ang
                    p_circ_in = (curr_pos[0] + roundabout_radius * math.cos(ang_in), curr_pos[1] + roundabout_radius * math.sin(ang_in))

                    num_fillet_pts = 5
                    for k in range(num_fillet_pts):
                        t = k / float(num_fillet_pts - 1)
                        bx = (1 - t)**2 * p_app[0] + 2 * (1 - t) * t * p_ctrl_in[0] + t**2 * p_circ_in[0]
                        by = (1 - t)**2 * p_app[1] + 2 * (1 - t) * t * p_ctrl_in[1] + t**2 * p_circ_in[1]
                        waypoints.append((bx, by))

                    # 1.2 Cung tròn lòng đường vòng xuyến (Circular Arc)
                    ang_out = theta_out - blend_ang
                    rem_arc = (ang_out - ang_in) % (2.0 * math.pi)
                    num_arc = max(8, int(round(rem_arc * 8.0)))
                    for k in range(1, num_arc):
                        t = k / float(num_arc)
                        ang = ang_in + t * rem_arc
                        arc_x = curr_pos[0] + roundabout_radius * math.cos(ang)
                        arc_y = curr_pos[1] + roundabout_radius * math.sin(ang)
                        waypoints.append((arc_x, arc_y))

                    # 1.3 Đoạn uốn cong dẫn xe thoát khỏi vòng xuyến (Exit Bezier Fillet)
                    p_circ_out = (curr_pos[0] + roundabout_radius * math.cos(ang_out), curr_pos[1] + roundabout_radius * math.sin(ang_out))
                    p_ctrl_out = (curr_pos[0] + u_out_x * roundabout_radius, curr_pos[1] + u_out_y * roundabout_radius)
                    p_exit = (curr_pos[0] + u_out_x * (roundabout_radius + l_blend), curr_pos[1] + u_out_y * (roundabout_radius + l_blend))

                    for k in range(num_fillet_pts):
                        t = k / float(num_fillet_pts - 1)
                        bx = (1 - t)**2 * p_circ_out[0] + 2 * (1 - t) * t * p_ctrl_out[0] + t**2 * p_exit[0]
                        by = (1 - t)**2 * p_circ_out[1] + 2 * (1 - t) * t * p_ctrl_out[1] + t**2 * p_exit[1]
                        waypoints.append((bx, by))
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
