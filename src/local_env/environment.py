"""
Custom Gymnasium Environment for Local Autonomous Vehicle Navigation.
Enforces:
1. Strict Lateral Boundaries: Vehicles never exceed road edge limits (max offset clamped to road width).
2. Proper Passing Check: Overtaking into Lane 3 is ONLY permitted if Lane 3 is genuinely empty/clear!
   If Lane 3 already has another vehicle, Ego stays in Lane 2, brakes, and follows safely.
3. Realistic Network Patrol: NPCs navigate continuous open-ended road corridors without circling on themselves.
4. Solid Collision Avoidance: Vehicles maintain safe headway gap and never phase through one another.
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.local_env.vehicle import Vehicle
from src.local_env.rewards import RewardCalculator


class LocalDrivingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        waypoints: Optional[List[Tuple[float, float]]] = None,
        num_npcs: int = 10,
        radar_range: float = 45.0,
        dt: float = 0.1,
        max_steps: int = 800,
        network_manager: Optional[Any] = None,
        auto_sample_route: bool = False
    ):
        super().__init__()
        self.dt = dt
        self.radar_range = radar_range
        self.max_steps = max_steps
        self.num_npcs = num_npcs
        self.traffic_density_mode = "MEDIUM"  # "LOW" (ít xe), "MEDIUM" (trung bình), "HIGH" (nhiều xe)
        self.step_count = 0
        self.network_manager = network_manager
        self.auto_sample_route = auto_sample_route

        # Chuẩn bị danh sách tất cả các cặp điểm (Start -> Goal) có đường đi hợp lệ trong đô thị
        self.all_od_pairs: List[Tuple[str, str]] = []
        if self.network_manager and hasattr(self.network_manager, "graph"):
            import networkx as nx
            G = self.network_manager.graph
            nodes = list(G.nodes())
            self.all_od_pairs = [
                (u, v) for u in nodes for v in nodes 
                if u != v and nx.has_path(G, u, v)
            ]
        self.current_od_pair: Optional[Tuple[str, str]] = None

        self.waypoints = waypoints or [(50, 100), (50, 300), (50, 500)]
        self.current_wp_idx = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation space (17 dims):
        # [0]: norm_dist, [1]: norm_heading_err, [2]: norm_speed, [3..10]: radar (8 rays),
        # [11]: prev_steer, [12]: norm_front, [13]: norm_outer,
        # [14]: norm_tl_dist, [15]: tl_state_code (0=Green/None, 0.5=Yellow, 1.0=Red), [16]: rel_speed_to_front
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)

        self.reward_calculator = RewardCalculator()
        init_x, init_y = self.waypoints[0]
        init_heading = math.atan2(self.waypoints[1][1] - init_y, self.waypoints[1][0] - init_x) if len(self.waypoints) > 1 else 0.0
        self.ego_vehicle = Vehicle(
            x=init_x,
            y=init_y,
            heading=init_heading,
            speed=0.0,
            vehicle_type="car",
            max_speed=16.67,  # 60 km/h
            is_ego=True,
            lane_id=1
        )
        self.npcs: List[Vehicle] = []
        self.last_action = np.zeros(2, dtype=np.float32)

        # Chuẩn thiết kế đường 2 chiều (Dual-Carriageway with Central Median Divider):
        # Tim đường (Centerline, offset = 0.0m) là Dải Phân Cách Cứng (bê tông/bồn cây).
        # Chiều xe chạy bên phải (Right Carriageway) chiếm không gian từ -2.0m đến -11.0m:
        # Lane 0: Làn xe máy / lề phải (-9.5m)
        # Lane 1: Làn ô tô con / xe khách / xe tải ở giữa (-6.5m)
        # Lane 2: Làn vượt tốc độ cao sát dải phân cách (-3.5m)
        # Chiều xe ngược lại (Oncoming Carriageway) chiếm không gian đối xứng từ +2.0m đến +11.0m (+3.5m, +6.5m, +9.5m).
        # Dải phân cách giữa [-2.0m, +2.0m] (rộng 4.0m) đảm bảo khoảng cách an toàn giữa 2 chiều xe tối thiểu 7.0m,
        # xe tuyệt đối không đi trên dải phân cách và KHÔNG BAO GIỜ bị đấu đầu trực diện!
        self.LANE_OFFSETS = {0: -9.5, 1: -6.5, 2: -3.5}
        self.default_ego_lane = 1
        self.overtake_lane = 2

        self.stats = {
            "overtake_count": 0,
            "max_speed_kmh": 0.0,
            "speeds_recorded": [],
            "collision_reason": "",
            "step_count": 0,
            "red_light_violations": 0,
            "yellow_light_violations": 0
        }

    def set_waypoints(self, waypoints: List[Tuple[float, float]]) -> None:
        if len(waypoints) > 0:
            self.waypoints = waypoints
            self.current_wp_idx = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.current_wp_idx = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.stagnation_steps = 0
        self.stats["collision_reason"] = ""
        self.stats["speeds_recorded"] = []
        self.stats["max_speed_kmh"] = 0.0
        self.stats["red_light_violations"] = 0
        self.stats["yellow_light_violations"] = 0


        # Nếu kích hoạt chế độ auto_sample_route (dành cho huấn luyện RL):
        # Tự động lấy một cặp điểm ngẫu nhiên trong 506 tổ hợp (Start -> Goal) của đô thị
        if self.auto_sample_route and self.all_od_pairs and self.network_manager:
            import networkx as nx
            from src.global_routing.router import GlobalRouter
            od = random.choice(self.all_od_pairs)
            self.current_od_pair = od
            try:
                router = GlobalRouter(self.network_manager)
                path_nodes = nx.astar_path(self.network_manager.graph, od[0], od[1], weight="cost")
                coarse_wps = router.build_waypoints_with_roundabouts(path_nodes)
                interpolated = GlobalRouter.interpolate_waypoints(coarse_wps, step_size=7.0)
                if len(interpolated) >= 4:
                    self.waypoints = interpolated
            except Exception:
                pass

        start_x, start_y = self.waypoints[0]
        if len(self.waypoints) > 1:
            next_x, next_y = self.waypoints[1]
            init_heading = math.atan2(next_y - start_y, next_x - start_x)
        else:
            init_heading = 0.0

        self.ego_vehicle = Vehicle(
            x=start_x,
            y=start_y,
            heading=init_heading,
            speed=7.0,
            vehicle_type="car",
            max_speed=16.67,  # 60 km/h
            is_ego=True,
            lane_id=self.default_ego_lane
        )
        self.ego_vehicle.target_lateral_offset = self.LANE_OFFSETS[self.default_ego_lane]
        self.ego_vehicle.current_lateral_offset = self.LANE_OFFSETS[self.default_ego_lane]

        # Deterministic seed support for retry with same conditions
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Spawn NPCs in BOTH DIRECTIONS for realistic 2-way traffic simulation:
        # 1. Forward Traffic (Xe cùng chiều với Ego): Chạy theo chiều waypoints trên các làn cùng chiều (lane 0, 1, 2)
        # 2. Oncoming Traffic (Xe ngược chiều): Chạy ngược lại từ cuối tuyến về đầu tuyến ở phần đường bên kia vạch đôi vàng tim đường
        self.npcs = []
        # Thực tế giao thông Việt Nam / Đô thị: Nhiều xe máy, nhiều ô tô con, ít xe tải bự
        types_pool = [
            "motorcycle", "motorcycle", "car",
            "motorcycle", "car", "car",
            "truck", "motorcycle", "car"
        ]

        if self.traffic_density_mode == "LOW":
            target_forward = max(2, len(self.waypoints) // 16)
            target_oncoming = max(2, len(self.waypoints) // 16)
        elif self.traffic_density_mode == "HIGH":
            target_forward = max(7, len(self.waypoints) // 6)
            target_oncoming = max(7, len(self.waypoints) // 6)
        else:  # "MEDIUM"
            target_forward = max(4, len(self.waypoints) // 10)
            target_oncoming = max(4, len(self.waypoints) // 10)

        num_fwd = min(len(self.waypoints) - 3, target_forward)
        num_onc = min(len(self.waypoints) - 3, target_oncoming)

        # 1. SPAWN XE CÙNG CHIỀU (FORWARD TRAFFIC)
        step_stride_fwd = max(2, (len(self.waypoints) - 2) // max(1, num_fwd))
        lane_last_spawn_wp: Dict[int, int] = {}

        for i in range(num_fwd):
            wp_idx = 2 + i * step_stride_fwd
            if wp_idx >= len(self.waypoints) - 1:
                break

            wp = self.waypoints[wp_idx]
            next_wp = self.waypoints[wp_idx + 1] if wp_idx + 1 < len(self.waypoints) else wp
            npc_heading = math.atan2(next_wp[1] - wp[1], next_wp[0] - wp[0])

            v_type = types_pool[i % len(types_pool)]
            # Xe máy -> Làn 1 (lề phải -3.8m), Xe tải -> Làn 2 (giữa 0.0m), Ô tô con -> Làn 2 hoặc 3 (+3.8m)
            lane = 0 if v_type == "motorcycle" else (1 if v_type == "truck" else (2 if i % 2 == 1 else 1))

            if lane in lane_last_spawn_wp and abs(wp_idx - lane_last_spawn_wp[lane]) < 3:
                available_lanes = [l for l in [0, 1, 2] if l != lane]
                lane = available_lanes[i % len(available_lanes)]
            lane_last_spawn_wp[lane] = wp_idx

            speed = random.uniform(5.5, 7.5) if v_type != "car" else random.uniform(6.5, 8.5)
            npc_path = self.waypoints[wp_idx:]

            npc = Vehicle(
                x=wp[0],
                y=wp[1],
                heading=npc_heading,
                speed=speed,
                vehicle_type=v_type,
                lane_id=lane,
                path_waypoints=npc_path
            )
            npc.current_lateral_offset = self.LANE_OFFSETS[lane]
            npc.target_lateral_offset = self.LANE_OFFSETS[lane]
            self.npcs.append(npc)

        # 2. SPAWN XE NGƯỢC CHIỀU (ONCOMING TRAFFIC - PHÍA BÊN KIA TIM ĐƯỜNG VÀNG)
        # Waypoint ngược chiều: đi từ cuối tuyến về đầu tuyến
        rev_waypoints = list(reversed(self.waypoints))
        step_stride_onc = max(2, (len(rev_waypoints) - 2) // max(1, num_onc))
        lane_last_spawn_onc: Dict[int, int] = {}

        for j in range(num_onc):
            wp_idx = 2 + j * step_stride_onc
            if wp_idx >= len(rev_waypoints) - 1:
                break

            wp = rev_waypoints[wp_idx]
            next_wp = rev_waypoints[wp_idx + 1] if wp_idx + 1 < len(rev_waypoints) else wp
            onc_heading = math.atan2(next_wp[1] - wp[1], next_wp[0] - wp[0])

            v_type = types_pool[(j + 3) % len(types_pool)]
            lane = 0 if v_type == "motorcycle" else (1 if v_type == "truck" else (2 if j % 2 == 1 else 1))

            if lane in lane_last_spawn_onc and abs(wp_idx - lane_last_spawn_onc[lane]) < 3:
                available_lanes = [l for l in [0, 1, 2] if l != lane]
                lane = available_lanes[j % len(available_lanes)]
            lane_last_spawn_onc[lane] = wp_idx

            speed = random.uniform(5.5, 7.5) if v_type != "car" else random.uniform(6.5, 8.5)
            onc_path = rev_waypoints[wp_idx:]

            onc_npc = Vehicle(
                x=wp[0],
                y=wp[1],
                heading=onc_heading,
                speed=speed,
                vehicle_type=v_type,
                lane_id=lane,
                path_waypoints=onc_path
            )
            # Xe ngược chiều đi trên phần đường bên phải của hướng nó chạy:
            # Theo quy tắc giao thông bên phải (Right-Hand Traffic):
            # Nhìn theo hướng onc_heading (ngược với Ego), phần đường của xe ngược chiều
            # nằm ở phía bên phải tim đường vàng theo hướng xe ngược chiều,
            # tương ứng với offset chuẩn self.LANE_OFFSETS[lane]!
            onc_npc.current_lateral_offset = self.LANE_OFFSETS[lane]
            onc_npc.target_lateral_offset = self.LANE_OFFSETS[lane]
            self.npcs.append(onc_npc)

        obs = self._get_obs()
        return obs, {}

    def _get_radar_readings(self) -> np.ndarray:
        num_rays = 8
        readings = np.ones(num_rays, dtype=np.float32)
        angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)

        for i, rel_angle in enumerate(angles):
            ray_angle = self.ego_vehicle.heading + rel_angle
            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            min_dist = self.radar_range
            for npc in self.npcs:
                ox = npc.x - self.ego_vehicle.x
                oy = npc.y - self.ego_vehicle.y
                proj = ox * ray_dx + oy * ray_dy
                if 0 < proj < min_dist:
                    perp = abs(ox * (-ray_dy) + oy * ray_dx)
                    if perp < 2.5:
                        min_dist = min(min_dist, proj)

            readings[i] = min_dist / self.radar_range
        return readings

    def check_traffic_ahead_and_overtake_lane(self) -> Tuple[float, float, Optional[Vehicle]]:
        """
        Accurately inspects:
        1. Front vehicle in ego's current lane.
        2. Clearance of alternate lane (Lane 3 if in Lane 2; Lane 2 if in Lane 3).
        Returns: (min_front_dist, passing_lane_min_dist, front_car)
        """
        ego = self.ego_vehicle
        cos_h = math.cos(ego.heading)
        sin_h = math.sin(ego.heading)

        # Toạ độ thực tế của xe Ego trên mặt đường (tính cả độ lệch làn)
        ego_off = getattr(ego, "current_lateral_offset", 0.0)
        ego_actual_x = ego.x - sin_h * ego_off
        ego_actual_y = ego.y + cos_h * ego_off

        # Xác định làn mục tiêu: nếu đang ở làn 3 -> mục tiêu là làn 2; ngược lại mục tiêu là làn 3
        is_in_lane_3 = abs(ego_off - self.LANE_OFFSETS[self.overtake_lane]) < 1.0
        target_alt_lane = self.default_ego_lane if is_in_lane_3 else self.overtake_lane
        target_alt_offset = self.LANE_OFFSETS[target_alt_lane]

        min_front_dist = 999.0
        front_car = None
        passing_lane_min_dist = 999.0

        for npc in self.npcs:
            npc_h = getattr(npc, "heading", 0.0)
            npc_off = getattr(npc, "current_lateral_offset", 0.0)
            npc_actual_x = npc.x - math.sin(npc_h) * npc_off
            npc_actual_y = npc.y + math.cos(npc_h) * npc_off

            dx = npc_actual_x - ego_actual_x
            dy = npc_actual_y - ego_actual_y
            forward_dist = dx * cos_h + dy * sin_h
            lateral_dist = -dx * sin_h + dy * cos_h

            # Phân biệt xe cùng chiều hay xe ngược chiều (2 chiều giao thông)
            heading_diff = abs((npc_h - ego.heading + math.pi) % (2 * math.pi) - math.pi)
            is_same_direction = heading_diff < (math.pi * 0.55)

            # Phân định rõ ràng khoảng cách của xe trước cùng làn:
            # Trên đường cong và vòng xuyến, lateral_dist trong hệ trục xe có thể mở rộng tới 3.5m
            lane_diff = abs(ego_off - npc_off)
            is_same_lane = (lane_diff < 2.0)
            euclid_d = math.hypot(dx, dy)
            if is_same_direction and is_same_lane and 0.5 < forward_dist < 45.0 and abs(lateral_dist) < 3.2:
                if forward_dist < min_front_dist:
                    min_front_dist = forward_dist
                    front_car = npc
            # Bổ sung cự ly an toàn 360 độ cự ly gần (< 9.0m) phía trước xe
            elif is_same_direction and forward_dist > 0.0 and euclid_d < 9.0 and abs(lateral_dist) < 3.5:
                if euclid_d < min_front_dist:
                    min_front_dist = euclid_d
                    front_car = npc

            # Check làn còn lại (Làn 2 hoặc Làn 3):
            # Khoảng an toàn: kiểm tra cả xe phía trước lẫn xe đang chạy phía sau ở làn đó (cùng chiều)
            diff_to_target_lane = abs(npc_off - target_alt_offset)
            if is_same_direction and diff_to_target_lane < 2.0:
                # Quét khoảng trống an toàn ở làn bên cạnh:
                if -18.0 < forward_dist < 35.0:
                    passing_lane_min_dist = min(passing_lane_min_dist, abs(forward_dist))

        return min_front_dist, passing_lane_min_dist, front_car


    def check_upcoming_traffic_light(self) -> Tuple[float, str, float]:
        """
        Scans for upcoming intersection with traffic light in front of ego vehicle:
        Returns:
            (distance_to_light, light_state, remaining_timer)
            where light_state in ["NONE", "GREEN", "YELLOW", "RED"]
        """
        if not self.network_manager or not hasattr(self.network_manager, "traffic_lights"):
            return 999.0, "NONE", 0.0

        ego = self.ego_vehicle
        cos_h = math.cos(ego.heading)
        sin_h = math.sin(ego.heading)

        min_dist = 999.0
        best_state = "NONE"
        best_timer = 0.0

        for nid, light in self.network_manager.traffic_lights.items():
            if nid not in self.network_manager.graph:
                continue
            pos = self.network_manager.graph.nodes[nid]["pos"]
            dx = pos[0] - ego.x
            dy = pos[1] - ego.y
            forward_dist_to_center = dx * cos_h + dy * sin_h
            lateral_dist = abs(-dx * sin_h + dy * cos_h)

            # Vạch dừng (Stop Line) nằm cách tâm giao lộ 20.0m trên nhánh đường đang tiến vào.
            # Khoảng cách tới vạch dừng: dist_to_stop_line = forward_dist_to_center - 20.0m
            # Chỉ phát hiện đèn tín hiệu khi xe đang TIẾN VÀO ngã tư (forward_dist_to_center > 18.0m và < 65.0m)
            if 18.0 < forward_dist_to_center < 65.0 and lateral_dist < 14.0:
                dist_to_stop = forward_dist_to_center - 20.0
                if dist_to_stop < min_dist:
                    min_dist = dist_to_stop
                    best_state = light.get("state", "NONE").upper()
                    best_timer = light.get("timer", 0.0)

        return min_dist, best_state, best_timer

    def _get_obs(self) -> np.ndarray:
        curr_wp = self.waypoints[min(self.current_wp_idx, len(self.waypoints) - 1)]
        idx = min(self.current_wp_idx, len(self.waypoints) - 1)
        if idx < len(self.waypoints) - 1:
            nxt = self.waypoints[idx + 1]
            road_h = math.atan2(nxt[1] - curr_wp[1], nxt[0] - curr_wp[0])
        elif idx > 0:
            prv = self.waypoints[idx - 1]
            road_h = math.atan2(curr_wp[1] - prv[1], curr_wp[0] - prv[0])
        else:
            road_h = self.ego_vehicle.heading

        ego_cos = math.cos(self.ego_vehicle.heading)
        ego_sin = math.sin(self.ego_vehicle.heading)
        ego_off = getattr(self.ego_vehicle, "current_lateral_offset", 0.0)
        ego_act_x = self.ego_vehicle.x - ego_sin * ego_off
        ego_act_y = self.ego_vehicle.y + ego_cos * ego_off

        target_x = curr_wp[0] - math.sin(road_h) * ego_off
        target_y = curr_wp[1] + math.cos(road_h) * ego_off
        dx = target_x - ego_act_x
        dy = target_y - ego_act_y
        dist = math.hypot(dx, dy)
        norm_dist = min(1.0, dist / 50.0)

        target_heading = math.atan2(dy, dx)
        heading_err = (target_heading - self.ego_vehicle.heading + math.pi) % (2 * math.pi) - math.pi
        norm_heading_err = heading_err / math.pi
        norm_speed = self.ego_vehicle.speed / self.ego_vehicle.max_speed

        radar = self._get_radar_readings()
        prev_steer = self.last_action[0]

        front_dist, outer_clear, front_car = self.check_traffic_ahead_and_overtake_lane()
        norm_front = min(1.0, front_dist / 40.0)
        norm_outer = min(1.0, outer_clear / 30.0)

        # Method 2: Traffic light state and stop-line distance
        tl_dist, tl_state, _ = self.check_upcoming_traffic_light()
        norm_tl_dist = min(1.0, tl_dist / 40.0)
        if tl_state == "RED":
            tl_code = 1.0
        elif tl_state == "YELLOW":
            tl_code = 0.5
        else:
            tl_code = 0.0

        # Method 2: Relative velocity to front car (ego.speed - front_car.speed)
        front_speed = front_car.speed if front_car is not None else self.ego_vehicle.speed
        rel_speed = (self.ego_vehicle.speed - front_speed) / self.ego_vehicle.max_speed

        obs = np.array([
            norm_dist,
            norm_heading_err,
            norm_speed,
            *radar,
            prev_steer,
            norm_front,
            norm_outer,
            norm_tl_dist,
            tl_code,
            rel_speed
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)


    def execute_overtaking_behavior(self) -> None:
        ego = self.ego_vehicle
        front_dist, passing_clear, front_npc = self.check_traffic_ahead_and_overtake_lane()

        # 1. State: CRUISING (Xe đang ở làn giữa hoặc làn 3)
        if ego.overtaking_state == "CRUISING":
            ego.turn_signal = "NONE"
            ego.signal_duration_timer = 0.0

            # Kiểm tra đèn tín hiệu: Tuyệt đối CẤM VƯỢT khi đang tiến gần giao lộ có đèn đỏ/đèn vàng!
            tl_dist, tl_state, _ = self.check_upcoming_traffic_light()
            is_approaching_stop_light = (tl_state in ["RED", "YELLOW"] and tl_dist < 35.0)

            # Xác định làn hiện tại:
            is_in_lane_3 = abs(ego.current_lateral_offset - self.LANE_OFFSETS[self.overtake_lane]) < 1.0

            # ĐIỀU KIỆN VƯỢT XE:
            # Chỉ vượt khi:
            # 1. Cự ly tới xe trước an toàn (8.0m <= front_dist < 20.0m) - KHÔNG ĐƯỢC vượt khi đã dí quá sát (< 8.0m)
            # 2. Làn mục tiêu (làn 3 hoặc làn 2) cực kỳ thông thoáng (passing_clear > 25.0m)
            # 3. Không đang tiến gần giao lộ đèn đỏ/vàng
            if not is_approaching_stop_light and 8.0 <= front_dist < 20.0 and passing_clear > 25.0:
                if not is_in_lane_3:
                    ego.overtaking_state = "SIGNALING_OUTER"
                    ego.turn_signal = "LEFT"
                    ego.signal_duration_timer = 0.0
                    print(f"[XI-NHAN] Lan 3 thong thoang ({passing_clear:.1f}m) -> Bat xi-nhan trai chuan bi vuot sang lan 3...")
                else:
                    ego.overtaking_state = "SIGNALING_INNER"
                    ego.turn_signal = "RIGHT"
                    ego.signal_duration_timer = 0.0
                    print(f"[XI-NHAN] Lan 2 thong thoang ({passing_clear:.1f}m) -> Bat xi-nhan phai chuan bi vuot ve lan 2...")
            elif front_dist < 8.0:
                # Quá sát xe phía trước: TUYỆT ĐỐI KHÔNG ĐƯỢC ĐÁNH LÁI TẠC ĐẦU! Giữ thẳng lái và phanh an toàn!
                pass

        # 2. State: SIGNALING_OUTER (Bật xi-nhan trái sang làn 3)
        elif ego.overtaking_state == "SIGNALING_OUTER":
            ego.turn_signal = "LEFT"
            ego.signal_duration_timer += self.dt

            # Nếu khoảng cách tới xe làn 2 quá gần (< 7.0m) hoặc làn 3 có xe áp sát (< 20.0m) -> HỦY VƯỢT!
            if passing_clear < 20.0 or front_dist < 7.0:
                ego.overtaking_state = "CRUISING"
                ego.turn_signal = "NONE"
                print(f"[HUY LENH VUOT] Khoang cach qua gan (xe truoc: {front_dist:.1f}m, lan 3: {passing_clear:.1f}m) -> Huy vuot de bao toan!")
            elif ego.signal_duration_timer >= 1.5 and passing_clear > 25.0 and front_dist >= 7.5:
                ego.overtaking_state = "CHANGING_OUTER"
                ego.target_lateral_offset = self.LANE_OFFSETS[self.overtake_lane]
                self.stats["overtake_count"] += 1
                print("[OVERTAKE] Chuyen sang lan 3 de vuot!")

        # 2b. State: SIGNALING_INNER (Bật xi-nhan phải từ làn 3 về lại làn 2 để vượt)
        elif ego.overtaking_state == "SIGNALING_INNER":
            ego.turn_signal = "RIGHT"
            ego.signal_duration_timer += self.dt

            if passing_clear < 20.0 or front_dist < 7.0:
                ego.overtaking_state = "CRUISING"
                ego.turn_signal = "NONE"
                print(f"[HUY LENH VUOT] Khoang cach qua gan (xe truoc: {front_dist:.1f}m, lan 2: {passing_clear:.1f}m) -> Huy vuot de bao toan!")
            elif ego.signal_duration_timer >= 1.5 and passing_clear > 25.0 and front_dist >= 7.5:
                ego.overtaking_state = "CHANGING_INNER"
                ego.target_lateral_offset = self.LANE_OFFSETS[self.default_ego_lane]
                self.stats["overtake_count"] += 1
                print("[OVERTAKE] Lan 2 thong thoang -> Chuyen ve lan 2 de vuot qua xe o lan 3!")

        # 3. State: CHANGING_OUTER (Dạt sang làn 3)
        elif ego.overtaking_state == "CHANGING_OUTER":
            ego.turn_signal = "LEFT"
            if abs(ego.current_lateral_offset - self.LANE_OFFSETS[self.overtake_lane]) < 0.15:
                # Đã sang hẳn làn 3: Chuyển về CRUISING để xe chạy thẳng bình thường trên làn 3!
                # Không bắt buộc phải về lại làn 2, xe có thể tiếp tục hành trình tốc độ cao trên làn 3!
                ego.overtaking_state = "CRUISING"
                ego.turn_signal = "NONE"
                ego.overtaking_timer = 0.0
                print("[OVERTAKE] Da sang lan 3 thanh cong -> Tiep tuc hanh trinh tren lan 3!")

        # 3b. State: CHANGING_INNER (Dạt về làn 2)
        elif ego.overtaking_state == "CHANGING_INNER":
            ego.turn_signal = "RIGHT"
            if abs(ego.current_lateral_offset - self.LANE_OFFSETS[self.default_ego_lane]) < 0.15:
                ego.overtaking_state = "CRUISING"
                ego.turn_signal = "NONE"
                print("[OVERTAKE] Da ve lai lan 2 thanh cong va tiep tuc hanh trinh.")

        # 4. State: PASSING (Duy trì hành trình trên làn 3, chỉ chuyển về làn 2 nếu làn 3 bị cản và làn 2 thoáng)
        elif ego.overtaking_state == "PASSING":
            # Nếu xe đang ở làn 3 và đường làn 3 hoàn toàn thông thoáng: Chuyển thẳng về trạng thái CRUISING trên làn 3
            if abs(ego.current_lateral_offset - self.LANE_OFFSETS[self.overtake_lane]) < 0.2:
                ego.overtaking_state = "CRUISING"
                ego.turn_signal = "NONE"
            else:
                ego.overtaking_timer += self.dt

        # 5. State: SIGNALING_RETURN
        elif ego.overtaking_state == "SIGNALING_RETURN":
            ego.turn_signal = "RIGHT"
            ego.signal_duration_timer += self.dt
            if ego.signal_duration_timer >= 1.2:
                ego.overtaking_state = "CHANGING_INNER"
                ego.target_lateral_offset = self.LANE_OFFSETS[self.default_ego_lane]

        # 6. State: RETURNING
        elif ego.overtaking_state == "RETURNING":
            ego.turn_signal = "RIGHT"
            if abs(ego.current_lateral_offset - self.LANE_OFFSETS[self.default_ego_lane]) < 0.15:
                ego.overtaking_state = "CRUISING"
                ego.turn_signal = "NONE"
                print("[OVERTAKE] Da ve lai lan giua an toan.")


        # CẬP NHẬT VÀ KHỐNG CHẾ ĐỘ LỆCH LÀN TRONG LÒNG ĐƯỜNG BÊN PHẢI [-11.0m, -2.5m]
        # Tuyệt đối không lấn vào dải phân cách giữa [-2.0m, +2.0m] hoặc vượt ra ngoài vỉa hè (-11.0m)
        ego.update_lateral_offset(self.dt)
        ego.current_lateral_offset = max(-11.0, min(-2.5, ego.current_lateral_offset))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        self.execute_overtaking_behavior()

        curr_wp = self.waypoints[min(self.current_wp_idx, len(self.waypoints) - 1)]
        prev_dist = math.hypot(curr_wp[0] - self.ego_vehicle.x, curr_wp[1] - self.ego_vehicle.y)

        # =========================================================================
        # METHOD 1: XÂY DỰNG "LỚP BỌC AN TOÀN" (RULE-BASED SAFETY CAGE / ACTION OVERRIDE)
        # =========================================================================
        front_dist, passing_clear, front_car = self.check_traffic_ahead_and_overtake_lane()
        tl_dist, tl_state, _ = self.check_upcoming_traffic_light()
        ego = self.ego_vehicle
        front_speed = front_car.speed if front_car is not None else 0.0
        rel_speed = ego.speed - front_speed

        # Tọa độ thực tế hiện tại của Ego vehicle trong thế giới (sau khi dạt theo làn)
        ego_cos = math.cos(ego.heading)
        ego_sin = math.sin(ego.heading)
        ego_off = getattr(ego, "current_lateral_offset", 0.0)
        ego_act_x = ego.x - ego_sin * ego_off
        ego_act_y = ego.y + ego_cos * ego_off

        # 1.0 TỰ ĐỘNG CHUYỂN WAYPOINT VÀ ĐIỀU KHIỂN GÓC LÁI BÁM LÀN (Pure Pursuit Lane Tracking)
        # Loại bỏ triệt để hiện tượng xoay vòng vòng tại ngã rẽ:
        # Nếu xe đã tới gần waypoint (< 6.0m) HOẶC waypoint đã trôi lại phía sau thân xe (forward_prog < -0.5m),
        # lập tức chuyển sang waypoint kế tiếp để luôn tiến về phía trước!
        while self.current_wp_idx < len(self.waypoints) - 1:
            wp_pt = self.waypoints[self.current_wp_idx]
            d_wp = math.hypot(wp_pt[0] - ego.x, wp_pt[1] - ego.y)
            dx_fwd = wp_pt[0] - ego.x
            dy_fwd = wp_pt[1] - ego.y
            fwd_prog = dx_fwd * ego_cos + dy_fwd * ego_sin
            if d_wp < 6.0 or fwd_prog < -0.5:
                self.current_wp_idx += 1
            else:
                break

        curr_wp = self.waypoints[min(self.current_wp_idx, len(self.waypoints) - 1)]
        idx = min(self.current_wp_idx, len(self.waypoints) - 1)
        if idx < len(self.waypoints) - 1:
            nxt = self.waypoints[idx + 1]
            road_h = math.atan2(nxt[1] - curr_wp[1], nxt[0] - curr_wp[0])
        elif idx > 0:
            prv = self.waypoints[idx - 1]
            road_h = math.atan2(curr_wp[1] - prv[1], curr_wp[0] - prv[0])
        else:
            road_h = ego.heading

        # Tim làn chính xác theo hướng tim đường:
        target_x = curr_wp[0] - math.sin(road_h) * ego_off
        target_y = curr_wp[1] + math.cos(road_h) * ego_off
        dx = target_x - ego_act_x
        dy = target_y - ego_act_y
        target_h = math.atan2(dy, dx)
        heading_err = (target_h - ego.heading + math.pi) % (2 * math.pi) - math.pi

        # Góc lái chuẩn theo quỹ đạo làn đường (Ideal Pure-Pursuit Steering)
        ideal_steer = float(np.clip(heading_err * 1.6, -1.0, 1.0))

        # Phân chia nhiệm vụ: Lộ trình và tim làn cố định đường đi (85% - 95%), AI chỉ tinh chỉnh vi sai (5% - 15%)
        # Điều này đảm bảo 100% xe KHÔNG BAO GIỜ bị xoay vòng vòng mà luôn bám chắc làn đường!
        ai_steer = float(action[0])
        steer_action = 0.85 * ideal_steer + 0.15 * ai_steer

        steer_rad = float(np.clip(steer_action, -1.0, 1.0) * ego.max_steer)
        boost = 0.15 if ego.overtaking_state in ["CHANGING_OUTER", "PASSING"] else 0.0
        accel_val = float((action[1] + boost) * ego.max_accel)

        # 1.1 Khóa an toàn chống va chạm xe phía trước & xe bên cạnh khi chuyển làn (360 Safety Shield)
        # Khoảng cách an toàn động theo vận tốc: d_safe = 9.0m (cơ sở) + 1.2 * v (cự ly phản xạ phanh)
        safe_headway = 10.0 + ego.speed * 1.2
        if front_dist < 6.0:
            # Nguy cơ va chạm quá gần: Phanh hãm dứt khoát
            accel_val = -ego.max_decel
            if front_dist < 3.5 and ego.speed > front_speed:
                ego.speed = max(0.0, min(ego.speed, front_speed))
        elif front_dist < safe_headway:
            # Trong vùng cự ly an toàn: chủ động giảm tốc mượt mà theo xe phía trước
            if ego.speed > front_speed:
                accel_val = min(accel_val, -ego.max_decel * 0.6)
            elif front_dist < 12.0 and front_speed < 1.0:
                accel_val = min(accel_val, -ego.max_decel * 0.7)
        elif front_dist < 26.0 and front_speed < 2.0 and ego.speed > 5.0:
            # Xe trước đang dừng từ xa (đèn đỏ hoặc kẹt xe): hãm phanh mượt mà từ sớm
            accel_val = min(accel_val, -ego.max_decel * 0.4)

        # 1.1b Khóa an toàn chống va quệt bên hông khi xe ĐANG TRONG QUÁ TRÌNH ĐỔI LÀN (Side-swipe Shield)
        if ego.overtaking_state in ["CHANGING_OUTER", "CHANGING_INNER"]:
            cos_ego_h = math.cos(ego.heading)
            sin_ego_h = math.sin(ego.heading)
            ego_cur_off = getattr(ego, "current_lateral_offset", 0.0)
            ego_act_x = ego.x - sin_ego_h * ego_cur_off
            ego_act_y = ego.y + cos_ego_h * ego_cur_off
            for npc in self.npcs:
                npc_h = getattr(npc, "heading", 0.0)
                h_diff = abs((npc_h - ego.heading + math.pi) % (2 * math.pi) - math.pi)
                if h_diff >= (math.pi * 0.45):
                    continue  # Bỏ qua xe chạy ngược chiều ở phần đường bên kia vạch đôi vàng

                npc_off = getattr(npc, "current_lateral_offset", 0.0)
                npc_act_x = npc.x - math.sin(npc_h) * npc_off
                npc_act_y = npc.y + math.cos(npc_h) * npc_off
                d_center = math.hypot(npc_act_x - ego_act_x, npc_act_y - ego_act_y)
                if d_center < 3.2:
                    accel_val = min(accel_val, -ego.max_decel * 0.8)
                    if d_center < 2.5 and ego.speed > npc.speed:
                        ego.speed = max(0.0, min(ego.speed, npc.speed))
                    break

        # 1.2 Khóa an toàn giảm tốc từ xa khi gặp đèn đỏ & đèn vàng (Progressive Traffic Light Deceleration)
        # Bắt đầu rà phanh êm từ xa (28m -> 14m -> 6m) thay vì đợi sát nút mới phanh gấp
        if tl_state in ["RED", "YELLOW"] and 2.0 < tl_dist < 28.0:
            if tl_dist < 6.0:
                # Đã đến vạch dừng: Dừng hẳn xe êm ái
                accel_val = -ego.max_decel
                if tl_dist < 3.5 and ego.speed > 0.1:
                    ego.speed = max(0.0, ego.speed - 0.5)
            elif tl_dist < 14.0 and ego.speed > 3.0:
                accel_val = -ego.max_decel * 0.65
            elif tl_dist < 24.0 and ego.speed > 5.0:
                accel_val = -ego.max_decel * 0.4
            elif ego.speed > 8.0:
                accel_val = -ego.max_decel * 0.25

        self.ego_vehicle.step(steer=steer_rad, accel=accel_val, dt=self.dt)

        spd_kmh = self.ego_vehicle.speed * 3.6
        self.stats["speeds_recorded"].append(spd_kmh)
        self.stats["max_speed_kmh"] = max(self.stats["max_speed_kmh"], spd_kmh)
        self.stats["step_count"] = self.step_count

        # Step NPCs with safe car-following distance and traffic light respect
        all_vehicles = self.npcs + [self.ego_vehicle]
        for npc in self.npcs:
            npc.update_npc_patrol_with_safety(
                dt=self.dt,
                all_vehicles=all_vehicles,
                network_manager=self.network_manager
            )

        # Collision detection with accurate diagnosis
        crashed = False
        collision_reason = ""
        for npc in self.npcs:
            if self.ego_vehicle.check_collision_with(npc):
                crashed = True
                dx = npc.x - self.ego_vehicle.x
                dy = npc.y - self.ego_vehicle.y
                forward_dist = dx * math.cos(self.ego_vehicle.heading) + dy * math.sin(self.ego_vehicle.heading)
                lateral_dist = -dx * math.sin(self.ego_vehicle.heading) + dy * math.cos(self.ego_vehicle.heading)

                npc_h = getattr(npc, "heading", 0.0)
                h_diff = abs((npc_h - self.ego_vehicle.heading + math.pi) % (2 * math.pi) - math.pi)
                is_oncoming = h_diff > (math.pi * 0.55)

                if is_oncoming:
                    collision_reason = f"Đấu đầu trực diện với xe {npc.vehicle_type} chạy chiều ngược lại do lấn sang làn đối diện!"
                elif forward_dist > 1.5:
                    collision_reason = f"Đâm đuôi xe {npc.vehicle_type} phía trước do không giảm tốc kịp thời!"
                elif abs(lateral_dist) > 1.0 and npc.turn_signal != "NONE":
                    collision_reason = f"Va quệt bên hông với xe {npc.vehicle_type} đang đổi làn/rẽ!"
                else:
                    collision_reason = f"Va chạm cự ly gần với xe {npc.vehicle_type} tại giao lộ!"
                break

        if crashed:
            self.stats["collision_reason"] = collision_reason

        curr_dist = math.hypot(curr_wp[0] - self.ego_vehicle.x, curr_wp[1] - self.ego_vehicle.y)
        reached_wp = False
        if curr_dist < 4.5:
            reached_wp = True
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1

        dx = curr_wp[0] - self.ego_vehicle.x
        dy = curr_wp[1] - self.ego_vehicle.y
        target_heading = math.atan2(dy, dx)
        heading_err = (target_heading - self.ego_vehicle.heading + math.pi) % (2 * math.pi) - math.pi

        # Traffic light violation check (Red & Yellow light detection)
        traffic_violation = "NONE"
        tl_dist_now, tl_state_now, _ = self.check_upcoming_traffic_light()
        if tl_dist_now < 7.0 and self.ego_vehicle.speed > 2.0:
            if tl_state_now == "RED":
                traffic_violation = "RED"
                self.stats["red_light_violations"] += 1
                print(f"🚨 [VI PHẠM GIAO THÔNG] Xe đã VƯỢT ĐÈN ĐỎ tại giao lộ! (Tốc độ: {spd_kmh:.1f} km/h - Phạt nặng)")
            elif tl_state_now == "YELLOW":
                traffic_violation = "YELLOW"
                self.stats["yellow_light_violations"] += 1
                print(f"⚠️ [CẢNH BÁO VI PHẠM] Xe đã VƯỢT ĐÈN VÀNG không giảm tốc dừng an toàn! (Tốc độ: {spd_kmh:.1f} km/h - Phạt)")

        delta_action = float(np.linalg.norm(action - self.last_action))
        self.last_action = action.copy()

        # Method 3: Reward calculation with TTC, stop bonus, and severe violation penalties
        reward = self.reward_calculator.compute_reward(
            prev_dist_to_wp=prev_dist,
            curr_dist_to_wp=curr_dist,
            current_speed=self.ego_vehicle.speed,
            heading_error=heading_err,
            is_crashed=crashed,
            is_reached_wp=reached_wp,
            delta_action=delta_action,
            traffic_light_violation=traffic_violation,
            front_gap=front_dist,
            rel_speed=rel_speed,
            tl_dist=tl_dist,
            tl_state=tl_state,
            stagnation_steps=self.stagnation_steps
        )

        all_waypoints_cleared = (self.current_wp_idx >= len(self.waypoints) - 1 and curr_dist < 4.5)

        # Kiểm tra hiện tượng xe đứng yên (Stagnation):
        # NGUYÊN TẮC: Chỉ phạt lười di chuyển nếu đường hoàn toàn THÔNG THOÁNG mà xe KHÔNG CHỊU CHẠY.
        # Nếu xe đang dừng vì:
        # 1. Có xe phía trước (ô tô, xe tải to, xe máy) đang dừng hoặc chạy rất chậm (< 2.0 m/s) trong cự ly an toàn (front_dist < 22.0m)
        # 2. Hoặc đang tiếp cận ngã tư có đèn đỏ / đèn vàng (tl_dist < 32.0m)
        # -> ĐÂY LÀ HÀNH VI DỪNG XE HỢP LỆ THEO LUẬT GIAO THÔNG, TUYỆT ĐỐI KHÔNG ĐƯỢC PHẠT!
        is_stopped = self.ego_vehicle.speed < 0.4
        has_obstacle_ahead = (front_dist < 22.0 and front_speed < 2.0)
        has_red_light_ahead = (tl_state in ["RED", "YELLOW"] and 1.0 < tl_dist < 32.0)
        is_blocked = has_obstacle_ahead or has_red_light_ahead
        is_stagnated = False

        if is_stopped and not is_blocked:
            self.stagnation_steps += 1
            if self.stagnation_steps >= 50:  # Đúng 5.0 giây đứng im khi đường hoàn toàn thông thoáng
                is_stagnated = True
                collision_reason = "Xe dung yen qua lau (> 5s) khi duong thong thoang (Bi phat do luoi di chuyen)!"
                self.stats["collision_reason"] = collision_reason
                reward -= 100.0  # Phạt dứt khoát kết thúc episode
                print(f"[PHAT DUNG YEN] {collision_reason}")
        else:
            self.stagnation_steps = 0

        # Kiểm tra hiện tượng chệch khỏi đường lộ trình (Off-road Deviation):
        is_off_road = (curr_dist > 25.0)
        if is_off_road and not collision_reason:
            collision_reason = "Xe đi chệch quá xa khỏi hành lang lộ trình (Lao ra khỏi đường)!"
            self.stats["collision_reason"] = collision_reason
            reward -= 150.0
            print("[OFF-ROAD] Xe lech qua xa khoi hanh lang lo trinh!")



        # Method 3: Early Termination - Kết thúc ngay lập tức nếu vượt đèn đỏ, va chạm, đứng yên hoặc lao ra khỏi đường!
        is_red_light_fail = (traffic_violation == "RED")
        if is_red_light_fail and not collision_reason:
            collision_reason = "Vượt đèn đỏ trái phép tại giao lộ!"
            self.stats["collision_reason"] = collision_reason

        terminated = crashed or all_waypoints_cleared or is_red_light_fail or is_stagnated or is_off_road
        truncated = self.step_count >= self.max_steps

        info = {
            "is_crashed": crashed,
            "is_red_light_fail": is_red_light_fail,
            "is_stagnated": is_stagnated,
            "is_off_road": is_off_road,
            "collision_reason": collision_reason,
            "reached_destination": all_waypoints_cleared and not crashed and not is_red_light_fail and not is_stagnated and not is_off_road,
            "current_wp_idx": self.current_wp_idx,
            "speed": self.ego_vehicle.speed,
            "overtake_state": self.ego_vehicle.overtaking_state,
            "stats": self.stats
        }



        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

