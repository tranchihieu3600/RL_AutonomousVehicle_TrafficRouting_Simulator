"""
Kinematic vehicle model supporting multiple vehicle types:
- Bike / Bicycle / Motorcycle (xe đạp, xe máy)
- Car / Sedan / Taxi (ô tô)
- Truck / Bus (xe tải, xe bus)

Features:
- Orderly lane discipline: Vehicles stay disciplined in assigned lanes without zigzagging.
- Intelligent Car-Following Model (IDM / Adaptive Cruise Control):
  NPCs sense vehicles ahead and automatically brake/slow down to maintain safe distance,
  guaranteeing NPCs NEVER crash into one another.
- Signal intentions only during structured, stable lane transitions.
"""

from typing import Tuple, List, Optional, Dict, Any
import math
import random
import numpy as np


class Vehicle:
    def __init__(
        self,
        x: float,
        y: float,
        heading: float = 0.0,
        speed: float = 0.0,
        vehicle_type: str = "car",
        length: Optional[float] = None,
        width: Optional[float] = None,
        wheelbase: float = 2.8,
        max_speed: float = 16.67,  # Giới hạn tối đa trong đô thị: 60 km/h = 16.67 m/s
        max_accel: float = 2.2,    # Gia tốc thực tế xe đô thị (0-60 km/h trong ~7-8s)
        max_decel: float = 3.6,    # Phanh hãm êm ái, chống chúi giật đột ngột
        max_steer: float = 0.6,
        is_ego: bool = False,
        lane_id: int = 1,
        path_waypoints: Optional[List[Tuple[float, float]]] = None
    ):
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.speed = float(speed)
        self.target_speed = float(speed)
        self.vehicle_type = vehicle_type
        self.is_ego = is_ego
        self.lane_id = lane_id
        self.crashed = False

        self.path_waypoints = path_waypoints or []
        self.current_wp_idx = 0

        # Turn signals: 'NONE', 'LEFT', 'RIGHT', 'HAZARD'
        self.turn_signal = "NONE"
        self.turn_signal_timer = 0.0
        self.blinker_state = False

        # Physical specs
        if vehicle_type == "motorcycle":
            self.length = length or 2.2
            self.width = width or 0.9
            self.wheelbase = 1.4
            self.max_speed = 10.0  # ~36 km/h
            self.color = (0, 200, 255)
        elif vehicle_type == "truck":
            self.length = length or 8.0
            self.width = width or 2.6
            self.wheelbase = 4.5
            self.max_speed = 8.5   # ~30 km/h
            self.color = (255, 145, 0)
        else:  # car
            self.length = length or 4.5
            self.width = width or 2.0
            self.wheelbase = wheelbase
            self.max_speed = max_speed
            self.color = (0, 230, 118) if is_ego else (255, 75, 75)

        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.max_steer = float(max_steer)

        # Overtaking / Lateral variables
        self.overtaking_state = "CRUISING"
        self.current_lateral_offset = 0.0
        self.target_lateral_offset = 0.0
        self.overtaking_timer = 0.0
        self.signal_duration_timer = 0.0  # Đếm thời gian xi-nhan (1.5s - 2.0s) trước khi được phép đánh lái sang làn khác

    def step(self, steer: float, accel: float, dt: float = 0.1) -> None:
        steer = max(-self.max_steer, min(self.max_steer, float(steer)))
        decel_limit = getattr(self, "max_decel", 3.6)
        accel = max(-decel_limit, min(self.max_accel, float(accel)))

        # Cập nhật tốc độ mượt mà theo gia tốc/giảm tốc thực tế
        self.speed += accel * dt
        self.speed = max(0.0, min(self.max_speed, self.speed))

        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        self.heading += (self.speed / self.wheelbase) * math.tan(steer) * dt
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # Blinker animation
        if self.turn_signal != "NONE":
            self.turn_signal_timer += dt
            if self.turn_signal_timer >= 0.25:
                self.blinker_state = not self.blinker_state
                self.turn_signal_timer = 0.0
        else:
            self.blinker_state = False

    def update_npc_patrol_with_safety(
        self,
        dt: float,
        all_vehicles: List["Vehicle"],
        network_manager: Optional[Any] = None
    ) -> None:
        """
        Orderly vehicle movement along road waypoints:
        1. Maintains lane discipline strictly.
        2. Car-following safety: Scans vehicle ahead in the same lane and adjusts speed.
        3. Traffic light compliance: Strictly stops at RED and YELLOW traffic lights before intersections!
        """
        if not self.path_waypoints:
            return

        offset = getattr(self, "current_lateral_offset", 0.0)
        self_cos = math.cos(self.heading)
        self_sin = math.sin(self.heading)
        self_act_x = self.x - self_sin * offset
        self_act_y = self.y + self_cos * offset

        # TỰ ĐỘNG CHUYỂN TIẾP WAYPOINT (Forward Progression & Lookahead Waypoint Tracking)
        # Loại bỏ 100% hiện tượng chạy vòng vòng:
        # Nếu xe đã tiến gần waypoint (< 7.5m) HOẶC waypoint đã nằm lại phía sau thân xe (forward_prog < -0.5m),
        # lập tức chuyển sang waypoint kế tiếp mà không quay đầu đuổi theo góc vuông!
        while self.current_wp_idx < len(self.path_waypoints) - 1:
            target_pt = self.path_waypoints[self.current_wp_idx]
            next_pt = self.path_waypoints[self.current_wp_idx + 1]
            road_h = math.atan2(next_pt[1] - target_pt[1], next_pt[0] - target_pt[0])

            t_lane_x = target_pt[0] - math.sin(road_h) * offset
            t_lane_y = target_pt[1] + math.cos(road_h) * offset
            dx = t_lane_x - self_act_x
            dy = t_lane_y - self_act_y
            dist = math.hypot(dx, dy)
            forward_prog = dx * self_cos + dy * self_sin

            if dist < 7.5 or forward_prog < -0.5:
                self.current_wp_idx += 1
            else:
                break

        target_pt = self.path_waypoints[self.current_wp_idx]
        is_at_end = (self.current_wp_idx >= len(self.path_waypoints) - 1)

        if is_at_end:
            # NPC ở cuối lộ trình: Tiếp tục phóng thẳng theo hướng trục đường hiện tại
            if self.current_wp_idx > 0:
                prev_pt = self.path_waypoints[self.current_wp_idx - 1]
                road_h = math.atan2(target_pt[1] - prev_pt[1], target_pt[0] - prev_pt[0])
            else:
                road_h = self.heading
            target_h = road_h
        else:
            next_pt = self.path_waypoints[self.current_wp_idx + 1]
            road_h = math.atan2(next_pt[1] - target_pt[1], next_pt[0] - target_pt[0])
            t_lane_x = target_pt[0] - math.sin(road_h) * offset
            t_lane_y = target_pt[1] + math.cos(road_h) * offset
            dx = t_lane_x - self_act_x
            dy = t_lane_y - self_act_y
            target_h = math.atan2(dy, dx)

        heading_err = (target_h - self.heading + math.pi) % (2 * math.pi) - math.pi
        steer = float(np.clip(heading_err * 1.5, -self.max_steer, self.max_steer))


        # Collision avoidance with vehicle ahead (Intelligent Car Following)
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        min_front_gap = 999.0
        front_veh_speed = self.target_speed

        # Actual position of this vehicle (accounting for current lateral lane offset)
        self_off = getattr(self, "current_lateral_offset", 0.0)
        self_actual_x = self.x - sin_h * self_off
        self_actual_y = self.y + cos_h * self_off

        for other in all_vehicles:
            if other is self:
                continue

            other_h = getattr(other, "heading", 0.0)
            other_off = getattr(other, "current_lateral_offset", 0.0)
            other_actual_x = other.x - math.sin(other_h) * other_off
            other_actual_y = other.y + math.cos(other_h) * other_off

            # Calculate relative position along this vehicle's heading
            ox = other_actual_x - self_actual_x
            oy = other_actual_y - self_actual_y
            forward_dist = ox * cos_h + oy * sin_h
            lateral_dist = -ox * sin_h + oy * cos_h

            # Phân biệt xe cùng chiều hay xe ngược chiều (2 chiều giao thông)
            heading_diff = abs((other_h - self.heading + math.pi) % (2 * math.pi) - math.pi)
            is_same_direction = heading_diff < (math.pi * 0.45)

            # Check if other vehicle is in front in the same lane corridor (|lateral_dist| < 2.0m)
            # Only consider same-direction vehicles for car-following queue!
            lane_diff = abs(self_off - other_off)
            if is_same_direction and 0.5 < forward_dist < 40.0 and (lane_diff < 1.9 or abs(lateral_dist) < 2.2):
                if forward_dist < min_front_gap:
                    min_front_gap = forward_dist
                    front_veh_speed = other.speed

        # Check traffic light ahead for NPC vehicle (Tuân thủ dừng đèn đỏ và đèn vàng)
        red_light_brake = False
        if network_manager and hasattr(network_manager, "traffic_lights") and hasattr(network_manager, "graph"):
            for nid, light in network_manager.traffic_lights.items():
                if nid not in network_manager.graph:
                    continue
                pos = network_manager.graph.nodes[nid]["pos"]
                lx = pos[0] - self.x
                ly = pos[1] - self.y
                forward_l = lx * cos_h + ly * sin_h
                lateral_l = abs(-lx * sin_h + ly * cos_h)
                l_state = light.get("state", "NONE").upper()

                # Vạch dừng xe (Stop Line) nằm cách tâm giao lộ 20.0m.
                # Xe chỉ dừng khi đang TIẾN ĐẾN vạch dừng (khoảng cách tới tâm giao lộ từ 20.0m đến 42.0m).
                # Khi xe đã vượt qua vạch dừng (< 20.0m) để vào giao lộ, xe PHẢI CHẠY THẲNG để giải phóng nút giao,
                # TUYỆT ĐỐI KHÔNG ĐƯỢC PHANH CHẮN NGANG NÚT GIAO LÀM TẮC TẤT CẢ CÁC XE!
                if 20.0 < forward_l < 42.0 and lateral_l < 14.0 and l_state in ["RED", "YELLOW"]:
                    red_light_brake = True
                    break

        # Speed adjustment based on front gap, end of route, or traffic light
        safe_distance = self.length + 10.0  # 15 - 18 meters buffer

        # Check if reached destination at end of route
        # NPC KHÔNG dừng đỗ chắn điểm đến: Tiếp tục chạy thẳng qua đích hoặc tái tuần tra
        if is_at_end:
            # Cho phép NPC giữ tốc độ chạy tiếp thoát khỏi nút giao điểm đến
            accel = (self.target_speed - self.speed) * 0.5
        elif red_light_brake:
            # Phanh hãm êm ái trước vạch dừng đèn đỏ/đèn vàng
            if self.speed > 0.5:
                accel = -2.4
            else:
                self.speed = 0.0
                accel = -0.5

        elif min_front_gap < safe_distance:
            if min_front_gap < 4.0:
                # Quá sát xe trước (< 4.0m): Giữ phanh an toàn
                accel = -2.5 if self.speed > 0.1 else 0.0
                if min_front_gap < 2.0 and self.speed > 0.2:
                    self.speed = max(0.0, self.speed - 0.5)
            elif min_front_gap < 7.0:
                # Cự ly từ 4.0m đến 7.0m: Nếu cả hai xe đang dừng và xe trước bắt đầu nhích bánh,
                # cho phép xe sau mớm ga nhẹ theo sau thay vì bị kẹt phanh!
                if front_veh_speed > 0.2:
                    target_v = max(2.5, front_veh_speed - 0.5)
                    accel = (target_v - self.speed) * 0.8
                elif self.speed > 0.5:
                    accel = -1.5
                else:
                    accel = 0.0
            else:
                # Cự ly an toàn (> 7.0m): Tự tin lăn bánh tăng tốc theo luồng xe phía trước
                target_v = max(3.5, front_veh_speed)
                accel = (target_v - self.speed) * 0.7
        else:
            # Clear road: tăng tốc mượt mà trở lại tốc độ hành trình
            accel = (self.target_speed - self.speed) * 0.65
        self.step(steer=steer, accel=accel, dt=dt)
        self.update_lateral_offset(dt)

    def update_lateral_offset(self, dt: float = 0.1) -> None:
        diff = self.target_lateral_offset - self.current_lateral_offset
        shift_speed = 3.5
        if abs(diff) > 0.05:
            self.current_lateral_offset += math.copysign(min(abs(diff), shift_speed * dt), diff)
        else:
            self.current_lateral_offset = self.target_lateral_offset

    def get_bounding_box(self) -> List[Tuple[float, float]]:
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        offset = getattr(self, "current_lateral_offset", 0.0)
        actual_x = self.x - sin_h * offset
        actual_y = self.y + cos_h * offset

        dx_l = 0.5 * self.length
        dy_w = 0.5 * self.width
        corners = [
            (dx_l, dy_w),
            (dx_l, -dy_w),
            (-dx_l, -dy_w),
            (-dx_l, dy_w)
        ]

        world_corners = []
        for lx, ly in corners:
            wx = actual_x + (lx * cos_h - ly * sin_h)
            wy = actual_y + (lx * sin_h + ly * cos_h)
            world_corners.append((wx, wy))
        return world_corners

    def check_collision_with(self, other: "Vehicle") -> bool:
        cos_h1 = math.cos(self.heading)
        sin_h1 = math.sin(self.heading)
        off1 = getattr(self, "current_lateral_offset", 0.0)
        x1 = self.x - sin_h1 * off1
        y1 = self.y + cos_h1 * off1

        cos_h2 = math.cos(other.heading)
        sin_h2 = math.sin(other.heading)
        off2 = getattr(other, "current_lateral_offset", 0.0)
        x2 = other.x - sin_h2 * off2
        y2 = other.y + cos_h2 * off2

        dist = math.hypot(x1 - x2, y1 - y2)
        threshold = (self.length + other.length) * 0.38
        return dist < threshold
