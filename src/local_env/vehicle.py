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
            target_h = self.heading
            steer = 0.0
            is_at_end = False
        else:
            offset = getattr(self, "current_lateral_offset", 0.0)

            # Determine road heading of current segment
            idx = min(self.current_wp_idx, len(self.path_waypoints) - 1)
            if idx < len(self.path_waypoints) - 1:
                p_c = self.path_waypoints[idx]
                p_n = self.path_waypoints[idx + 1]
                raw_road_h = math.atan2(p_n[1] - p_c[1], p_n[0] - p_c[0])
            elif idx > 0:
                p_c = self.path_waypoints[idx]
                p_p = self.path_waypoints[idx - 1]
                raw_road_h = math.atan2(p_c[1] - p_p[1], p_c[0] - p_p[0])
            else:
                raw_road_h = self.heading

            # Smooth road heading transition (Rate-limited low pass filter)
            # Triệt tiêu bước nhảy góc gián đoạn giữa các đoạn thẳng khi chuyển waypoint
            prev_rh = getattr(self, "current_road_heading", raw_road_h)
            diff_rh = (raw_road_h - prev_rh + math.pi) % (2 * math.pi) - math.pi
            max_rh_rate = 0.45 * dt  # tối đa ~0.045 rad / 0.1s
            if abs(diff_rh) > max_rh_rate:
                road_h = prev_rh + math.copysign(max_rh_rate, diff_rh)
            else:
                road_h = raw_road_h
            road_h = (road_h + math.pi) % (2 * math.pi) - math.pi
            self.current_road_heading = road_h

            # Stable actual position based on road heading (eliminates lever-arm jump when steering)
            self_act_x = self.x - math.sin(road_h) * offset
            self_act_y = self.y + math.cos(road_h) * offset

            # TỰ ĐỘNG CHUYỂN TIẾP WAYPOINT MƯỢT MÀ (Windowed Progress-based Waypoint Advancement)
            # Quét cửa sổ waypoint phía trước: nếu xe đã vượt qua waypoint thì tự động chuyển tiếp,
            # triệt tiêu hoàn toàn lỗi kẹt lại phía sau gây quay đầu hoặc xoay vòng tròn!
            window_end = min(self.current_wp_idx + 4, len(self.path_waypoints) - 1)
            for k in range(self.current_wp_idx, window_end):
                cur_pt = self.path_waypoints[k]
                nxt_pt = self.path_waypoints[k + 1]
                seg_h = math.atan2(nxt_pt[1] - cur_pt[1], nxt_pt[0] - cur_pt[0])

                t_lane_x = cur_pt[0] - math.sin(seg_h) * offset
                t_lane_y = cur_pt[1] + math.cos(seg_h) * offset
                d_to_lane_wp = math.hypot(t_lane_x - self_act_x, t_lane_y - self_act_y)

                dx_p = self_act_x - t_lane_x
                dy_p = self_act_y - t_lane_y
                fwd_progress = dx_p * math.cos(seg_h) + dy_p * math.sin(seg_h)

                if d_to_lane_wp < 7.0 or fwd_progress > 0.5:
                    self.current_wp_idx = max(self.current_wp_idx, k + 1)
                else:
                    break

            # Lookahead Pure Pursuit Steering (Bám đường mượt mà, triệt tiêu góc ngoặt đột ngột)
            # Tăng lookahead khi chạy nhanh hoặc khi góc cua rộng để bám mượt mà
            lookahead_dist = max(9.0, min(22.0, 7.0 + self.speed * 1.6))
            target_pt = self.path_waypoints[self.current_wp_idx]
            target_seg_h = road_h

            accum_d = 0.0
            for k in range(self.current_wp_idx, len(self.path_waypoints) - 1):
                pt_a = self.path_waypoints[k]
                pt_b = self.path_waypoints[k + 1]
                d_seg = math.hypot(pt_b[0] - pt_a[0], pt_b[1] - pt_a[1])
                accum_d += d_seg
                if accum_d >= lookahead_dist:
                    target_pt = pt_b
                    target_seg_h = math.atan2(pt_b[1] - pt_a[1], pt_b[0] - pt_a[0])
                    break
            else:
                target_pt = self.path_waypoints[-1]
                if len(self.path_waypoints) >= 2:
                    target_seg_h = math.atan2(
                        self.path_waypoints[-1][1] - self.path_waypoints[-2][1],
                        self.path_waypoints[-1][0] - self.path_waypoints[-2][0]
                    )

            is_at_end = (self.current_wp_idx >= len(self.path_waypoints) - 1)
            if is_at_end:
                target_h = target_seg_h
            else:
                t_lane_x = target_pt[0] - math.sin(target_seg_h) * offset
                t_lane_y = target_pt[1] + math.cos(target_seg_h) * offset
                dx = t_lane_x - self_act_x
                dy = t_lane_y - self_act_y
                target_h = math.atan2(dy, dx)

            # Strict Forward Discipline (TUYỆT ĐỐI CẤM QUAY ĐẦU / CẤM XOAY VÒNG):
            # Xe lưu thông 1 chiều trên đường đô thị chỉ lái theo chiều tiến của tim đường
            heading_err = (target_h - self.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(heading_err) > math.pi * 0.45:
                # Nếu mục tiêu bị lệch góc lớn (> 80 độ) do cua gấp: bám theo tiếp tuyến đường tiến
                target_h = road_h
                heading_err = (target_h - self.heading + math.pi) % (2 * math.pi) - math.pi

            # Đánh lái mượt mà tự nhiên: giảm độ gắt steering gain từ 1.5 xuống 1.05
            raw_steer = float(np.clip(heading_err * 1.05, -self.max_steer, self.max_steer))
            prev_st = getattr(self, "current_steer", 0.0)
            max_st_rate = 1.2 * dt  # Giới hạn tốc độ vặn vô lăng (không giật gắt)
            steer = prev_st + np.clip(raw_steer - prev_st, -max_st_rate, max_st_rate)
            self.current_steer = float(steer)

            # Tính toán độ cong góc cua phía trước để tự động giảm tốc êm ái khi vào cua/vòng xuyến:
            curve_look_idx = min(self.current_wp_idx + 4, len(self.path_waypoints) - 1)
            if curve_look_idx > self.current_wp_idx:
                p_ahead_cur = self.path_waypoints[curve_look_idx - 1]
                p_ahead_nxt = self.path_waypoints[curve_look_idx]
                ahead_h = math.atan2(p_ahead_nxt[1] - p_ahead_cur[1], p_ahead_nxt[0] - p_ahead_cur[0])
                ahead_curve_deg = abs((ahead_h - road_h + math.pi) % (2 * math.pi) - math.pi)
            else:
                ahead_curve_deg = 0.0

            # Vận tốc an toàn khi vào cua (Cornering Speed Reduction):
            # Đường thẳng -> max 10.0 - 14.0 m/s; Cua vòng xuyến/giao lộ -> tự động hãm xuống 4.0 - 6.0 m/s
            if ahead_curve_deg > 0.25:
                self.curve_safe_speed = max(3.8, min(self.target_speed, 12.0 / (1.0 + ahead_curve_deg * 2.8)))
            else:
                self.curve_safe_speed = self.target_speed


        # Collision avoidance with vehicle ahead (Intelligent Car Following & Queueing)
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        # Actual position of this vehicle (accounting for current lateral lane offset and road heading)
        self_off = getattr(self, "current_lateral_offset", 0.0)
        self_rh = getattr(self, "current_road_heading", self.heading)
        self_actual_x = self.x - math.sin(self_rh) * self_off
        self_actual_y = self.y + math.cos(self_rh) * self_off

        min_bumper_gap = 999.0
        front_veh_speed = self.target_speed
        lead_vehicle = None
        overlapping_vehicle = None

        for other in all_vehicles:
            if other is self:
                continue

            other_rh = getattr(other, "current_road_heading", getattr(other, "heading", 0.0))
            other_off = getattr(other, "current_lateral_offset", 0.0)
            other_actual_x = other.x - math.sin(other_rh) * other_off
            other_actual_y = other.y + math.cos(other_rh) * other_off

            # Calculate relative position along this vehicle's heading
            ox = other_actual_x - self_actual_x
            oy = other_actual_y - self_actual_y
            forward_dist = ox * cos_h + oy * sin_h
            lateral_dist = -ox * sin_h + oy * cos_h

            other_h = getattr(other, "heading", 0.0)
            heading_diff = abs((other_h - self.heading + math.pi) % (2 * math.pi) - math.pi)
            is_same_direction = (heading_diff < math.pi * 0.35)

            half_l_sum = 0.5 * (self.length + other.length)
            half_w_sum = 0.5 * (self.width + other.width)
            bumper_gap = forward_dist - half_l_sum

            # Kiểm tra va quệt / chồng lấn vật lý đa hướng (Physical Overlap Check)
            dist_centers = math.hypot(ox, oy)
            if dist_centers < (half_l_sum * 0.9) and abs(lateral_dist) < (half_w_sum * 0.9):
                overlapping_vehicle = other

            # Xác định xe phía trước đang chắn hướng di chuyển:
            # 1. Xe khác ở phía trước (forward_dist > 0.1m)
            # 2. Cùng chiều lưu thông
            # 3. Nằm trong hành lang cản trở: khoảng cách ngang nhỏ hơn bề rộng 2 xe + 0.8m đệm
            #    hoặc cùng làn đường và khoảng cách ngang < 2.5m
            lane_diff = abs(self_off - other_off)
            is_same_lane = (lane_diff < 2.5)
            is_in_corridor = (is_same_lane and abs(lateral_dist) < 3.2) or (abs(lateral_dist) < (half_w_sum + 0.4))

            if is_same_direction and forward_dist > 0.1 and is_in_corridor:
                if bumper_gap < min_bumper_gap:
                    min_bumper_gap = bumper_gap
                    front_veh_speed = other.speed
                    lead_vehicle = other

        # Check traffic light ahead for NPC vehicle (Tuân thủ dừng đèn đỏ và đèn vàng)
        red_light_active = False
        dist_to_stop = 999.0
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

                # Vạch dừng xe (Stop Line) nằm cách tâm giao lộ 28.0m (giao lộ mở rộng).
                # Chỉ kích hoạt dừng đèn đỏ khi xe ĐANG TIẾN VÀO trước vạch dừng (0.0 <= d_stop_cand < 45.0m).
                # Xe đã vượt qua vạch dừng (d_stop_cand < 0.0) phải tiếp tục giải phóng giao lộ, không được dừng khựng lại!
                d_stop_cand = forward_l - 28.0
                if 0.0 <= d_stop_cand < 45.0 and lateral_l < 26.0:
                    approaches = light.get("approaches", {})
                    matched_sig = None
                    if approaches:
                        best_dot = -1.0
                        for nb in network_manager.graph.to_undirected().neighbors(nid):
                            nb_pos = network_manager.get_node_pos(nb)
                            arm_dx = pos[0] - nb_pos[0]
                            arm_dy = pos[1] - nb_pos[1]
                            arm_len = max(1e-3, math.hypot(arm_dx, arm_dy))
                            dot = (arm_dx / arm_len) * cos_h + (arm_dy / arm_len) * sin_h
                            if dot > best_dot:
                                best_dot = dot
                                matched_sig = approaches.get(nb)

                    is_left_turn = False
                    if getattr(self, "lane_id", 1) == 2 or getattr(self, "turn_signal", "NONE") == "LEFT":
                        is_left_turn = True
                    elif self.path_waypoints and len(self.path_waypoints) >= 4:
                        best_wp_d = 1e9
                        best_wp_k = -1
                        for k, wp in enumerate(self.path_waypoints):
                            d_wp = math.hypot(wp[0] - pos[0], wp[1] - pos[1])
                            if d_wp < best_wp_d:
                                best_wp_d = d_wp
                                best_wp_k = k
                        if best_wp_d < 38.0 and 1 <= best_wp_k < len(self.path_waypoints) - 1:
                            wp_pre = self.path_waypoints[max(0, best_wp_k - 2)]
                            wp_mid = self.path_waypoints[best_wp_k]
                            wp_post = self.path_waypoints[min(len(self.path_waypoints) - 1, best_wp_k + 2)]
                            h_pre = math.atan2(wp_mid[1] - wp_pre[1], wp_mid[0] - wp_pre[0])
                            h_post = math.atan2(wp_post[1] - wp_mid[1], wp_post[0] - wp_mid[0])
                            diff = (h_post - h_pre + math.pi) % (2 * math.pi) - math.pi
                            if diff > 0.35:
                                is_left_turn = True

                    if is_left_turn and 0.0 < d_stop_cand < 35.0:
                        self.turn_signal = "LEFT"

                    if matched_sig:
                        if is_left_turn:
                            check_state = matched_sig.get("left_state", "red").upper()
                        else:
                            check_state = matched_sig.get("straight_state", "green").upper()
                    else:
                        check_state = light.get("state", "NONE").upper()

                    if check_state in ["RED", "YELLOW"]:
                        red_light_active = True
                        dist_to_stop = d_stop_cand
                        break

        # -----------------------------------------------------------------
        # TÍNH TOÁN LỰC PHANH VÀ GA AN TOÀN (SAFE ACCELERATION COMPUTATION)
        # -----------------------------------------------------------------
        # 1. Gia tốc bám đuôi xe trước (Car-following / Queueing):
        hold_behind_lead = False
        if lead_vehicle is not None:
            if front_veh_speed < 0.3:
                # Xe trước đang dừng (đèn đỏ hoặc kẹt xe):
                if min_bumper_gap <= 2.2:
                    accel_front = -self.max_decel
                    hold_behind_lead = True
                elif min_bumper_gap < 4.0:
                    # Giữ khoảng cách êm ái 2.5m - 3.5m: nếu đã dừng thì đứng im, nếu đang chạy thì phanh nhẹ
                    accel_front = -self.max_decel * 0.6 if self.speed > 0.5 else 0.0
                elif min_bumper_gap < 8.0:
                    # Có khoảng hở an toàn phía trước: mớm nhẹ 2.0 m/s để bám hàng chờ, chống đứng yên
                    target_v = 2.0
                    accel_front = (target_v - self.speed) * 0.6 if self.speed < target_v else -0.5
                else:
                    target_v = 3.5
                    accel_front = (target_v - self.speed) * 0.6
            else:
                # Xe trước đang chuyển động (bắt đầu khởi hành hoặc đang chạy):
                if min_bumper_gap <= 1.8:
                    accel_front = -self.max_decel
                    hold_behind_lead = True
                elif min_bumper_gap < 4.5:
                    target_v = max(1.5, front_veh_speed - 0.2)
                    accel_front = (target_v - self.speed) * 0.8
                elif min_bumper_gap < 9.0:
                    target_v = max(2.5, front_veh_speed)
                    accel_front = (target_v - self.speed) * 0.7
                else:
                    target_spd = getattr(self, "curve_safe_speed", self.target_speed)
                    accel_front = (target_spd - self.speed) * 0.65
        else:
            target_spd = getattr(self, "curve_safe_speed", self.target_speed)
            accel_front = (target_spd - self.speed) * 0.65

        # 2. Gia tốc dừng đèn đỏ (Traffic Light Braking):
        hold_at_stop_line = False
        target_spd = getattr(self, "curve_safe_speed", self.target_speed)
        if red_light_active:
            if 0.0 <= dist_to_stop <= 2.2:
                accel_light = -self.max_decel
                hold_at_stop_line = True
            elif dist_to_stop < 6.5:
                accel_light = -self.max_decel
                if self.speed > 1.0:
                    self.speed = max(0.0, self.speed - 0.8)
            elif dist_to_stop < 15.0 and self.speed > 2.5:
                accel_light = -self.max_decel * 0.7
            elif dist_to_stop < 25.0 and self.speed > 4.5:
                accel_light = -self.max_decel * 0.45
            elif self.speed > 7.0:
                accel_light = -self.max_decel * 0.25
            else:
                accel_light = 0.0
        else:
            accel_light = (target_spd - self.speed) * 0.65

        # 3. Phối hợp ưu tiên an toàn (AN TOÀN CHỐNG ĐÂM XE TRƯỚC LÀ ƯU TIÊN SỐ 1):
        if is_at_end:
            # Ở cuối hành trình: vẫn KHÔNG ĐƯỢC đâm vào xe khác hoặc vượt đèn đỏ!
            accel = (self.target_speed - self.speed) * 0.5
            if (lead_vehicle is not None and min_bumper_gap <= 2.5) or hold_at_stop_line:
                self.speed = 0.0
                accel = -self.max_decel
        elif hold_behind_lead:
            # Dừng xe kiên nhẫn trong hàng chờ phía sau xe trước:
            self.speed = 0.0
            accel = -getattr(self, "max_decel", 3.6)
        elif hold_at_stop_line:
            # Dừng xe trước vạch dừng đèn đỏ (nếu không có xe khác chắn trước):
            self.speed = 0.0
            accel = -getattr(self, "max_decel", 3.6)
        else:
            # Lấy lực phanh mạnh hơn giữa xe trước và đèn đỏ:
            accel = min(accel_front, accel_light)

        # 4. KHÓA VẬT LÝ TUYỆT ĐỐI CHỐNG XUYÊN THÂN XE (HARD RIGID-BODY NON-PENETRATION):
        # Nếu khoảng cách mũi xe sau và đuôi xe trước chớm chạm (< 0.6m):
        if lead_vehicle is not None and min_bumper_gap < 0.6:
            self.speed = 0.0
            accel = -getattr(self, "max_decel", 3.6)
            penetration = 1.0 - min_bumper_gap
            if penetration > 0:
                self.x -= cos_h * penetration
                self.y -= sin_h * penetration

        # Nếu phát hiện 2 xe bị đè hình học lên nhau (Bounding Box Overlap):
        if overlapping_vehicle is not None:
            self.speed = 0.0
            accel = -getattr(self, "max_decel", 3.6)

        self.step(steer=steer, accel=accel, dt=dt)
        self.update_lateral_offset(dt)

    def update_lateral_offset(self, dt: float = 0.1) -> None:
        diff = self.target_lateral_offset - self.current_lateral_offset
        # Tốc độ chuyển làn thực tế (~2.0 m/s): chuyển trọn vẹn làn đường 5m trong ~2.5 giây
        shift_speed = 2.0
        if abs(diff) > 0.05:
            delta_off = math.copysign(min(abs(diff), shift_speed * dt), diff)
            self.current_lateral_offset += delta_off
            # Góc nghiêng thân xe khi chuyển làn (Lane Change Yaw Steering Angle):
            # Khi rẽ sang trái (diff > 0): đầu xe chếch nhẹ sang trái (+0.12 rad ~ 7 độ)
            # Khi rẽ sang phải (diff < 0): đầu xe chếch nhẹ sang phải (-0.12 rad ~ -7 độ)
            target_yaw = math.copysign(min(0.14, abs(diff) * 0.05), diff)
            cur_yaw = getattr(self, "lane_change_yaw", 0.0)
            self.lane_change_yaw = cur_yaw + np.clip(target_yaw - cur_yaw, -1.0 * dt, 1.0 * dt)
        else:
            self.current_lateral_offset = self.target_lateral_offset
            cur_yaw = getattr(self, "lane_change_yaw", 0.0)
            # Xe đã vào đúng làn: tự động trả lái về thẳng hàng với làn đường
            if abs(cur_yaw) > 0.01:
                self.lane_change_yaw = cur_yaw - math.copysign(min(abs(cur_yaw), 1.2 * dt), cur_yaw)
            else:
                self.lane_change_yaw = 0.0

    def get_bounding_box(self) -> List[Tuple[float, float]]:
        eff_h = self.heading + getattr(self, "lane_change_yaw", 0.0)
        cos_h = math.cos(eff_h)
        sin_h = math.sin(eff_h)

        offset = getattr(self, "current_lateral_offset", 0.0)
        # Sử dụng road heading ổn định để không bị văng đòn bẩy khi đánh lái
        rh = getattr(self, "current_road_heading", self.heading)
        actual_x = self.x - math.sin(rh) * offset
        actual_y = self.y + math.cos(rh) * offset

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
        """
        Kiểm tra va chạm chuẩn xác bằng Định lý Tách Trục (Separating Axis Theorem - SAT)
        cho hình chữ nhật định hướng (Oriented Bounding Box - OBB).
        Triệt tiêu 100% hiện tượng báo va chạm giả giữa 2 làn đường cạnh nhau (ví dụ làn 3 và làn 2)!
        """
        corners1 = self.get_bounding_box()
        corners2 = other.get_bounding_box()

        # Bác bỏ nhanh bằng khoảng cách tâm bao (Quick Circle Reject)
        c1x = sum(c[0] for c in corners1) * 0.25
        c1y = sum(c[1] for c in corners1) * 0.25
        c2x = sum(c[0] for c in corners2) * 0.25
        c2y = sum(c[1] for c in corners2) * 0.25

        max_reach1 = math.hypot(self.length * 0.5, self.width * 0.5)
        max_reach2 = math.hypot(other.length * 0.5, other.width * 0.5)
        if math.hypot(c1x - c2x, c1y - c2y) > (max_reach1 + max_reach2):
            return False

        # 4 trục kiểm tra phân tách (2 pháp tuyến của xe 1 và 2 pháp tuyến của xe 2)
        axes = [
            (math.cos(self.heading), math.sin(self.heading)),
            (-math.sin(self.heading), math.cos(self.heading)),
            (math.cos(other.heading), math.sin(other.heading)),
            (-math.sin(other.heading), math.cos(other.heading)),
        ]

        for ax, ay in axes:
            proj1 = [cx * ax + cy * ay for cx, cy in corners1]
            min1, max1 = min(proj1), max(proj1)

            proj2 = [cx * ax + cy * ay for cx, cy in corners2]
            min2, max2 = min(proj2), max(proj2)

            # Nếu hình chiếu trên bất kỳ trục nào không chồng lấn (có khe hở > 0.05m) -> Không va chạm!
            if max1 < min2 + 0.05 or max2 < min1 + 0.05:
                return False

        # Chồng lấn trên tất cả 4 trục -> Thực sự có va chạm vật lý!
        return True
