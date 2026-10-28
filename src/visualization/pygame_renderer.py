"""
Realistic 2D Autonomous Driving Visualization Engine (Pygame).
Features:
- Pan & Zoom Controls (Kéo di chuyển và Thu phóng bản đồ):
  - Cuộn chuột (Mouse Wheel Up/Down) hoặc phím (+/-) để Zoom In / Zoom Out từ 1.0x đến 6.0x.
  - Nhấp giữ chuột phải hoặc chuột giữa và kéo rê (Mouse Drag) để di chuyển tự do khắp bản đồ.
  - Phím [C] hoặc nút [BẬT/TẮT THEO DÕI XE] để chuyển đổi giữa chế độ tự do quan sát và chế độ camera bám theo xe.
- Multi-Route Proposal Selection on BOTH Main Map and Minimap:
  - Vẽ tất cả lộ trình đề xuất (A* né kẹt xe, Dijkstra ngắn nhất, Đại lộ lớn, Điểm ghé qua) lên cả bản đồ lớn và Minimap.
  - Tuyến được chọn phát sáng Neon rực rỡ, các tuyến phụ hiển thị màu mờ.
  - Nhấp trực tiếp vào thẻ hoặc tuyến đường để đổi thuật toán tìm đường ngay tức khắc.
"""

from typing import Optional, List, Tuple, Dict, Any
import math
import sys
import pygame
import numpy as np

from src.integration.hybrid_controller import HybridNavigationController


class PygameRenderer:
    # Color Palette
    BG_GRASS = (20, 26, 22)
    SIDEWALK = (50, 56, 65)
    ROAD_ASPHALT = (32, 36, 44)
    LANE_SOLID_YELLOW = (255, 204, 0)
    LANE_DIVIDER_WHITE = (180, 190, 205)

    EGO_BODY = (0, 230, 118)          # Vibrant Neon Emerald
    MOTORCYCLE_COLOR = (0, 200, 255)  # Cyan
    TRUCK_COLOR = (255, 145, 0)       # Amber
    CAR_COLOR = (255, 75, 75)         # Crimson
    BLINKER_YELLOW = (255, 230, 0)

    # Route proposal colors
    ROUTE_COLORS = [
        (0, 230, 118),   # Route 1: A* Fast - Emerald
        (41, 182, 246),  # Route 2: Dijkstra Shortest - Sky Blue
        (186, 104, 200), # Route 3: Highway Bypass - Violet
        (255, 167, 38)   # Route 4: Custom Via Stop - Orange
    ]

    HUD_BG = (16, 20, 28, 245)
    HUD_BORDER = (45, 55, 75)
    TEXT_MAIN = (240, 245, 255)
    TEXT_MUTED = (160, 175, 195)
    ACCENT_BLUE = (0, 176, 255)

    def __init__(
        self,
        controller: HybridNavigationController,
        width: int = 1440,
        height: int = 820,
        scale: float = 2.2
    ):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.scale = scale  # Zoom level
        self.controller = controller
        self.env = controller.env

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Smart Autonomous Driving - Pan & Zoom Multi-Route Simulator")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_tiny = pygame.font.SysFont("Segoe UI", 12)
        self.font_small = pygame.font.SysFont("Segoe UI", 13)
        self.font_mid = pygame.font.SysFont("Segoe UI", 15, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_title = pygame.font.SysFont("Segoe UI", 21, bold=True)
        self.font_banner = pygame.font.SysFont("Segoe UI", 24, bold=True)

        # Camera & Pan/Zoom State
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.is_dragging_map = False
        self.drag_start_mouse = (0, 0)
        self.drag_start_cam = (0.0, 0.0)
        self.follow_ego_camera = True  # True: Camera auto follows car, False: User freely pans

        # UI State
        self.sim_state = "CONFIGURING"
        self.retry_count = 0
        self.last_crash_reason = ""
        self.selection_mode: Optional[str] = None

        # Simulation Speed Multiplier (0.5x, 1x, 2x, 5x, 10x, 20x, 50x)
        self.sim_speed_multiplier = 1.0
        self.SPEED_OPTIONS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        self.speed_buttons: List[Tuple[pygame.Rect, float]] = []

        # Display Toggles (Bật / Tắt hiển thị Bảng điều khiển trái, Minimap & Tham số xe Telemetry HUD)
        self.show_left_dashboard: bool = True
        self.show_minimap: bool = True
        self.show_telemetry: bool = True

        # Reinforcement Learning Online Training Toggle (Bật / Tắt học tăng cường trực tiếp)
        self.rl_training_enabled = False       # True: Model PPO tự học và update trọng số; False: Chạy mô phỏng thường
        self.rl_updates_count = 0              # Số lần đã lan truyền ngược cập nhật trọng số
        self.rl_total_timesteps = 0            # Tổng số bước nạp vào mô hình PPO

        # Auto Retry & Continuous Training Mode
        self.auto_retry_until_success = False  # Tự động thử lại liên tục đến khi thành công
        self.auto_training_mode = False        # Tự động huấn luyện: Thành công -> chuyển tuyến mới, Thất bại -> thử lại
        self.training_episodes_completed = 0
        self.training_success_count = 0
        self.deterministic_traffic = True      # Giữ nguyên cùng điều kiện mật độ xe cộ khi thử lại
        self.traffic_seed = 42                 # Seed cố định cho điều kiện môi trường


        self.interactive_nodes: List[Dict[str, Any]] = []
        self.minimap_nodes: List[Dict[str, Any]] = []
        self.proposal_buttons: List[Tuple[pygame.Rect, int]] = []

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        sx = int((wx - self.cam_x) * self.scale + self.width * 0.48)
        sy = int(self.height * 0.52 - (wy - self.cam_y) * self.scale)
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        """Converts screen pixel coordinates back to world 2D coordinates."""
        wx = (sx - self.width * 0.48) / self.scale + self.cam_x
        wy = (self.height * 0.52 - sy) / self.scale + self.cam_y
        return wx, wy

    def draw_environment_background(self) -> None:
        self.screen.fill(self.BG_GRASS)

    def draw_multi_lane_roads(self) -> None:
        G = self.controller.net.graph

        drawn_edges = set()
        for u, v in G.edges():
            edge_key = tuple(sorted([u, v]))
            is_bidirectional = G[u][v].get("bidirectional", True)

            # Avoid rendering the same 2-way road twice
            if is_bidirectional and edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)

            pos_u = G.nodes[u]["pos"]
            pos_v = G.nodes[v]["pos"]
            su = self.world_to_screen(*pos_u)
            sv = self.world_to_screen(*pos_v)

            lanes = G[u][v].get("lanes", 3)

            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]
            mag = max(1e-3, math.hypot(dx, dy))
            px = -dy / mag
            py = dx / mag

            density = G[u][v].get("density", 0.0)
            is_prohibited = G[u][v].get("is_prohibited", False)
            if is_prohibited:
                road_col = (70, 25, 25)
            else:
                r = int(32 + density * 130)
                g = int(36 * (1.0 - density * 0.6))
                b = int(44 * (1.0 - density * 0.6))
                road_col = (r, g, b)

            if is_bidirectional:
                # =========================================================================
                # ĐƯỜNG 2 CHIỀU: 2 LÒNG ĐƯỜNG RIÊNG BIỆT NỐI LẠI VỚI DẢI PHÂN CÁCH CỨNG Ở GIỮA
                # Mỗi chiều 3 làn xe rộng rãi (mỗi làn rộng 5.0m - mở rộng 65% diện tích mặt đường):
                # - Chiều đi bên phải: offset -10.0m (Làn 0 le phai -15.0m, Làn 1 giua -10.0m, Làn 2 sat dai -5.0m)
                # - Chiều đi ngược lại: offset +10.0m (Làn 2 sat dai +5.0m, Làn 1 giua +10.0m, Làn 0 le phai +15.0m)
                # - Dải phân cách giữa: [-2.5m, +2.5m] (rộng 5.0m) cỏ xanh / gờ bê tông ngăn cách tuyệt đối
                # =========================================================================
                carr_width_px = max(24, int(42 * self.scale / 2.5))     # Bề rộng mỗi lòng đường (15.0m ~ 3 làn xe rộng)
                median_width_px = max(6, int(14 * self.scale / 2.5))    # Bề rộng dải phân cách giữa (5.0m)
                sidewalk_margin_px = max(4, int(6 * self.scale / 2.5))

                # Tọa độ tim đường chiều bên phải (Forward Carriageway center = -10.0m)
                fwd_u = (pos_u[0] - px * 10.0, pos_u[1] - py * 10.0)
                fwd_v = (pos_v[0] - px * 10.0, pos_v[1] - py * 10.0)
                s_fwd_u = self.world_to_screen(*fwd_u)
                s_fwd_v = self.world_to_screen(*fwd_v)

                # Tọa độ tim đường chiều ngược lại (Oncoming Carriageway center = +10.0m)
                onc_u = (pos_u[0] + px * 10.0, pos_u[1] + py * 10.0)
                onc_v = (pos_v[0] + px * 10.0, pos_v[1] + py * 10.0)
                s_onc_u = self.world_to_screen(*onc_u)
                s_onc_v = self.world_to_screen(*onc_v)

                # 1. Vẽ Vỉa hè / Lề đường bên ngoài cho cả 2 chiều
                total_corridor_px = carr_width_px * 2 + median_width_px + sidewalk_margin_px * 2
                pygame.draw.line(self.screen, self.SIDEWALK, su, sv, total_corridor_px)

                # 2. Vẽ 2 lòng đường nhựa riêng biệt (Asphalt cho mỗi chiều)
                pygame.draw.line(self.screen, road_col, s_fwd_u, s_fwd_v, carr_width_px)
                pygame.draw.line(self.screen, road_col, s_onc_u, s_onc_v, carr_width_px)

                # Bo góc giao lộ cho 2 lòng đường
                pygame.draw.circle(self.screen, road_col, s_fwd_u, carr_width_px // 2)
                pygame.draw.circle(self.screen, road_col, s_fwd_v, carr_width_px // 2)
                pygame.draw.circle(self.screen, road_col, s_onc_u, carr_width_px // 2)
                pygame.draw.circle(self.screen, road_col, s_onc_v, carr_width_px // 2)

                # 3. Vẽ DẢI PHÂN CÁCH CỨNG Ở GIỮA (Central Median Strip: gờ bê tông + thảm cỏ xanh viền vàng)
                pygame.draw.line(self.screen, (35, 75, 45), su, sv, median_width_px)
                # Viền dải phân cách (gờ bê tông / vạch vàng ranh giới)
                sep_half = median_width_px / 2.0
                m_edge1_u = (int(su[0] + px * sep_half), int(su[1] + py * sep_half))
                m_edge1_v = (int(sv[0] + px * sep_half), int(sv[1] + py * sep_half))
                m_edge2_u = (int(su[0] - px * sep_half), int(su[1] - py * sep_half))
                m_edge2_v = (int(sv[0] - px * sep_half), int(sv[1] - py * sep_half))
                pygame.draw.line(self.screen, self.LANE_SOLID_YELLOW, m_edge1_u, m_edge1_v, max(1, int(1.5 * self.scale / 2.5)))
                pygame.draw.line(self.screen, self.LANE_SOLID_YELLOW, m_edge2_u, m_edge2_v, max(1, int(1.5 * self.scale / 2.5)))

                # 4. Vẽ VẠCH PHÂN LÀN ĐỨT ĐOẠN TRẮNG (3 làn xe mỗi chiều)
                # Chiều bên phải: phân chia làn xe máy (-15.0m) với làn giữa (-10.0m) tại -12.5m, và làn giữa với làn vượt (-5.0m) tại -7.5m
                for off in [-12.5, -7.5, 7.5, 12.5]:
                    p1 = (pos_u[0] + px * off, pos_u[1] + py * off)
                    p2 = (pos_v[0] + px * off, pos_v[1] + py * off)
                    pygame.draw.line(self.screen, (160, 180, 205), self.world_to_screen(*p1), self.world_to_screen(*p2), 1)

            else:
                # ĐƯỜNG 1 CHIỀU: 3 làn đường cùng chiều với vạch trắng và mũi tên chỉ hướng
                road_w = int((30 + lanes * 28) * self.scale / 2.5)
                pygame.draw.line(self.screen, self.SIDEWALK, su, sv, road_w + max(6, int(8 * self.scale / 2.5)))
                pygame.draw.line(self.screen, road_col, su, sv, road_w)
                pygame.draw.circle(self.screen, road_col, su, road_w // 2)
                pygame.draw.circle(self.screen, road_col, sv, road_w // 2)

                # Vạch phân chia 3 làn đường
                for offset_val in [-5.0, 5.0]:
                    p1_off = (pos_u[0] + px * offset_val, pos_u[1] + py * offset_val)
                    p2_off = (pos_v[0] + px * offset_val, pos_v[1] + py * offset_val)
                    pygame.draw.line(self.screen, (160, 180, 205), self.world_to_screen(*p1_off), self.world_to_screen(*p2_off), 1)

                # Mũi tên chỉ hướng lưu thông
                mid_x = (pos_u[0] + pos_v[0]) * 0.5
                mid_y = (pos_u[1] + pos_v[1]) * 0.5
                s_mid = self.world_to_screen(mid_x, mid_y)
                dir_x = (dx / mag) * 14 * (self.scale / 2.5)
                dir_y = (-dy / mag) * 14 * (self.scale / 2.5)
                tip = (int(s_mid[0] + dir_x), int(s_mid[1] + dir_y))
                left_wing = (int(s_mid[0] - dir_x * 0.5 - dir_y * 0.5), int(s_mid[1] - dir_y * 0.5 + dir_x * 0.5))
                right_wing = (int(s_mid[0] - dir_x * 0.5 + dir_y * 0.5), int(s_mid[1] - dir_y * 0.5 - dir_x * 0.5))
                pygame.draw.polygon(self.screen, (0, 230, 255), [tip, left_wing, right_wing])

        # Intersection nodes (seamless junction hubs & Roundabouts)
        self.interactive_nodes.clear()
        for node_id, data in G.nodes(data=True):
            pos = data["pos"]
            sn = self.world_to_screen(*pos)
            is_roundabout = data.get("is_roundabout", False)
            deg = G.to_undirected().degree(node_id) if node_id in G else 0

            # Phân loại độ lớn giao lộ (Ngã 3, Ngã 4, Ngã 5+ và Vòng xuyến lớn)
            # Mở rộng mạnh mẽ không gian lòng đường giao lộ để đảm bảo 100% các dòng xe lưu thông rộng rãi, thoát nút giao thông suốt
            if is_roundabout or deg >= 5:
                # Ngã 5+ hoặc Vòng xuyến lớn (Roundabout): Bán kính lòng đường 125px (~50m), đảo trung tâm 40px (~16m)
                # Bề rộng lòng đường vòng xuyến lên tới 34m (thoải mái cho 4-5 làn xe lưu thông mà không lo chạm viền)
                rb_radius = int(125 * self.scale / 2.5)
                island_radius = int(40 * self.scale / 2.5)

                # 1. Vỉa hè bao ngoài vòng xuyến
                pygame.draw.circle(self.screen, self.SIDEWALK, sn, rb_radius + 6)
                # 2. Lòng đường nhựa vòng xuyến (Lòng đường siêu rộng)
                pygame.draw.circle(self.screen, self.ROAD_ASPHALT, sn, rb_radius)
                # 3. Các vạch phân làn tròn đứt đoạn bên trong vòng xuyến
                lane_step = (rb_radius - island_radius) / 4.0
                for ls in range(1, 4):
                    pygame.draw.circle(self.screen, (160, 180, 205), sn, int(island_radius + lane_step * ls), 1)

                # 4. Đảo cỏ trung tâm vòng xuyến (Roundabout Central Island)
                pygame.draw.circle(self.screen, (30, 85, 45), sn, island_radius)
                pygame.draw.circle(self.screen, self.SIDEWALK, sn, island_radius, 2)
                # Biểu tượng mũi tên xoay tròn trên đảo giao thông
                lbl_rb = self.font_tiny.render("⟳", True, (255, 255, 255))
                self.screen.blit(lbl_rb, (sn[0] - lbl_rb.get_width() // 2, sn[1] - lbl_rb.get_height() // 2))

                radius = rb_radius
            elif deg == 4:
                # Ngã 4 (4-way intersection): Mở rộng thành 85px (~ 34.0m)
                radius = int(85 * self.scale / 2.5)
            else:
                # Ngã 3 (3-way intersection): Mở rộng thành 70px (~ 28.0m)
                radius = int(70 * self.scale / 2.5)

            if not (is_roundabout or deg >= 5):
                is_start = (node_id == self.controller.start_node)
                is_goal = (node_id == self.controller.goal_node)
                is_via = (node_id == getattr(self.controller, "via_node", None))

                if is_start:
                    pygame.draw.circle(self.screen, (0, 200, 255), sn, radius + 6)
                elif is_goal:
                    pygame.draw.circle(self.screen, (255, 60, 60), sn, radius + 6)
                elif is_via:
                    pygame.draw.circle(self.screen, (255, 167, 38), sn, radius + 6)

                pygame.draw.circle(self.screen, self.ROAD_ASPHALT, sn, radius)
                pygame.draw.circle(self.screen, self.SIDEWALK, sn, radius, 2)

            lbl = self.font_tiny.render(data.get("name", node_id), True, (255, 255, 255))
            self.screen.blit(lbl, (sn[0] - lbl.get_width() // 2, sn[1] + radius + 4))

            self.interactive_nodes.append({
                "id": node_id,
                "screen_pos": sn,
                "radius": max(16, radius),
                "name": data.get("name", node_id)
            })

    def draw_traffic_signs(self) -> None:
        if not hasattr(self.controller.net, "traffic_signs"):
            return

        for sign in self.controller.net.traffic_signs:
            sp = self.world_to_screen(*sign["pos"])
            stype = sign["type"]

            # Sign pole on sidewalk
            pygame.draw.line(self.screen, (150, 160, 175), (sp[0], sp[1]), (sp[0], sp[1] + 10), 2)
            pygame.draw.circle(self.screen, (70, 75, 85), (sp[0], sp[1] + 10), 3)

            if stype == "SPEED_LIMIT":
                pygame.draw.circle(self.screen, (230, 40, 40), sp, 12)
                pygame.draw.circle(self.screen, (255, 255, 255), sp, 10)
                txt = self.font_tiny.render(sign["label"], True, (20, 20, 20))
                self.screen.blit(txt, (sp[0] - txt.get_width() // 2, sp[1] - txt.get_height() // 2))
            elif stype == "NO_ENTRY":
                pygame.draw.circle(self.screen, (230, 40, 40), sp, 12)
                pygame.draw.rect(self.screen, (255, 255, 255), (sp[0] - 8, sp[1] - 2, 16, 4))
            elif stype == "ONE_WAY":
                # Biển báo đường 1 chiều (Hình chữ nhật màu xanh dương với mũi tên trắng)
                rect_ow = pygame.Rect(sp[0] - 12, sp[1] - 8, 24, 16)
                pygame.draw.rect(self.screen, (0, 102, 204), rect_ow, border_radius=3)
                pygame.draw.rect(self.screen, (255, 255, 255), rect_ow, 1, border_radius=3)
                t_ow = self.font_tiny.render("➔", True, (255, 255, 255))
                self.screen.blit(t_ow, (sp[0] - 5, sp[1] - 7))

        # Render Dual Directional Traffic Lights (1 Cột đèn Đi Thẳng cho Làn 2, 1 Cột đèn Rẽ Trái cho Làn 3)
        if hasattr(self.controller.net, "traffic_lights"):
            G = self.controller.net.graph
            for nid, light in self.controller.net.traffic_lights.items():
                if nid not in G:
                    continue
                pos_junc = G.nodes[nid]["pos"]
                approaches = light.get("approaches", {})

                # Tìm các nhánh đường nối vào nút giao thông này
                neighbors = list(G.to_undirected().neighbors(nid))
                for nb in neighbors:
                    pos_nb = G.nodes[nb]["pos"]
                    dx = pos_junc[0] - pos_nb[0]
                    dy = pos_junc[1] - pos_nb[1]
                    dist_arm = max(1e-3, math.hypot(dx, dy))
                    ux = dx / dist_arm
                    uy = dy / dist_arm
                    # Véc-tơ pháp tuyến sang phải theo hướng đi vào ngã tư: (uy, -ux)

                    # Lấy thông số đèn riêng của nhánh đường này (Làn 2 đi thẳng & Làn 3 rẽ trái)
                    sig = approaches.get(nb, {
                        "straight_state": light.get("state", "green"),
                        "straight_timer": light.get("timer", 10.0),
                        "left_state": "red",
                        "left_timer": 15.0
                    })
                    s_state = sig.get("straight_state", "green")
                    s_timer = max(0, int(math.ceil(sig.get("straight_timer", 0.0))))
                    l_state = sig.get("left_state", "red")
                    l_timer = max(0, int(math.ceil(sig.get("left_timer", 0.0))))

                    # Vạch dừng đèn đỏ (Stop line) lùi ra cách tâm giao lộ 28.0m (phù hợp với giao lộ mở rộng)
                    # Chiều rộng lòng đường 15.0m từ sát dải phân cách (+2.5m) tới vỉa hè (+18.0m)
                    p_stop1 = (pos_junc[0] - ux * 28.0 + uy * 2.5, pos_junc[1] - uy * 28.0 - ux * 2.5)
                    p_stop2 = (pos_junc[0] - ux * 28.0 + uy * 18.0, pos_junc[1] - uy * 28.0 - ux * 18.0)
                    s_p_stop1 = self.world_to_screen(*p_stop1)
                    s_p_stop2 = self.world_to_screen(*p_stop2)

                    # 1. Đường viền phát quang cảnh báo dừng (Glow Stop Line theo trạng thái đèn)
                    stop_glow_col = (255, 60, 60) if (s_state == "red" or l_state == "red") else ((255, 200, 0) if (s_state == "yellow" or l_state == "yellow") else (100, 240, 150))
                    pygame.draw.line(self.screen, stop_glow_col, s_p_stop1, s_p_stop2, max(5, int(7 * self.scale / 2.5)))
                    # 2. Vạch sơn trắng liền nét dày chuẩn quốc tế (Solid White Stop Bar)
                    pygame.draw.line(self.screen, (255, 255, 255), s_p_stop1, s_p_stop2, max(3, int(4 * self.scale / 2.5)))

                    # 3. Chữ "STOP" sơn trắng nổi bật trên mặt đường nhựa ngay trước vạch dừng
                    p_txt = (pos_junc[0] - ux * 31.5 + uy * 10.25, pos_junc[1] - uy * 31.5 - ux * 10.25)
                    s_txt = self.world_to_screen(*p_txt)
                    lbl_stop = self.font_tiny.render("STOP", True, (255, 255, 255))
                    self.screen.blit(lbl_stop, (s_txt[0] - lbl_stop.get_width() // 2, s_txt[1] - lbl_stop.get_height() // 2))

                    # -------------------------------------------------------------
                    # ĐÈN 1: CỘT ĐÈN ĐI THẲNG (LÀN 2 & 1 - OFFSET ~ 19.5M BÊN PHẢI VỈA HÈ)
                    # -------------------------------------------------------------
                    tl_s_wx = pos_junc[0] - ux * 30.0 + uy * 19.5
                    tl_s_wy = pos_junc[1] - uy * 30.0 - ux * 19.5
                    s_tl_s = self.world_to_screen(tl_s_wx, tl_s_wy)

                    col_s_r = (255, 40, 40) if s_state == "red" else (60, 20, 20)
                    col_s_y = (255, 215, 0) if s_state == "yellow" else (60, 50, 15)
                    col_s_g = (0, 230, 118) if s_state == "green" else (15, 55, 25)

                    box_w, box_h = 10, 26
                    bx_s = s_tl_s[0] - box_w // 2
                    by_s = s_tl_s[1] - box_h // 2
                    pygame.draw.rect(self.screen, (20, 22, 28), (bx_s, by_s, box_w, box_h), border_radius=3)
                    pygame.draw.rect(self.screen, (120, 130, 145), (bx_s, by_s, box_w, box_h), 1, border_radius=3)
                    pygame.draw.circle(self.screen, col_s_r, (bx_s + box_w // 2, by_s + 5), 3)
                    pygame.draw.circle(self.screen, col_s_y, (bx_s + box_w // 2, by_s + 13), 3)
                    pygame.draw.circle(self.screen, col_s_g, (bx_s + box_w // 2, by_s + 21), 3)

                    # Nhãn biểu tượng Đi Thẳng (↑) kèm số giây đếm ngược
                    timer_s_col = col_s_r if s_state == "red" else (col_s_y if s_state == "yellow" else col_s_g)
                    txt_s = self.font_tiny.render(f"↑{s_timer}", True, timer_s_col)
                    self.screen.blit(txt_s, (bx_s - 2, by_s - 14))

                    # -------------------------------------------------------------
                    # ĐÈN 2: CỘT ĐÈN RẼ TRÁI (LÀN 3 - SÁT DẢI PHÂN CÁCH TRUNG TÂM ~ 2.0M)
                    # -------------------------------------------------------------
                    tl_l_wx = pos_junc[0] - ux * 30.0 + uy * 2.0
                    tl_l_wy = pos_junc[1] - uy * 30.0 - ux * 2.0
                    s_tl_l = self.world_to_screen(tl_l_wx, tl_l_wy)

                    col_l_r = (255, 40, 40) if l_state == "red" else (60, 20, 20)
                    col_l_y = (255, 215, 0) if l_state == "yellow" else (60, 50, 15)
                    col_l_g = (0, 230, 118) if l_state == "green" else (15, 55, 25)

                    bx_l = s_tl_l[0] - box_w // 2
                    by_l = s_tl_l[1] - box_h // 2
                    pygame.draw.rect(self.screen, (20, 22, 28), (bx_l, by_l, box_w, box_h), border_radius=3)
                    pygame.draw.rect(self.screen, (100, 180, 240), (bx_l, by_l, box_w, box_h), 1, border_radius=3)
                    pygame.draw.circle(self.screen, col_l_r, (bx_l + box_w // 2, by_l + 5), 3)
                    pygame.draw.circle(self.screen, col_l_y, (bx_l + box_w // 2, by_l + 13), 3)
                    pygame.draw.circle(self.screen, col_l_g, (bx_l + box_w // 2, by_l + 21), 3)

                    # Nhãn biểu tượng Rẽ Trái (↰) kèm số giây đếm ngược
                    timer_l_col = col_l_r if l_state == "red" else (col_l_y if l_state == "yellow" else col_l_g)
                    txt_l = self.font_tiny.render(f"↰{l_timer}", True, timer_l_col)
                    self.screen.blit(txt_l, (bx_l - 2, by_l - 14))

    def draw_all_route_proposals_on_map(self) -> None:
        proposals = getattr(self.controller, "route_proposals", [])
        active_idx = getattr(self.controller, "active_proposal_index", 0)

        # 1. Inactive routes
        for idx, prop in enumerate(proposals):
            if idx == active_idx:
                continue

            wps = prop.get("waypoints", [])
            col = self.ROUTE_COLORS[idx % len(self.ROUTE_COLORS)]
            muted_col = (int(col[0] * 0.45), int(col[1] * 0.45), int(col[2] * 0.45))

            for i in range(len(wps) - 1):
                s1 = self.world_to_screen(*wps[i])
                s2 = self.world_to_screen(*wps[i + 1])
                pygame.draw.line(self.screen, muted_col, s1, s2, max(2, int(4 * self.scale / 2.5)))

        # 2. Active selected route with neon glow
        if proposals:
            active_prop = proposals[active_idx]
            wps = self.env.waypoints
            curr_idx = self.env.current_wp_idx
            active_color = self.ROUTE_COLORS[active_idx % len(self.ROUTE_COLORS)]

            for i in range(len(wps) - 1):
                s1 = self.world_to_screen(*wps[i])
                s2 = self.world_to_screen(*wps[i + 1])
                pygame.draw.line(self.screen, active_color, s1, s2, max(4, int(7 * self.scale / 2.5)))
                pygame.draw.line(self.screen, (255, 255, 255), s1, s2, max(1, int(2 * self.scale / 2.5)))

            for i, wp in enumerate(wps):
                sp = self.world_to_screen(*wp)
                if i == curr_idx:
                    pygame.draw.circle(self.screen, active_color, sp, 8)
                    pygame.draw.circle(self.screen, (255, 255, 255), sp, 10, 2)
                else:
                    pygame.draw.circle(self.screen, (255, 215, 0), sp, 3)

            # Draw Custom Along-Road Start Pin (if selected along road)
            s_info = getattr(self.controller, "start_point_info", None)
            if s_info and "proj_point" in s_info:
                s_pin_pos = self.world_to_screen(*s_info["proj_point"])
                pygame.draw.circle(self.screen, (0, 200, 255), s_pin_pos, 9)
                pygame.draw.circle(self.screen, (255, 255, 255), s_pin_pos, 11, 2)
                lbl_start = self.font_tiny.render("A (Điểm Đi)", True, (0, 230, 255))
                self.screen.blit(lbl_start, (s_pin_pos[0] + 12, s_pin_pos[1] - 8))
                # Direction arrow indicating chosen carriageway
                h = s_info.get("heading", 0.0)
                ax = s_pin_pos[0] + math.cos(h) * 16
                ay = s_pin_pos[1] - math.sin(h) * 16
                pygame.draw.line(self.screen, (0, 255, 255), s_pin_pos, (int(ax), int(ay)), 3)

            # Draw Custom Along-Road Goal Pin (if selected along road)
            g_info = getattr(self.controller, "goal_point_info", None)
            if g_info and "proj_point" in g_info:
                g_pin_pos = self.world_to_screen(*g_info["proj_point"])
                pygame.draw.circle(self.screen, (255, 60, 60), g_pin_pos, 9)
                pygame.draw.circle(self.screen, (255, 255, 255), g_pin_pos, 11, 2)
                lbl_goal = self.font_tiny.render("B (Đích Đến)", True, (255, 100, 100))
                self.screen.blit(lbl_goal, (g_pin_pos[0] + 12, g_pin_pos[1] - 8))
                # Direction arrow indicating required arrival flow
                h = g_info.get("heading", 0.0)
                ax = g_pin_pos[0] + math.cos(h) * 16
                ay = g_pin_pos[1] - math.sin(h) * 16
                pygame.draw.line(self.screen, (255, 80, 80), g_pin_pos, (int(ax), int(ay)), 3)

    def draw_vehicle(self, vehicle, is_ego: bool = False) -> None:
        eff_h = vehicle.heading + getattr(vehicle, "lane_change_yaw", 0.0)
        cos_h = math.cos(eff_h)
        sin_h = math.sin(eff_h)

        offset = getattr(vehicle, "current_lateral_offset", 0.0)
        rh = getattr(vehicle, "current_road_heading", vehicle.heading)
        actual_x = vehicle.x - math.sin(rh) * offset
        actual_y = vehicle.y + math.cos(rh) * offset

        dx_l = 0.5 * vehicle.length
        dy_w = 0.5 * vehicle.width
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
            world_corners.append(self.world_to_screen(wx, wy))

        shadow_sc = [(x + 2, y + 2) for x, y in world_corners]
        pygame.draw.polygon(self.screen, (10, 12, 16, 150), shadow_sc)

        body_col = self.EGO_BODY if is_ego else getattr(vehicle, "color", self.CAR_COLOR)
        pygame.draw.polygon(self.screen, body_col, world_corners)
        pygame.draw.polygon(self.screen, (255, 255, 255), world_corners, 1)

        if getattr(vehicle, "blinker_state", False) and vehicle.turn_signal != "NONE":
            if vehicle.turn_signal == "LEFT":
                pygame.draw.circle(self.screen, self.BLINKER_YELLOW, world_corners[0], 5)
                pygame.draw.circle(self.screen, self.BLINKER_YELLOW, world_corners[3], 5)
            elif vehicle.turn_signal == "RIGHT":
                pygame.draw.circle(self.screen, self.BLINKER_YELLOW, world_corners[1], 5)
                pygame.draw.circle(self.screen, self.BLINKER_YELLOW, world_corners[2], 5)

        nose_x = actual_x + cos_h * (vehicle.length * 0.4)
        nose_y = actual_y + sin_h * (vehicle.length * 0.4)
        s_nose = self.world_to_screen(nose_x, nose_y)
        pygame.draw.circle(self.screen, (255, 255, 200), s_nose, 2)

        if is_ego:
            angles = np.linspace(-math.pi, math.pi, 8, endpoint=False)
            for rel_a in angles:
                ra = vehicle.heading + rel_a
                rx = actual_x + math.cos(ra) * 20.0
                ry = actual_y + math.sin(ra) * 20.0
                sr = self.world_to_screen(rx, ry)
                pygame.draw.line(self.screen, (40, 160, 255, 80), s_nose, sr, 1)

        # Floating Vehicle Info Badge (Tốc độ km/h, trạng thái, làn đường)
        spd_kmh = vehicle.speed * 3.6
        c_x = sum(pt[0] for pt in world_corners) / 4.0
        c_y = min(pt[1] for pt in world_corners) - 18

        if is_ego:
            info_txt = f"{spd_kmh:.0f} km/h"
            t_surf = self.font_tiny.render(info_txt, True, (0, 255, 128))
            bw, bh = t_surf.get_width() + 10, t_surf.get_height() + 4
            badge_rect = pygame.Rect(c_x - bw // 2, c_y - bh, bw, bh)
            pygame.draw.rect(self.screen, (15, 25, 35, 230), badge_rect, border_radius=4)
            pygame.draw.rect(self.screen, (0, 230, 118), badge_rect, 1, border_radius=4)
            self.screen.blit(t_surf, (badge_rect.x + 5, badge_rect.y + 2))
        elif spd_kmh > 0.5:
            # NPC floating speed tag
            t_surf = self.font_tiny.render(f"{spd_kmh:.0f}", True, (240, 240, 240))
            bw, bh = t_surf.get_width() + 6, t_surf.get_height() + 2
            badge_rect = pygame.Rect(c_x - bw // 2, c_y - bh, bw, bh)
            pygame.draw.rect(self.screen, (20, 22, 28, 200), badge_rect, border_radius=3)
            self.screen.blit(t_surf, (badge_rect.x + 3, badge_rect.y + 1))

    def draw_multi_route_and_navigation_dashboard(self) -> None:
        # Mini collapsed toggle button when dashboard is hidden
        if not self.show_left_dashboard:
            btn_collapsed = pygame.Rect(20, 20, 150, 36)
            pygame.draw.rect(self.screen, (20, 30, 45, 230), btn_collapsed, border_radius=8)
            pygame.draw.rect(self.screen, (0, 230, 118), btn_collapsed, 1, border_radius=8)
            t_expand = self.font_tiny.render(">> MO BANG DIEU KHIEN", True, (0, 230, 118))
            self.screen.blit(t_expand, (btn_collapsed.x + 10, btn_collapsed.y + 11))
            self.btn_toggle_left_panel = btn_collapsed
            self.proposal_buttons = []
            return

        pw, ph = 430, 750
        surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        surf.fill(self.HUD_BG)
        self.screen.blit(surf, (20, 20))
        pygame.draw.rect(self.screen, self.HUD_BORDER, (20, 20, pw, ph), 2, border_radius=10)

        # Title: Autonomous Vehicle Navigation & Traffic Dispatch Center
        t_sub = self.font_tiny.render("AUTONOMOUS TRAFFIC SIMULATOR", True, (130, 170, 210))
        t = self.font_mid.render("DIEU HANH XE TU HANH & DINH TUYEN", True, (0, 230, 118))
        self.screen.blit(t_sub, (35, 28))
        self.screen.blit(t, (35, 46))

        # Close / Hide Dashboard Button (x)
        btn_hide_dash = pygame.Rect(20 + pw - 34, 26, 24, 24)
        pygame.draw.rect(self.screen, (40, 50, 70), btn_hide_dash, border_radius=5)
        pygame.draw.rect(self.screen, (70, 85, 115), btn_hide_dash, 1, border_radius=5)
        t_x = self.font_tiny.render("<<", True, (200, 220, 245))
        self.screen.blit(t_x, (btn_hide_dash.x + 5, btn_hide_dash.y + 4))
        self.btn_toggle_left_panel = btn_hide_dash

        # Start and Destination (Fix square missing-glyph tofu box by using standard ASCII '->')
        via = getattr(self.controller, "via_node", None)
        via_str = f" [Ghe: {via}] ->" if via else ""
        trip_txt = f"{self.controller.start_node} ->{via_str} {self.controller.goal_node}"
        t_trip = self.font_large.render(trip_txt, True, (255, 215, 0))
        self.screen.blit(t_trip, (35, 74))

        # Proposals
        t_prop_head = self.font_small.render("CAC LO TRINH DE XUAT (CLICK DE DOI TUYEN):", True, self.TEXT_MUTED)
        self.screen.blit(t_prop_head, (35, 102))

        self.proposal_buttons = []
        proposals = getattr(self.controller, "route_proposals", [])
        active_idx = getattr(self.controller, "active_proposal_index", 0)

        y_card = 124
        ego = self.env.ego_vehicle
        curr_speed_kmh = ego.speed * 3.6 if self.sim_state == "RUNNING" else 35.0

        for i, prop in enumerate(proposals[:3]):
            card_rect = pygame.Rect(35, y_card, 360, 42)
            is_selected = (i == active_idx)

            route_col = self.ROUTE_COLORS[i % len(self.ROUTE_COLORS)]
            bg_col = (25, 40, 55) if is_selected else (24, 28, 38)
            border_col = route_col if is_selected else (55, 65, 85)

            pygame.draw.rect(self.screen, bg_col, card_rect, border_radius=6)
            pygame.draw.rect(self.screen, border_col, card_rect, 2 if is_selected else 1, border_radius=6)

            p_len = prop.get("total_length", 0.0)
            p_eta_sec = int(p_len / max(1.0, curr_speed_kmh / 3.6))
            p_mins = p_eta_sec // 60
            p_secs = p_eta_sec % 60
            eta_txt = f"{p_mins}p {p_secs}s" if p_mins > 0 else f"{p_secs}s"

            tag = prop.get("tag", "TUYEN")
            tag_surf = self.font_tiny.render(f"[{tag}]", True, route_col)
            name_surf = self.font_small.render(f"{prop.get('name', '')[:28]}...", True, (255, 255, 255) if is_selected else (190, 205, 220))
            stat_surf = self.font_small.render(f"{p_len/1000.0:.2f} km | ETA: {eta_txt}", True, (255, 215, 0) if is_selected else (160, 175, 190))

            self.screen.blit(tag_surf, (card_rect.x + 8, card_rect.y + 6))
            self.screen.blit(name_surf, (card_rect.x + 82, card_rect.y + 6))
            self.screen.blit(stat_surf, (card_rect.x + 8, card_rect.y + 22))

            self.proposal_buttons.append((card_rect, i))
            y_card += 48

        # Main Action Button: START / PAUSE
        btn_action = pygame.Rect(35, y_card + 4, 360, 38)
        if self.sim_state in ["CONFIGURING", "CRASHED", "FINISHED"]:
            pygame.draw.rect(self.screen, (0, 190, 95), btn_action, border_radius=8)
            t_act = self.font_banner.render(">  BAT DAU CHAY", True, (255, 255, 255))
            self.screen.blit(t_act, (btn_action.x + 80, btn_action.y + 4))
        else:
            pygame.draw.rect(self.screen, (230, 130, 0), btn_action, border_radius=8)
            t_act = self.font_banner.render("||  TAM DUNG", True, (255, 255, 255))
            self.screen.blit(t_act, (btn_action.x + 105, btn_action.y + 4))

        # View & HUD Controls Row: [Theo Doi Xe] | [Mini-map: On/Off] | [Thong So: On/Off]
        # (Replaces old zoom in / zoom out buttons; zoom is comfortably done via mouse scroll wheel)
        y_ctrl = y_card + 48
        btn_cam_mode = pygame.Rect(35, y_ctrl, 140, 30)
        cam_col = (0, 150, 210) if self.follow_ego_camera else (180, 110, 30)
        pygame.draw.rect(self.screen, cam_col, btn_cam_mode, border_radius=6)
        t_cam = self.font_tiny.render("Cam: Bam Xe" if self.follow_ego_camera else "Cam: Tu Do", True, (255, 255, 255))
        self.screen.blit(t_cam, (btn_cam_mode.x + 18, btn_cam_mode.y + 7))

        # Mini-map Toggle Button
        btn_toggle_minimap = pygame.Rect(182, y_ctrl, 105, 30)
        mm_col = (0, 170, 130) if self.show_minimap else (50, 60, 75)
        pygame.draw.rect(self.screen, mm_col, btn_toggle_minimap, border_radius=6)
        t_mm = self.font_tiny.render(f"Map: {'BAT' if self.show_minimap else 'TAT'}", True, (255, 255, 255) if self.show_minimap else (170, 185, 205))
        self.screen.blit(t_mm, (btn_toggle_minimap.x + 16, btn_toggle_minimap.y + 7))

        # Telemetry HUD Toggle Button
        btn_toggle_telemetry = pygame.Rect(293, y_ctrl, 102, 30)
        telem_col = (140, 70, 210) if self.show_telemetry else (50, 60, 75)
        pygame.draw.rect(self.screen, telem_col, btn_toggle_telemetry, border_radius=6)
        t_tl = self.font_tiny.render(f"So Xe: {'BAT' if self.show_telemetry else 'TAT'}", True, (255, 255, 255) if self.show_telemetry else (170, 185, 205))
        self.screen.blit(t_tl, (btn_toggle_telemetry.x + 12, btn_toggle_telemetry.y + 7))

        # Turn Directions
        y_turns = y_ctrl + 36
        t_turn_head = self.font_mid.render("HUONG DAN TUNG BUOC:", True, (255, 215, 0))
        self.screen.blit(t_turn_head, (35, y_turns))

        inst_rect = pygame.Rect(35, y_turns + 22, 360, 88)
        pygame.draw.rect(self.screen, (22, 28, 38), inst_rect, border_radius=6)
        pygame.draw.rect(self.screen, (50, 65, 90), inst_rect, 1, border_radius=6)

        instructions = getattr(self.controller, "current_instructions", [])
        y_inst = y_turns + 26
        if instructions:
            for i, inst in enumerate(instructions[:3]):
                action = inst.get("action", "Di thang")
                to_name = inst.get("to_name", inst.get("to_node", ""))
                d_m = int(inst.get("distance_m", 0))
                line = f"{i+1}. {action} ({d_m}m) -> {to_name}"
                txt = self.font_tiny.render(line, True, (230, 240, 255))
                self.screen.blit(txt, (42, y_inst))
                y_inst += 20
        else:
            txt = self.font_small.render("Dang o ngay tai diem dich.", True, self.TEXT_MUTED)
            self.screen.blit(txt, (45, y_inst))

        # Bottom Route Modifier Buttons: [Diem Di] [Dich Den] [Them Tram]
        y_btns = y_turns + 116
        btn_start_pick = pygame.Rect(35, y_btns, 115, 30)
        pygame.draw.rect(self.screen, (0, 130, 220) if self.selection_mode == "SET_START" else (40, 50, 70), btn_start_pick, border_radius=6)
        t_p1 = self.font_tiny.render("Diem Di", True, (255, 255, 255))
        self.screen.blit(t_p1, (btn_start_pick.x + 30, btn_start_pick.y + 7))

        btn_goal_pick = pygame.Rect(158, y_btns, 115, 30)
        pygame.draw.rect(self.screen, (230, 110, 0) if self.selection_mode == "SET_GOAL" else (40, 50, 70), btn_goal_pick, border_radius=6)
        t_p2 = self.font_tiny.render("Dich Den", True, (255, 255, 255))
        self.screen.blit(t_p2, (btn_goal_pick.x + 28, btn_goal_pick.y + 7))

        btn_via_pick = pygame.Rect(280, y_btns, 115, 30)
        pygame.draw.rect(self.screen, (180, 100, 220) if self.selection_mode == "SET_VIA" else (40, 50, 70), btn_via_pick, border_radius=6)
        t_p3 = self.font_tiny.render("+ Them Tram", True, (255, 255, 255))
        self.screen.blit(t_p3, (btn_via_pick.x + 18, btn_via_pick.y + 7))

        # Auto-Retry & Condition Toggle Buttons
        y_retry = y_btns + 36
        btn_auto_retry = pygame.Rect(35, y_retry, 175, 30)
        auto_bg = (0, 170, 95) if self.auto_retry_until_success else (50, 60, 75)
        pygame.draw.rect(self.screen, auto_bg, btn_auto_retry, border_radius=6)
        t_ar_txt = f"Tu Thu Lai: {'BAT' if self.auto_retry_until_success else 'TAT'}"
        t_ar = self.font_tiny.render(t_ar_txt, True, (255, 255, 255))
        self.screen.blit(t_ar, (btn_auto_retry.x + 20, btn_auto_retry.y + 7))

        btn_traffic_cond = pygame.Rect(218, y_retry, 177, 30)
        cond_bg = (0, 135, 210) if self.deterministic_traffic else (90, 60, 120)
        pygame.draw.rect(self.screen, cond_bg, btn_traffic_cond, border_radius=6)
        t_tc_txt = f"DK: {'CO DINH' if self.deterministic_traffic else 'NGAU NHIEN'}"
        t_tc = self.font_tiny.render(t_tc_txt, True, (255, 255, 255))
        self.screen.blit(t_tc, (btn_traffic_cond.x + 20, btn_traffic_cond.y + 7))

        # Traffic Density Selector Row: [Mat do xe: IT XE / TRUNG BINH / NHIEU XE]
        y_density = y_retry + 34
        curr_mode = getattr(self.env, "traffic_density_mode", "MEDIUM")
        mode_label = "IT XE" if curr_mode == "LOW" else ("NHIEU XE" if curr_mode == "HIGH" else "TRUNG BINH")
        mode_color = (0, 190, 120) if curr_mode == "LOW" else ((240, 80, 80) if curr_mode == "HIGH" else (0, 160, 230))
        btn_traffic_density = pygame.Rect(35, y_density, 360, 30)
        pygame.draw.rect(self.screen, (30, 38, 52), btn_traffic_density, border_radius=6)
        pygame.draw.rect(self.screen, mode_color, btn_traffic_density, 1, border_radius=6)
        t_dens_txt = f"Mat do xe: {mode_label} (Click de doi: It / Vua / Nhieu)"
        t_dens = self.font_tiny.render(t_dens_txt, True, mode_color)
        self.screen.blit(t_dens, (btn_traffic_density.x + 18, btn_traffic_density.y + 7))

        # Auto-Run & Online RL Buttons
        y_train = y_density + 34
        btn_auto_train = pygame.Rect(35, y_train, 175, 30)
        train_bg = (170, 40, 170) if self.auto_training_mode else (45, 50, 65)
        train_border = (240, 120, 240) if self.auto_training_mode else (70, 80, 100)
        pygame.draw.rect(self.screen, train_bg, btn_auto_train, border_radius=6)
        pygame.draw.rect(self.screen, train_border, btn_auto_train, 2 if self.auto_training_mode else 1, border_radius=6)
        t_at_txt = f"AUTO-RUN: {'BAT' if self.auto_training_mode else 'TAT'}"
        t_at = self.font_tiny.render(t_at_txt, True, (255, 255, 255) if self.auto_training_mode else (180, 195, 215))
        self.screen.blit(t_at, (btn_auto_train.x + 22, btn_auto_train.y + 7))

        btn_rl_train = pygame.Rect(218, y_train, 177, 30)
        rl_bg = (220, 75, 0) if self.rl_training_enabled else (35, 45, 60)
        rl_border = (255, 160, 0) if self.rl_training_enabled else (60, 75, 95)
        pygame.draw.rect(self.screen, rl_bg, btn_rl_train, border_radius=6)
        pygame.draw.rect(self.screen, rl_border, btn_rl_train, 2 if self.rl_training_enabled else 1, border_radius=6)
        t_rl_txt = f"RL HOC ONLINE: {'BAT' if self.rl_training_enabled else 'TAT'}"
        t_rl = self.font_tiny.render(t_rl_txt, True, (255, 255, 255) if self.rl_training_enabled else (180, 195, 215))
        self.screen.blit(t_rl, (btn_rl_train.x + 18, btn_rl_train.y + 8))

        # Simulation Speed Multiplier Selector (Clean: 0.5x, 1x, 2x, 5x, 10x, 20x, 50x)
        # Note: 100x, 1000x, 5000x removed as requested!
        y_spd = y_train + 40
        t_spd_label = self.font_tiny.render("TOC DO MO PHONG (QUAN SAT / FAST-FORWARD):", True, (255, 215, 0))
        self.screen.blit(t_spd_label, (35, y_spd))

        self.speed_buttons = []
        # Row 1: 0.5x, 1x, 2x, 5x (width: 86 each)
        r1_w = 86
        spd_btn_h = 24
        x_spd_start = 35
        for s_idx, spd_val in enumerate(self.SPEED_OPTIONS[:4]):
            btn_s = pygame.Rect(x_spd_start + s_idx * (r1_w + 5), y_spd + 18, r1_w, spd_btn_h)
            is_active = (abs(self.sim_speed_multiplier - spd_val) < 0.01)
            b_bg = (0, 190, 110) if is_active else (38, 48, 64)
            b_border = (255, 255, 255) if is_active else (60, 75, 95)
            pygame.draw.rect(self.screen, b_bg, btn_s, border_radius=5)
            pygame.draw.rect(self.screen, b_border, btn_s, 2 if is_active else 1, border_radius=5)

            lbl = f"{spd_val:g}x"
            txt_s = self.font_tiny.render(lbl, True, (255, 255, 255) if is_active else (180, 195, 215))
            tx_off = (r1_w - txt_s.get_width()) // 2
            ty_off = (spd_btn_h - txt_s.get_height()) // 2
            self.screen.blit(txt_s, (btn_s.x + tx_off, btn_s.y + ty_off))
            self.speed_buttons.append((btn_s, spd_val))

        # Row 2: 10x, 20x, 50x (width: 116 each)
        r2_w = 116
        for s_idx, spd_val in enumerate(self.SPEED_OPTIONS[4:]):
            btn_s = pygame.Rect(x_spd_start + s_idx * (r2_w + 6), y_spd + 46, r2_w, spd_btn_h)
            is_active = (abs(self.sim_speed_multiplier - spd_val) < 0.01)
            b_bg = (210, 50, 70) if is_active else (55, 30, 45)
            b_border = (255, 120, 150) if is_active else (90, 50, 75)
            pygame.draw.rect(self.screen, b_bg, btn_s, border_radius=5)
            pygame.draw.rect(self.screen, b_border, btn_s, 2 if is_active else 1, border_radius=5)

            lbl = f"Tua: {spd_val:g}x"
            txt_s = self.font_tiny.render(lbl, True, (255, 255, 255) if is_active else (230, 180, 200))
            tx_off = (r2_w - txt_s.get_width()) // 2
            ty_off = (spd_btn_h - txt_s.get_height()) // 2
            self.screen.blit(txt_s, (btn_s.x + tx_off, btn_s.y + ty_off))
            self.speed_buttons.append((btn_s, spd_val))

        # Save Report & Chart Button
        y_save = y_spd + 78
        btn_save = pygame.Rect(35, y_save, 360, 32)
        pygame.draw.rect(self.screen, (0, 130, 200), btn_save, border_radius=6)
        pygame.draw.rect(self.screen, (0, 200, 255), btn_save, 1, border_radius=6)
        t_save = self.font_tiny.render("LUU KET QUA & VE BIEU DO (VAO RESULTS/)", True, (255, 255, 255))
        self.screen.blit(t_save, (btn_save.x + 45, btn_save.y + 8))

        self.btn_action = btn_action
        self.btn_cam_mode = btn_cam_mode
        self.btn_toggle_minimap = btn_toggle_minimap
        self.btn_toggle_telemetry = btn_toggle_telemetry
        self.btn_set_start = btn_start_pick
        self.btn_set_goal = btn_goal_pick
        self.btn_set_via = btn_via_pick
        self.btn_auto_retry = btn_auto_retry
        self.btn_traffic_cond = btn_traffic_cond
        self.btn_traffic_density = btn_traffic_density
        self.btn_auto_train = btn_auto_train
        self.btn_rl_train = btn_rl_train
        self.btn_save_report = btn_save


    def draw_interactive_minimap(self) -> None:
        """Draws minimap top-right with ALL candidate routes and live vehicle markers."""
        if not self.show_minimap:
            self.minimap_nodes = []
            return

        mw, mh = 280, 230
        mx = self.width - mw - 20
        my = 20

        surf = pygame.Surface((mw, mh), pygame.SRCALPHA)
        surf.fill(self.HUD_BG)
        self.screen.blit(surf, (mx, my))
        pygame.draw.rect(self.screen, self.HUD_BORDER, (mx, my, mw, mh), 2, border_radius=10)

        t = self.font_mid.render("GPS TRAFFIC MINIMAP", True, (0, 200, 255))
        self.screen.blit(t, (mx + 15, my + 10))

        G = self.controller.net.graph
        all_x = [d["pos"][0] for _, d in G.nodes(data=True)]
        all_y = [d["pos"][1] for _, d in G.nodes(data=True)]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        dx = max(1.0, max_x - min_x)
        dy = max(1.0, max_y - min_y)

        self.minimap_nodes = []

        def to_m(wx, wy):
            px = mx + 20 + int((wx - min_x) / dx * (mw - 40))
            py = my + mh - 20 - int((wy - min_y) / dy * (mh - 45))
            return px, py

        # 1. Base road network on minimap
        for u, v in G.edges():
            p1 = to_m(*G.nodes[u]["pos"])
            p2 = to_m(*G.nodes[v]["pos"])
            col = (255, 60, 60) if G[u][v].get("density", 0.0) > 0.65 else (70, 80, 95)
            pygame.draw.line(self.screen, col, p1, p2, 2)

        # 2. Draw ALL proposed routes on minimap!
        proposals = getattr(self.controller, "route_proposals", [])
        active_idx = getattr(self.controller, "active_proposal_index", 0)

        # Inactive routes on minimap
        for idx, prop in enumerate(proposals):
            if idx == active_idx:
                continue
            nodes = prop.get("nodes", [])
            col = self.ROUTE_COLORS[idx % len(self.ROUTE_COLORS)]
            muted = (int(col[0] * 0.5), int(col[1] * 0.5), int(col[2] * 0.5))
            for i in range(len(nodes) - 1):
                p1 = to_m(*G.nodes[nodes[i]]["pos"])
                p2 = to_m(*G.nodes[nodes[i + 1]]["pos"])
                pygame.draw.line(self.screen, muted, p1, p2, 2)

        # Active selected route highlighted with vivid neon glow on minimap
        if proposals:
            active_nodes = proposals[active_idx].get("nodes", [])
            active_col = self.ROUTE_COLORS[active_idx % len(self.ROUTE_COLORS)]
            for i in range(len(active_nodes) - 1):
                p1 = to_m(*G.nodes[active_nodes[i]]["pos"])
                p2 = to_m(*G.nodes[active_nodes[i + 1]]["pos"])
                # Outer glow line
                pygame.draw.line(self.screen, active_col, p1, p2, 5)
                # Inner bright white core
                pygame.draw.line(self.screen, (255, 255, 255), p1, p2, 2)

        # 3. Nodes on minimap with live traffic light indicators
        tl_dict = getattr(self.controller.net, "traffic_lights", {})
        for node_id, data in G.nodes(data=True):
            p = to_m(*data["pos"])
            if node_id == self.controller.start_node:
                pygame.draw.circle(self.screen, (0, 220, 255), p, 6)
            elif node_id == self.controller.goal_node:
                pygame.draw.circle(self.screen, (255, 80, 80), p, 6)
            elif node_id == getattr(self.controller, "via_node", None):
                pygame.draw.circle(self.screen, (255, 167, 38), p, 6)
            elif node_id in tl_dict:
                # Live traffic light color indicator on minimap node
                st = tl_dict[node_id].get("state", "green")
                col_light = (255, 50, 50) if st == "red" else ((255, 215, 0) if st == "yellow" else (0, 230, 118))
                pygame.draw.circle(self.screen, col_light, p, 5)
                pygame.draw.circle(self.screen, (255, 255, 255), p, 6, 1)
            else:
                pygame.draw.circle(self.screen, (160, 180, 200), p, 3)
            self.minimap_nodes.append({"id": node_id, "pos": p})

        # Ego vehicle position on minimap
        ego_m = to_m(self.env.ego_vehicle.x, self.env.ego_vehicle.y)
        pygame.draw.circle(self.screen, (0, 255, 120), ego_m, 5)

    def draw_telemetry_hud(self) -> None:
        """
        Displays a sleek, high-tech Cockpit Telemetry & Vehicle Gauge HUD in the top-right:
        - Real-time Digital Speedometer (km/h) with dynamic speed gauge arc / bar
        - Max speed & Average speed
        - Current Lane & Lateral offset
        - Overtaking status / Turn signal indicator
        - Upcoming Traffic Light distance & state
        """
        if not self.show_telemetry:
            return

        tw, th = 280, 190
        tx = self.width - tw - 20
        # If minimap is visible, place telemetry HUD right under it (ty = 265); otherwise place it at top (ty = 20)
        ty = 265 if self.show_minimap else 20

        surf = pygame.Surface((tw, th), pygame.SRCALPHA)
        surf.fill(self.HUD_BG)
        self.screen.blit(surf, (tx, ty))
        pygame.draw.rect(self.screen, self.HUD_BORDER, (tx, ty, tw, th), 2, border_radius=10)

        ego = getattr(self.env, "ego_vehicle", None)
        if not ego:
            return

        spd_kmh = ego.speed * 3.6
        max_spd_kmh = ego.max_speed * 3.6

        # Header Title
        t_title = self.font_mid.render("THONG SO XE DANG CHAY", True, (0, 230, 118))
        self.screen.blit(t_title, (tx + 16, ty + 12))

        # Big Digital Speed Display
        t_spd = self.font_banner.render(f"{spd_kmh:.1f}", True, (255, 255, 255))
        t_unit = self.font_tiny.render("KM/H", True, (0, 230, 118))
        self.screen.blit(t_spd, (tx + 16, ty + 36))
        self.screen.blit(t_unit, (tx + 22 + t_spd.get_width(), ty + 46))

        # Speed Gauge Bar
        bar_x = tx + 16
        bar_y = ty + 70
        bar_w = tw - 32
        bar_h = 8
        pygame.draw.rect(self.screen, (40, 50, 65), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pct = min(1.0, max(0.0, spd_kmh / max_spd_kmh))
        # Color transition from cyan -> green -> yellow -> red
        fill_col = (0, 230, 118) if spd_kmh < 45 else ((255, 200, 0) if spd_kmh < 60 else (255, 75, 75))
        if pct > 0:
            pygame.draw.rect(self.screen, fill_col, (bar_x, bar_y, int(bar_w * pct), bar_h), border_radius=4)

        # Telemetry Details: Lane & Overtake State
        if ego.current_lateral_offset < -12.5:
            lane_name = "Lan 1 (Xe may)"
        elif ego.current_lateral_offset > -7.5:
            lane_name = "Lan 3 (Lan vuot)"
        else:
            lane_name = "Lan 2 (Lan giua)"

        # Overtake state badge
        st_color = (0, 230, 118)
        if ego.overtaking_state in ["SIGNALING_OUTER", "CHANGING_OUTER", "PASSING"]:
            st_color = (255, 170, 0)
        elif ego.overtaking_state in ["SIGNALING_RETURN", "RETURNING"]:
            st_color = (0, 190, 255)

        sig_txt = f"Xi-nhan {ego.turn_signal}" if ego.turn_signal != "NONE" else "Xi-nhan: Tat"
        
        y_row = ty + 88
        t_lane = self.font_tiny.render(f"- Vi tri: {lane_name}", True, (220, 230, 245))
        self.screen.blit(t_lane, (tx + 16, y_row))

        y_row += 20
        t_ov = self.font_tiny.render(f"- TT: {ego.overtaking_state} ({sig_txt})", True, st_color)
        self.screen.blit(t_ov, (tx + 16, y_row))

        # Traffic light ahead radar
        y_row += 20
        tl_dist, tl_state, _ = self.env.check_upcoming_traffic_light()
        if tl_dist < 45.0:
            tl_col = (255, 60, 60) if tl_state == "RED" else ((255, 215, 0) if tl_state == "YELLOW" else (0, 230, 118))
            tl_txt = f"- Den truoc: {tl_state} ({tl_dist:.1f}m)"
        else:
            tl_col = (160, 175, 195)
            tl_txt = "- Tin hieu den: Thong thoang"
        t_tl = self.font_tiny.render(tl_txt, True, tl_col)
        self.screen.blit(t_tl, (tx + 16, y_row))

        # Overall trip stats
        y_row += 20
        stats = self.env.stats
        max_rec = stats.get("max_speed_kmh", spd_kmh)
        t_rec = self.font_tiny.render(f"- Toc do dinh: {max_rec:.1f} km/h | Vuot: {stats.get('overtake_count', 0)}", True, (255, 215, 0))
        self.screen.blit(t_rec, (tx + 16, y_row))

    def draw_dialog_overlay(self) -> None:
        if self.sim_state not in ["CRASHED", "FINISHED"]:
            return

        dw, dh = 560, 280
        dx = (self.width - dw) // 2
        dy = (self.height - dh) // 2

        overlay = pygame.Surface((dw, dh), pygame.SRCALPHA)
        overlay.fill((20, 24, 34, 250))
        self.screen.blit(overlay, (dx, dy))

        border_col = (255, 60, 60) if self.sim_state == "CRASHED" else (0, 230, 118)
        pygame.draw.rect(self.screen, border_col, (dx, dy, dw, dh), 3, border_radius=12)

        if self.sim_state == "CRASHED":
            t_title = self.font_banner.render("⚠️ THÔNG BÁO VA CHẠM!", True, (255, 75, 75))
            self.screen.blit(t_title, (dx + 30, dy + 25))

            reason = self.last_crash_reason or "Va chạm phương tiện giao thông."
            t_r1 = self.font_mid.render("Nguyên nhân va chạm:", True, (255, 200, 0))
            t_r2 = self.font_mid.render(f"• {reason}", True, (240, 245, 255))
            self.screen.blit(t_r1, (dx + 30, dy + 75))
            self.screen.blit(t_r2, (dx + 30, dy + 105))

            t_retry = self.font_small.render(f"Số lần đã thử lại: {self.retry_count} | Chế độ: {'Cùng điều kiện' if self.deterministic_traffic else 'Môi trường ngẫu nhiên'}", True, self.TEXT_MUTED)
            self.screen.blit(t_retry, (dx + 30, dy + 145))

            # 2 action buttons on crash dialog:
            # Button 1: Thử lại 1 lần
            btn_retry = pygame.Rect(dx + 30, dy + 195, 235, 48)
            pygame.draw.rect(self.screen, (220, 50, 50), btn_retry, border_radius=8)
            t_btn = self.font_mid.render("THỬ LẠI (1 LẦN)", True, (255, 255, 255))
            self.screen.blit(t_btn, (btn_retry.x + 45, btn_retry.y + 14))
            self.btn_dialog_action = btn_retry

            # Button 2: Tự thử lại liên tục đến khi thành công
            btn_auto = pygame.Rect(dx + 280, dy + 195, 250, 48)
            pygame.draw.rect(self.screen, (0, 160, 230), btn_auto, border_radius=8)
            t_auto = self.font_mid.render("🔁 TỰ THỬ LẠI LIÊN TỤC", True, (255, 255, 255))
            self.screen.blit(t_auto, (btn_auto.x + 25, btn_auto.y + 14))
            self.btn_dialog_auto_retry = btn_auto

        elif self.sim_state == "FINISHED":
            t_title = self.font_banner.render("🎉 CHÚC MỪNG: ĐẾN ĐÍCH AN TOÀN!", True, (0, 230, 118))
            self.screen.blit(t_title, (dx + 30, dy + 25))

            stats = self.env.stats
            speeds = stats.get("speeds_recorded", [0.0])
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            max_speed = stats.get("max_speed_kmh", 0.0)
            overtakes = stats.get("overtake_count", 0)

            red_viols = stats.get("red_light_violations", 0)
            yellow_viols = stats.get("yellow_light_violations", 0)

            lines = [
                f"- Lo trinh hoan thanh: {self.controller.start_node} -> {self.controller.goal_node}",
                f"- Toc do toi da dat duoc: {max_speed:.1f} km/h",
                f"- Toc do trung binh toan chang: {avg_speed:.1f} km/h",
                f"- Tong so lan vuot xe truoc thanh cong: {overtakes} lan",
                f"- Vi pham tin hieu: {red_viols} lan den do | {yellow_viols} lan den vang",
                f"- So lan thu lai truoc khi hoan thanh: {self.retry_count} lan"
            ]

            y_pos = dy + 62
            for line in lines:
                tl = self.font_tiny.render(line, True, self.TEXT_MAIN)
                self.screen.blit(tl, (dx + 30, y_pos))
                y_pos += 22

            btn_next = pygame.Rect(dx + 30, dy + 205, 500, 46)
            pygame.draw.rect(self.screen, (0, 180, 100), btn_next, border_radius=8)
            t_btn = self.font_large.render("TIẾP TỤC CHỌN LỘ TRÌNH MỚI", True, (255, 255, 255))
            self.screen.blit(t_btn, (btn_next.x + 110, btn_next.y + 10))
            self.btn_dialog_action = btn_next

    def handle_click_event(self, mouse_pos: Tuple[int, int]) -> None:
        mx, my = mouse_pos

        # Check Dialog Action
        if self.sim_state in ["CRASHED", "FINISHED"] and hasattr(self, "btn_dialog_action"):
            if self.btn_dialog_action.collidepoint(mx, my):
                if self.sim_state == "CRASHED":
                    self.retry_count += 1
                    active_seed = self.traffic_seed if self.deterministic_traffic else None
                    self.env.reset(seed=active_seed)
                    self.sim_state = "RUNNING"
                elif self.sim_state == "FINISHED":
                    self.sim_state = "CONFIGURING"
                return

        # Check Dialog Auto-Retry Button
        if self.sim_state == "CRASHED" and hasattr(self, "btn_dialog_auto_retry"):
            if self.btn_dialog_auto_retry.collidepoint(mx, my):
                self.auto_retry_until_success = True
                self.retry_count += 1
                active_seed = self.traffic_seed if self.deterministic_traffic else None
                self.env.reset(seed=active_seed)
                self.sim_state = "RUNNING"
                print(f"[AUTO-RETRY] Kích hoạt tự động thử lại liên tục đến khi thành công! (Lần thử #{self.retry_count})")
                return

        # Check Toggle Auto-Retry Button on Dashboard
        if hasattr(self, "btn_auto_retry") and self.btn_auto_retry.collidepoint(mx, my):
            self.auto_retry_until_success = not self.auto_retry_until_success
            print(f"[AUTO-RETRY] Chế độ tự động thử lại liên tục: {'BẬT' if self.auto_retry_until_success else 'TẮT'}")
            return

        # Check Toggle Traffic Conditions Button on Dashboard
        if hasattr(self, "btn_traffic_cond") and self.btn_traffic_cond.collidepoint(mx, my):
            self.deterministic_traffic = not self.deterministic_traffic
            cond_str = "CỐ ĐỊNH (Cùng mật độ/vị trí xe)" if self.deterministic_traffic else "NGẪU NHIÊN (Mỗi lần thử xe khác nhau)"
            print(f"[TRAFFIC CONDITIONS] Đã đổi điều kiện xe cộ: {cond_str}")
            return

        # Check Toggle Traffic Density Mode (LOW -> MEDIUM -> HIGH -> LOW)
        if hasattr(self, "btn_traffic_density") and self.btn_traffic_density.collidepoint(mx, my):
            cycle = ["LOW", "MEDIUM", "HIGH"]
            curr = getattr(self.env, "traffic_density_mode", "MEDIUM")
            next_mode = cycle[(cycle.index(curr) + 1) % len(cycle)]
            self.env.traffic_density_mode = next_mode
            # Respawn NPCs with new density settings immediately
            active_seed = self.traffic_seed if self.deterministic_traffic else None
            self.env.reset(seed=active_seed)
            dens_vn = "ÍT XE" if next_mode == "LOW" else ("NHIỀU XE" if next_mode == "HIGH" else "TRUNG BÌNH")
            print(f"[TRAFFIC DENSITY] Đã chuyển mật độ phương tiện sang: {dens_vn} ({len(self.env.npcs)} xe trên tuyến)")
            return

        # Check Toggle Auto-Training Button on Dashboard
        if hasattr(self, "btn_auto_train") and self.btn_auto_train.collidepoint(mx, my):
            self.auto_training_mode = not self.auto_training_mode
            if self.auto_training_mode:
                self.sim_state = "RUNNING"
            print(f"[AUTO-TRAINING] Chế độ tự động chạy liên tục: {'BẬT (Tự đổi tuyến mới khi đến đích, tự thử lại khi lỗi)' if self.auto_training_mode else 'TẮT'}")
            return

        # Check Toggle RL Online Learning Button on Dashboard
        if hasattr(self, "btn_rl_train") and self.btn_rl_train.collidepoint(mx, my):
            self.rl_training_enabled = not self.rl_training_enabled
            status_str = "BẬT (Mạng nơ-ron PPO trực tiếp điều khiển & tự động lan truyền cập nhật trọng số!)" if self.rl_training_enabled else "TẮT (Chế độ mô phỏng thông thường)"
            print(f"\n🧠 [HỌC TĂNG CƯỜNG PPO ONLINE]: {status_str}")
            return


        # Check Click on Route Proposal Cards (Google Maps style)
        for card_rect, prop_idx in self.proposal_buttons:
            if card_rect.collidepoint(mx, my):
                self.controller.select_route_proposal(prop_idx)
                return

        # Check Click on Simulation Speed Buttons
        for s_rect, s_val in self.speed_buttons:
            if s_rect.collidepoint(mx, my):
                self.sim_speed_multiplier = s_val
                print(f"[SPEED] Đã chuyển tốc độ mô phỏng sang: {s_val:g}x")
                return

        # Check Click on Save Results & Charts Button
        if hasattr(self, "btn_save_report") and self.btn_save_report.collidepoint(mx, my):
            if hasattr(self, "on_save_report_callback") and callable(self.on_save_report_callback):
                self.on_save_report_callback()
            return

        # Check Toggle Left Control Dashboard Button
        if hasattr(self, "btn_toggle_left_panel") and self.btn_toggle_left_panel.collidepoint(mx, my):
            self.show_left_dashboard = not self.show_left_dashboard
            print(f"[DISPLAY] Bang dieu khien ben trai: {'HIEN' if self.show_left_dashboard else 'AN (Chi con Map & Xe)'}")
            return

        # Check Camera Follow Toggle Button
        if hasattr(self, "btn_cam_mode") and self.btn_cam_mode.collidepoint(mx, my):
            self.follow_ego_camera = not self.follow_ego_camera
            print(f"[CAMERA] Che do camera: {'Bam theo xe' if self.follow_ego_camera else 'Keo re tu do'}")
            return

        # Check Mini-map Toggle Button
        if hasattr(self, "btn_toggle_minimap") and self.btn_toggle_minimap.collidepoint(mx, my):
            self.show_minimap = not self.show_minimap
            print(f"[DISPLAY] Mini-map: {'BAT' if self.show_minimap else 'TAT'}")
            return

        # Check Telemetry HUD Toggle Button
        if hasattr(self, "btn_toggle_telemetry") and self.btn_toggle_telemetry.collidepoint(mx, my):
            self.show_telemetry = not self.show_telemetry
            print(f"[DISPLAY] Bang thong so xe: {'BAT' if self.show_telemetry else 'TAT'}")
            return

        # Check Action Button (START / PAUSE)
        if hasattr(self, "btn_action") and self.btn_action.collidepoint(mx, my):
            if self.sim_state == "RUNNING":
                self.sim_state = "CONFIGURING"
            else:
                self.sim_state = "RUNNING"
            return

        # Check Buttons
        if hasattr(self, "btn_set_start") and self.btn_set_start.collidepoint(mx, my):
            self.selection_mode = "SET_START"
            return
        if hasattr(self, "btn_set_goal") and self.btn_set_goal.collidepoint(mx, my):
            self.selection_mode = "SET_GOAL"
            return
        if hasattr(self, "btn_set_via") and self.btn_set_via.collidepoint(mx, my):
            self.selection_mode = "SET_VIA"
            return

        # Check Node Click on Map or Minimap
        clicked_node = None
        for n in self.interactive_nodes:
            if math.hypot(n["screen_pos"][0] - mx, n["screen_pos"][1] - my) <= n["radius"] + 15:
                clicked_node = n["id"]
                break

        if not clicked_node and hasattr(self, "minimap_nodes"):
            for mn in self.minimap_nodes:
                if math.hypot(mn["pos"][0] - mx, mn["pos"][1] - my) <= 12:
                    clicked_node = mn["id"]
                    break

        if clicked_node:
            if self.selection_mode == "SET_START":
                if clicked_node != self.controller.goal_node:
                    self.controller.set_route(clicked_node, self.controller.goal_node, self.controller.via_node)
                    self.sim_state = "CONFIGURING"
                self.selection_mode = None
            elif self.selection_mode == "SET_GOAL":
                if clicked_node != self.controller.start_node:
                    self.controller.set_route(self.controller.start_node, clicked_node, self.controller.via_node)
                    self.sim_state = "CONFIGURING"
                self.selection_mode = None
            elif self.selection_mode == "SET_VIA":
                self.controller.set_route(self.controller.start_node, self.controller.goal_node, clicked_node)
                self.sim_state = "CONFIGURING"
                self.selection_mode = None
            else:
                if clicked_node != self.controller.start_node:
                    self.controller.set_route(self.controller.start_node, clicked_node, None)
                    self.sim_state = "CONFIGURING"
            return

        # If not clicking a junction node, check if clicking ALONG A ROAD SEGMENT (2 chiều đường)
        wx, wy = self.screen_to_world(mx, my)
        road_pt_info = self.controller.router.find_nearest_road_edge_point(wx, wy, max_dist=30.0)

        if road_pt_info:
            print(f"[ROAD CLICK] Chọn điểm dọc tuyến: {road_pt_info['road_label']} (t={road_pt_info['t']:.2f}, chiều: {road_pt_info['u']} ➔ {road_pt_info['v']})")
            if self.selection_mode == "SET_START":
                # User clicked Start on a specific carriageway along this road
                if hasattr(self.controller, "goal_point_info") and self.controller.goal_point_info:
                    self.controller.set_point_route(road_pt_info, self.controller.goal_point_info, self.controller.via_node)
                else:
                    # Synthesize goal point info from current goal node
                    g_node = self.controller.goal_node
                    g_pos = self.controller.net.graph.nodes[g_node]["pos"]
                    fake_goal_pt = {
                        "u": g_node,
                        "v": g_node,
                        "t": 0.0,
                        "proj_point": g_pos,
                        "heading": 0.0,
                        "road_label": self.controller.net.graph.nodes[g_node].get("name", g_node)
                    }
                    self.controller.set_point_route(road_pt_info, fake_goal_pt, self.controller.via_node)
                self.sim_state = "CONFIGURING"
                self.selection_mode = None

            elif self.selection_mode == "SET_GOAL":
                # User clicked Goal on a specific carriageway along this road
                if hasattr(self.controller, "start_point_info") and self.controller.start_point_info:
                    self.controller.set_point_route(self.controller.start_point_info, road_pt_info, self.controller.via_node)
                else:
                    # Synthesize start point info from current ego or start node
                    s_node = self.controller.start_node
                    s_pos = self.controller.net.graph.nodes[s_node]["pos"]
                    fake_start_pt = {
                        "u": s_node,
                        "v": list(self.controller.net.graph.neighbors(s_node))[0] if list(self.controller.net.graph.neighbors(s_node)) else s_node,
                        "t": 0.0,
                        "proj_point": (self.env.ego_vehicle.x, self.env.ego_vehicle.y),
                        "heading": self.env.ego_vehicle.heading,
                        "road_label": self.controller.net.graph.nodes[s_node].get("name", s_node)
                    }
                    self.controller.set_point_route(fake_start_pt, road_pt_info, self.controller.via_node)
                self.sim_state = "CONFIGURING"
                self.selection_mode = None

            else:
                # Default click without active mode: set as new Goal along the road
                if hasattr(self.controller, "start_point_info") and self.controller.start_point_info:
                    self.controller.set_point_route(self.controller.start_point_info, road_pt_info, None)
                else:
                    s_node = self.controller.start_node
                    nbrs = list(self.controller.net.graph.neighbors(s_node))
                    v_node = nbrs[0] if nbrs else s_node
                    fake_start_pt = {
                        "u": s_node,
                        "v": v_node,
                        "t": 0.0,
                        "proj_point": (self.env.ego_vehicle.x, self.env.ego_vehicle.y),
                        "heading": self.env.ego_vehicle.heading,
                        "road_label": self.controller.net.graph.nodes[s_node].get("name", s_node)
                    }
                    self.controller.set_point_route(fake_start_pt, road_pt_info, None)
                self.sim_state = "CONFIGURING"

    def render_frame(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if hasattr(self, "on_close_callback") and callable(self.on_close_callback):
                    try:
                        self.on_close_callback()
                    except Exception as e:
                        print(f"[SHUTDOWN ERROR] {e}")
                pygame.quit()
                sys.exit(0)

            # 1. Mouse Dragging to Pan Map (Chuột trái kéo nền map, chuột phải hoặc chuột giữa)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if self.show_left_dashboard:
                    hud_panel = pygame.Rect(20, 20, 430, 750)
                else:
                    hud_panel = getattr(self, "btn_toggle_left_panel", pygame.Rect(20, 20, 150, 36))

                minimap_rect = pygame.Rect(self.width - 300, 20, 280, 230) if self.show_minimap else pygame.Rect(0, 0, 0, 0)
                telem_y = 265 if self.show_minimap else 20
                telem_rect = pygame.Rect(self.width - 300, telem_y, 280, 190) if self.show_telemetry else pygame.Rect(0, 0, 0, 0)
                dialog_rect = pygame.Rect((self.width - 560) // 2, (self.height - 280) // 2, 560, 280) if self.sim_state in ["CRASHED", "FINISHED"] else pygame.Rect(0, 0, 0, 0)
                is_on_hud = hud_panel.collidepoint(mx, my) or minimap_rect.collidepoint(mx, my) or telem_rect.collidepoint(mx, my) or dialog_rect.collidepoint(mx, my)

                if event.button == 1:
                    if is_on_hud:
                        self.handle_click_event(event.pos)
                    else:
                        # Nhấp vào nền bản đồ lớn: Bắt đầu kéo di chuyển map (Pan)
                        self.is_dragging_map = True
                        self.follow_ego_camera = False
                        self.drag_start_mouse = event.pos
                        self.drag_start_cam = (self.cam_x, self.cam_y)
                elif event.button in [2, 3]:  # Middle or Right click to drag anywhere
                    self.is_dragging_map = True
                    self.follow_ego_camera = False  # Switch to manual pan
                    self.drag_start_mouse = event.pos
                    self.drag_start_cam = (self.cam_x, self.cam_y)
                elif event.button == 4:  # Wheel Up -> Zoom In
                    self.scale = min(6.5, self.scale + 0.35)
                elif event.button == 5:  # Wheel Down -> Zoom Out
                    self.scale = max(0.8, self.scale - 0.35)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    # Nếu đang kéo di chuyển thì thả ra
                    if self.is_dragging_map:
                        self.is_dragging_map = False
                        # Nếu chỉ nhấp chuột nhẹ (không di chuyển chuột nhiều), coi như là click chọn nút/đích đến
                        dist_moved = math.hypot(event.pos[0] - self.drag_start_mouse[0], event.pos[1] - self.drag_start_mouse[1])
                        if dist_moved < 5:
                            self.handle_click_event(event.pos)
                elif event.button in [2, 3]:
                    self.is_dragging_map = False

            elif event.type == pygame.MOUSEMOTION:
                if self.is_dragging_map:
                    dx = event.pos[0] - self.drag_start_mouse[0]
                    dy = event.pos[1] - self.drag_start_mouse[1]
                    self.cam_x = self.drag_start_cam[0] - dx / self.scale
                    self.cam_y = self.drag_start_cam[1] + dy / self.scale

            # 2. Keyboard Hotkeys for Pan & Zoom
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                    self.scale = min(6.0, self.scale + 0.4)
                elif event.key in [pygame.K_MINUS]:
                    self.scale = max(1.2, self.scale - 0.4)
                elif event.key == pygame.K_c:
                    self.follow_ego_camera = not self.follow_ego_camera
                elif event.key == pygame.K_h:
                    self.show_left_dashboard = not self.show_left_dashboard
                    print(f"[DISPLAY] Bang dieu khien ben trai (Phim H): {'HIEN' if self.show_left_dashboard else 'AN'}")
                elif event.key == pygame.K_m:
                    self.show_minimap = not self.show_minimap
                    print(f"[DISPLAY] Mini-map (Phim M): {'BAT' if self.show_minimap else 'TAT'}")
                elif event.key == pygame.K_t:
                    self.show_telemetry = not self.show_telemetry
                    print(f"[DISPLAY] Bang thong so xe (Phim T): {'BAT' if self.show_telemetry else 'TAT'}")
                elif event.key == pygame.K_1:
                    self.sim_speed_multiplier = 0.5
                elif event.key == pygame.K_2:
                    self.sim_speed_multiplier = 0.75
                elif event.key == pygame.K_3:
                    self.sim_speed_multiplier = 1.0
                elif event.key == pygame.K_4:
                    self.sim_speed_multiplier = 2.0
                elif event.key == pygame.K_5:
                    self.sim_speed_multiplier = 5.0
                elif event.key == pygame.K_6:
                    self.sim_speed_multiplier = 10.0
                elif event.key == pygame.K_7:
                    self.sim_speed_multiplier = 20.0
                elif event.key == pygame.K_8:
                    self.sim_speed_multiplier = 50.0
                elif event.key == pygame.K_9:
                    self.sim_speed_multiplier = 100.0

        if hasattr(self.controller.net, "update_traffic_lights"):
            speed_factor = self.sim_speed_multiplier if self.sim_state == "RUNNING" else 1.0
            self.controller.net.update_traffic_lights(dt=(1.0 / 30.0) * speed_factor)

        ego = getattr(self.env, "ego_vehicle", None)

        # Smooth camera tracking (if in Follow Mode and not dragging)
        if ego and self.follow_ego_camera and not self.is_dragging_map:
            self.cam_x += (ego.x - self.cam_x) * 0.08
            self.cam_y += (ego.y - self.cam_y) * 0.08

        self.draw_environment_background()
        self.draw_multi_lane_roads()
        self.draw_traffic_signs()

        # Draw multiple candidate routes on map simultaneously
        self.draw_all_route_proposals_on_map()

        for npc in self.env.npcs:
            self.draw_vehicle(npc, is_ego=False)

        if ego is not None:
            self.draw_vehicle(ego, is_ego=True)

        self.draw_multi_route_and_navigation_dashboard()
        self.draw_interactive_minimap()
        self.draw_telemetry_hud()

        self.draw_dialog_overlay()

        pygame.display.flip()
        target_fps = int(30 * max(0.1, min(10.0, self.sim_speed_multiplier)))
        self.clock.tick(target_fps)
