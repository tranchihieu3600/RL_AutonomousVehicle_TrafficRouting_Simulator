"""
Realistic City Map Generator.
Generates an organic city layout with:
- Multiple intersections (3-way, 4-way, roundabouts, diagonals)
- Diverse road types (highways, downtown streets, boulevard)
- Speed limits (30 km/h, 50 km/h, 80 km/h)
- Traffic signs: Speed limit, Stop signs, Traffic lights, One-way/No entry
"""

from typing import Dict, Tuple, Optional, List, Any
import math
import random
import networkx as nx
from src.global_routing.cost_function import calculate_edge_cost


class RealisticCityManager:
    """Manages an enriched road network with realistic layout, signage, and traffic dynamics."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.traffic_lights: Dict[str, Dict[str, Any]] = {}
        self.traffic_signs: List[Dict[str, Any]] = []

    def add_intersection(
        self,
        node_id: str,
        x: float,
        y: float,
        name: str = "",
        has_traffic_light: bool = False,
        is_roundabout: bool = False,
        light_cycle: float = 30.0
    ) -> None:
        """Add intersection with coordinates and attributes."""
        self.graph.add_node(
            node_id,
            pos=(float(x), float(y)),
            name=name or node_id,
            has_light=has_traffic_light,
            is_roundabout=is_roundabout
        )
        if has_traffic_light:
            # Ngã 3: chu kỳ ngắn ~20s; Ngã 4: chu kỳ ~30s
            init_timer = random.uniform(light_cycle * 0.8, light_cycle * 1.1)
            self.traffic_lights[node_id] = {
                "state": "green",  # green, yellow, red
                "timer": init_timer,
                "cycle": float(light_cycle)
            }

    def add_road(
        self,
        u: str,
        v: str,
        road_type: str = "avenue",  # highway, avenue, residential
        max_speed: float = 13.89,   # 50 km/h
        lanes: int = 3,             # 3 làn đường mỗi chiều
        density: float = 0.0,
        bidirectional: bool = True,
        is_prohibited: bool = False  # Biển cấm
    ) -> None:
        """Add road segment with realistic road types and signs."""
        pos_u = self.graph.nodes[u]["pos"]
        pos_v = self.graph.nodes[v]["pos"]
        length = math.hypot(pos_v[0] - pos_u[0], pos_v[1] - pos_u[1])

        # If prohibited, set infinite cost
        cost = 1e9 if is_prohibited else calculate_edge_cost(length, max_speed, density)

        edge_data = {
            "length": float(length),
            "max_speed": float(max_speed),
            "lanes": int(lanes),
            "density": float(density),
            "cost": float(cost),
            "road_type": road_type,
            "is_prohibited": is_prohibited,
            "bidirectional": bidirectional
        }

        self.graph.add_edge(u, v, **edge_data)
        if bidirectional:
            edge_data_rev = dict(edge_data)
            self.graph.add_edge(v, u, **edge_data_rev)

        # Place realistic signs along the roadside (offset to sidewalk curbside)
        mid_x = (pos_u[0] + pos_v[0]) * 0.5
        mid_y = (pos_u[1] + pos_v[1]) * 0.5
        dx = pos_v[0] - pos_u[0]
        dy = pos_v[1] - pos_u[1]
        mag = max(1e-3, math.hypot(dx, dy))
        # Perpendicular normal pointing to right curbside / sidewalk
        curb_offset = 12.0  # meters away from road center to sidewalk curb
        curb_x = mid_x - (dy / mag) * curb_offset
        curb_y = mid_y + (dx / mag) * curb_offset

        speed_kmh = int(round(max_speed * 3.6))

        if is_prohibited:
            self.traffic_signs.append({
                "type": "NO_ENTRY",
                "pos": (curb_x, curb_y),
                "label": "CẤM",
                "road": (u, v)
            })
        else:
            self.traffic_signs.append({
                "type": "SPEED_LIMIT",
                "pos": (curb_x, curb_y),
                "label": f"{speed_kmh}",
                "road": (u, v)
            })
            if not bidirectional:
                # Sign indicating strict 1-way street (Đường 1 chiều)
                self.traffic_signs.append({
                    "type": "ONE_WAY",
                    "pos": (curb_x + (dx / mag) * 6.0, curb_y + (dy / mag) * 6.0),
                    "label": "1 CHIỀU",
                    "direction": math.atan2(dy, dx),
                    "road": (u, v)
                })

    def update_density(self, u: str, v: str, density: float) -> None:
        """Update traffic density and recalculate cost."""
        if self.graph.has_edge(u, v):
            data = self.graph[u][v]
            data["density"] = max(0.0, min(1.0, float(density)))
            if data.get("is_prohibited", False):
                data["cost"] = 1e9
            else:
                data["cost"] = calculate_edge_cost(data["length"], data["max_speed"], data["density"])

    def update_traffic_lights(self, dt: float = 0.1) -> None:
        """Step traffic light state timers with realistic cycles: Ngã 3 (~20s) và Ngã 4 (~30s)."""
        for nid, light in self.traffic_lights.items():
            light["timer"] -= dt
            cycle = light.get("cycle", 30.0)
            if light["timer"] <= 0:
                if light["state"] == "green":
                    light["state"] = "yellow"
                    light["timer"] = 3.5 if cycle <= 22.0 else 4.0
                elif light["state"] == "yellow":
                    light["state"] = "red"
                    # Thời gian đèn đỏ: ~18-20s đối với ngã 3, ~28-32s đối với ngã 4
                    light["timer"] = random.uniform(cycle * 0.9, cycle * 1.05)
                else:
                    light["state"] = "green"
                    # Thời gian đèn xanh: ~20s đối với ngã 3, ~30s đối với ngã 4
                    light["timer"] = random.uniform(cycle * 0.95, cycle * 1.15)

    def get_node_pos(self, node_id: str) -> Tuple[float, float]:
        return self.graph.nodes[node_id]["pos"]

    @classmethod
    def create_metropolitan_city(cls) -> "RealisticCityManager":
        """
        Create a rich, realistic metropolitan city:
        - Central Ring Road / Roundabout
        - High-speed outer highway bypass (80 km/h)
        - Downtown arterial avenues (50 km/h)
        - Residential districts with cul-de-sacs (30 km/h)
        - Dedicated traffic lights and signs (Speed limit, Cấm rẽ)
        """
        city = cls()

        # 1. Central Core & Roundabout (Plaza)
        cx, cy = 450.0, 380.0
        r_dist = 75.0
        # 4 plaza nodes - Vòng xuyến trung tâm không đặt đèn giao thông (lưu thông tự do liên tục)
        city.add_intersection("Plaza_North", cx, cy + r_dist, "Quảng Trường Bắc", has_traffic_light=False)
        city.add_intersection("Plaza_East", cx + r_dist, cy, "Quảng Trường Đông", has_traffic_light=False)
        city.add_intersection("Plaza_South", cx, cy - r_dist, "Quảng Trường Nam", has_traffic_light=False)
        city.add_intersection("Plaza_West", cx - r_dist, cy, "Quảng Trường Tây", has_traffic_light=False)

        # Connect central roundabout (Vòng xuyến 3 làn đường chuẩn lưu thông rộng thoáng)
        city.add_road("Plaza_North", "Plaza_West", road_type="avenue", max_speed=11.11, lanes=3, bidirectional=False)
        city.add_road("Plaza_West", "Plaza_South", road_type="avenue", max_speed=11.11, lanes=3, bidirectional=False)
        city.add_road("Plaza_South", "Plaza_East", road_type="avenue", max_speed=11.11, lanes=3, bidirectional=False)
        city.add_road("Plaza_East", "Plaza_North", road_type="avenue", max_speed=11.11, lanes=3, bidirectional=False)

        # 2. Western Hubs (Harbor & Logistics)
        city.add_intersection("Port_Harbor", 50.0, 150.0, "Cảng Biển Tân Cảng")
        # Ngã 3 Logistics: thêm đèn đỏ chu kỳ 20s
        city.add_intersection("Logistics_Center", 50.0, 380.0, "Trung Tâm Logistics", has_traffic_light=True, light_cycle=20.0)
        city.add_intersection("Industrial_Park", 200.0, 380.0, "Khu Công Nghiệp Tân Bình", has_traffic_light=True, light_cycle=30.0)
        city.add_intersection("Airport_Terminal", 50.0, 620.0, "Sân Bay Quốc Tế")

        # 3. Northern High-Tech & Educational Zone
        # Ngã 3 Tech Park North: thêm đèn đỏ chu kỳ 20s
        city.add_intersection("Tech_Park_North", 200.0, 720.0, "Khu Công Nghệ Cao 2", has_traffic_light=True, light_cycle=20.0)
        city.add_intersection("Tech_Hub", 300.0, 600.0, "Công Viên Phần Mềm Quang Trung", has_traffic_light=True, light_cycle=30.0)
        # Ngã 3 University: thêm đèn đỏ chu kỳ 20s
        city.add_intersection("University", 450.0, 600.0, "Làng Đại Học Quốc Gia", has_traffic_light=True, light_cycle=20.0)
        city.add_intersection("Research_Institute", 600.0, 720.0, "Viện Nghiên Cứu AI")
        city.add_intersection("Metro_Station", 600.0, 600.0, "Ga Metro Trung Tâm", has_traffic_light=True, light_cycle=30.0)

        # 4. Eastern Business & Commercial Hub
        city.add_intersection("Financial_Tower", 780.0, 600.0, "Tháp Tài Chính Landmark", has_traffic_light=True, light_cycle=30.0)
        # Nút giao hơn 4 ngã rẽ (5 ngã): Quy hoạch thành VÒNG XUYẾN LỚN (Roundabout 3 làn xe)
        city.add_intersection("Shopping_Mall", 780.0, 380.0, "Vòng Xuyến MegaMall", is_roundabout=True)
        # Ngã 3 Hospital: thêm đèn đỏ chu kỳ 20s
        city.add_intersection("Hospital", 780.0, 150.0, "Bệnh Viện Đa Khoa Quốc Tế", has_traffic_light=True, light_cycle=20.0)
        city.add_intersection("New_Urban_East", 950.0, 380.0, "Khu Đô Thị Mới Đông TP")
        city.add_intersection("Science_Park", 950.0, 600.0, "Công Viên Khoa Học Đổi Mới")

        # 5. Southern Residential & Recreation District
        # Ngã 3 Old Town: thêm đèn đỏ chu kỳ 20s
        city.add_intersection("Old_Town", 200.0, 150.0, "Phố Cổ Trung Tâm", has_traffic_light=True, light_cycle=20.0)
        # Nút giao hơn 4 ngã rẽ (5 ngã): Quy hoạch thành VÒNG XUYẾN SINH THÁI (Roundabout 3 làn xe)
        city.add_intersection("Residential_Eco", 450.0, 150.0, "Vòng Xuyến Ecopark", is_roundabout=True)
        # Ngã 3 Sports Stadium: thêm đèn đỏ chu kỳ 20s
        city.add_intersection("Sports_Stadium", 620.0, 150.0, "Sân Vận Động TP", has_traffic_light=True, light_cycle=20.0)
        city.add_intersection("River_Resort", 350.0, 20.0, "Khu Nghỉ Dưỡng Ven Sông")
        city.add_intersection("Golf_Club", 620.0, 20.0, "Sân Golf & Biệt Thự Nam")

        # 6. CONNECT ROADS (Đường 2 chiều, mỗi chiều 3 làn đường: lanes=3)

        # Western Corridor & Logistics
        city.add_road("Port_Harbor", "Logistics_Center", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Logistics_Center", "Industrial_Park", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        # Giới hạn đường cao tốc nội đô tối đa 60 km/h (16.67 m/s)
        city.add_road("Logistics_Center", "Airport_Terminal", road_type="highway", max_speed=16.67, lanes=3, bidirectional=True)
        city.add_road("Airport_Terminal", "Tech_Park_North", road_type="highway", max_speed=16.67, lanes=3, bidirectional=True)
        city.add_road("Tech_Park_North", "Tech_Hub", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)

        # Northern Expressway & Arterials
        city.add_road("Tech_Hub", "University", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("University", "Metro_Station", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Tech_Park_North", "Research_Institute", road_type="highway", max_speed=16.67, lanes=3, bidirectional=True)
        city.add_road("Research_Institute", "Financial_Tower", road_type="highway", max_speed=16.67, lanes=3, bidirectional=True)
        city.add_road("Metro_Station", "Financial_Tower", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)

        # Eastern Corridor & Outer Extensions
        city.add_road("Financial_Tower", "Shopping_Mall", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Shopping_Mall", "Hospital", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Financial_Tower", "Science_Park", road_type="highway", max_speed=16.67, lanes=3, bidirectional=True)
        city.add_road("Science_Park", "New_Urban_East", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Shopping_Mall", "New_Urban_East", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)

        # Southern Corridor & Waterfront
        city.add_road("Port_Harbor", "Old_Town", road_type="residential", max_speed=11.11, lanes=3, bidirectional=True)
        city.add_road("Old_Town", "Residential_Eco", road_type="residential", max_speed=11.11, lanes=3, bidirectional=True)
        city.add_road("Residential_Eco", "Sports_Stadium", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Sports_Stadium", "Hospital", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Residential_Eco", "River_Resort", road_type="residential", max_speed=11.11, lanes=3, bidirectional=True)
        city.add_road("River_Resort", "Golf_Club", road_type="residential", max_speed=11.11, lanes=3, bidirectional=True)
        city.add_road("Sports_Stadium", "Golf_Club", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)

        # Radiate connections to Central Plaza
        city.add_road("Plaza_West", "Industrial_Park", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Plaza_North", "University", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Plaza_East", "Shopping_Mall", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Plaza_South", "Residential_Eco", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)

        # Strategic Crossings & Diagonal Shortcuts
        city.add_road("Industrial_Park", "Old_Town", road_type="residential", max_speed=11.11, lanes=3, bidirectional=True)
        city.add_road("Tech_Hub", "Industrial_Park", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Tech_Hub", "Plaza_West", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Plaza_East", "Metro_Station", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        city.add_road("Metro_Station", "Shopping_Mall", road_type="avenue", max_speed=13.89, lanes=3, bidirectional=True)
        # Biển Cấm: đoạn từ Hospital sang Residential_Eco có giờ cao điểm hạn chế
        city.add_road("Hospital", "Residential_Eco", road_type="residential", max_speed=8.33, lanes=3, bidirectional=False, is_prohibited=True)

        # Dynamic congestion spots
        city.update_density("Plaza_East", "Shopping_Mall", 0.85)
        city.update_density("Shopping_Mall", "Plaza_East", 0.85)
        city.update_density("Plaza_North", "University", 0.72)
        city.update_density("University", "Plaza_North", 0.72)

        return city
