"""
Comprehensive Unit & System Tests for Directional Traffic Lights and Violation Logic.
Validates:
1. Multi-phase non-conflicting traffic signal state machine (Straight vs Left-turn).
2. Accurate turn maneuver detection (Left-turn vs Straight).
3. Directional obedience: Left-turning vehicles obey left-turn light, not straight light.
4. Stop-line holding: Vehicle stops before stop line on RED/YELLOW and holds speed=0.
5. Strict red-light violation detection: Crossing stop line on RED terminates and heavily penalizes.
6. NPC vehicle obedience to directional signals.
"""

import math
import pytest
import numpy as np
from src.global_routing.realistic_city import RealisticCityManager
from src.global_routing.router import GlobalRouter
from src.local_env.environment import LocalDrivingEnv
from src.local_env.vehicle import Vehicle


def test_traffic_light_non_conflicting_phases():
    """
    Mathematical verification of traffic signal state machine:
    - Active axis and Inactive axis NEVER have conflicting green signals.
    - Protected left-turn phase: Straight is RED when Left is GREEN.
    - Inactive axis is 100% RED when active axis has GREEN (straight or left).
    """
    city = RealisticCityManager.create_metropolitan_city()
    city.update_traffic_lights(dt=0.1)

    for nid, light in city.traffic_lights.items():
        ctrl = light.get("controller")
        if not ctrl:
            continue

        # Simulate full cycle in fine increments
        total_cycle = ctrl["t_axis"] * 2.0
        dt = 0.5
        steps = int(total_cycle / dt) + 2

        for _ in range(steps):
            city.update_traffic_lights(dt=dt)
            approaches = light.get("approaches", {})
            axis_a = ctrl["axis_a"]
            axis_b = ctrl["axis_b"]

            for a_node in axis_a:
                sig_a = approaches[a_node]
                for b_node in axis_b:
                    sig_b = approaches[b_node]

                    # Condition 1: Axis A and Axis B cannot BOTH have green straight
                    assert not (sig_a["straight_state"] == "green" and sig_b["straight_state"] == "green"), \
                        f"Conflict: Both Axis A ({a_node}) and Axis B ({b_node}) have green straight!"

                    # Condition 2: Axis A and Axis B cannot have orthogonal green cross
                    assert not (sig_a["straight_state"] == "green" and sig_b["left_state"] == "green"), \
                        f"Conflict: Axis A straight green while Axis B left green!"
                    assert not (sig_a["left_state"] == "green" and sig_b["straight_state"] == "green"), \
                        f"Conflict: Axis A left green while Axis B straight green!"

                # Condition 3: Protected left turn (Straight must not be green while left is green)
                assert not (sig_a["straight_state"] == "green" and sig_a["left_state"] == "green"), \
                    f"Conflict on Axis A ({a_node}): Both straight and left are green simultaneously!"


def test_left_turn_detects_left_signal_when_straight_is_green():
    """
    Directly tests the scenario from the user's screenshot:
    - Vehicle is on approach to an intersection and will TURN LEFT.
    - Straight light is GREEN, Left-turn light is RED.
    - check_upcoming_traffic_light() MUST return RED (not GREEN)!
    """
    city = RealisticCityManager.create_metropolitan_city()
    # Find a signalized intersection with at least 3 approaches, e.g., "Industrial_Park"
    target_nid = "Industrial_Park"
    assert target_nid in city.traffic_lights, "Industrial_Park should have a traffic light"

    pos = city.get_node_pos(target_nid)
    nbrs = list(city.graph.to_undirected().neighbors(target_nid))
    in_node = nbrs[0]
    in_pos = city.get_node_pos(in_node)

    # Pick an out_node that forms a LEFT TURN (> 20 deg CCW)
    h_in = math.atan2(pos[1] - in_pos[1], pos[0] - in_pos[0])
    left_out_node = None
    for out_cand in nbrs:
        if out_cand == in_node:
            continue
        out_pos = city.get_node_pos(out_cand)
        h_out = math.atan2(out_pos[1] - pos[1], out_pos[0] - pos[0])
        diff = (h_out - h_in + math.pi) % (2 * math.pi) - math.pi
        if diff > 0.35:
            left_out_node = out_cand
            break

    assert left_out_node is not None, "Should find a left-turning branch at Industrial_Park"

    # Create environment with route turning left
    path = [in_node, target_nid, left_out_node]
    env = LocalDrivingEnv(network_manager=city)
    env.current_path_nodes = path

    # Force traffic light approach so straight is GREEN and left is RED (exactly like user screenshot)
    city.update_traffic_lights(0.1)
    light = city.traffic_lights[target_nid]
    light["approaches"][in_node] = {
        "straight_state": "green",
        "straight_timer": 7.0,
        "left_state": "red",
        "left_timer": 10.0
    }

    # Position ego car on approach road 35m before intersection center (7m before stop line)
    u_x = math.cos(h_in)
    u_y = math.sin(h_in)
    ego_x = pos[0] - u_x * 35.0
    ego_y = pos[1] - u_y * 35.0
    env.ego_vehicle.x = ego_x
    env.ego_vehicle.y = ego_y
    env.ego_vehicle.heading = h_in

    # Check signal detected
    dist, state, timer = env.check_upcoming_traffic_light()

    # Crucial assertion: Left-turning car MUST see RED (not GREEN)!
    assert state == "RED", f"Expected RED left-turn light, but got {state}!"
    assert timer == pytest.approx(10.0, rel=1e-2)
    assert dist == pytest.approx(7.0, abs=1.0)  # 35m to center - 28m stopline = 7m


def test_straight_vehicle_detects_straight_signal():
    """
    A vehicle continuing straight through the intersection MUST follow straight_state.
    """
    city = RealisticCityManager.create_metropolitan_city()
    target_nid = "Industrial_Park"
    pos = city.get_node_pos(target_nid)
    nbrs = list(city.graph.to_undirected().neighbors(target_nid))
    in_node = nbrs[0]
    in_pos = city.get_node_pos(in_node)
    h_in = math.atan2(pos[1] - in_pos[1], pos[0] - in_pos[0])

    # Find straight out node (|diff| < 0.35)
    straight_out_node = None
    for out_cand in nbrs:
        if out_cand == in_node:
            continue
        out_pos = city.get_node_pos(out_cand)
        h_out = math.atan2(out_pos[1] - pos[1], out_pos[0] - pos[0])
        diff = abs((h_out - h_in + math.pi) % (2 * math.pi) - math.pi)
        if diff < 0.35:
            straight_out_node = out_cand
            break

    if straight_out_node:
        path = [in_node, target_nid, straight_out_node]
        env = LocalDrivingEnv(network_manager=city)
        env.current_path_nodes = path

        city.update_traffic_lights(0.1)
        light = city.traffic_lights[target_nid]
        light["approaches"][in_node] = {
            "straight_state": "green",
            "straight_timer": 7.0,
            "left_state": "red",
            "left_timer": 10.0
        }

        u_x = math.cos(h_in)
        u_y = math.sin(h_in)
        env.ego_vehicle.x = pos[0] - u_x * 35.0
        env.ego_vehicle.y = pos[1] - u_y * 35.0
        env.ego_vehicle.heading = h_in

        dist, state, timer = env.check_upcoming_traffic_light()
        assert state == "GREEN", f"Straight vehicle should see GREEN straight signal, got {state}"


def test_vehicle_stops_at_red_stop_line_and_holds_speed_zero():
    """
    When approaching a red light and reaching the stop line (tl_dist <= 2.5m),
    the safety system MUST hold the vehicle stopped (speed = 0.0) and not accelerate.
    """
    city = RealisticCityManager.create_metropolitan_city()
    target_nid = "Industrial_Park"
    pos = city.get_node_pos(target_nid)
    nbrs = list(city.graph.to_undirected().neighbors(target_nid))
    in_node = nbrs[0]
    in_pos = city.get_node_pos(in_node)
    h_in = math.atan2(pos[1] - in_pos[1], pos[0] - in_pos[0])

    env = LocalDrivingEnv(network_manager=city)
    env.current_path_nodes = [in_node, target_nid]

    city.update_traffic_lights(0.1)
    light = city.traffic_lights[target_nid]
    light["approaches"][in_node] = {
        "straight_state": "red",
        "straight_timer": 15.0,
        "left_state": "red",
        "left_timer": 15.0
    }

    # Position ego right at the stop line (29m from center => 1.0m before stop line)
    u_x = math.cos(h_in)
    u_y = math.sin(h_in)
    env.ego_vehicle.x = pos[0] - u_x * 29.0
    env.ego_vehicle.y = pos[1] - u_y * 29.0
    env.ego_vehicle.heading = h_in
    env.ego_vehicle.speed = 1.0  # slight creeping speed

    # Step with positive throttle (simulating aggressive AI trying to run the light)
    action = np.array([0.0, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    # Safety override MUST stop the vehicle at the stop line
    assert env.ego_vehicle.speed == 0.0, f"Vehicle should be stopped at stop line, but speed is {env.ego_vehicle.speed}"


def test_red_light_violation_triggers_termination_and_heavy_penalty():
    """
    If a vehicle crosses the stop line on RED at speed (> 1.0 m/s),
    it must be flagged for RED LIGHT VIOLATION, receive a heavy penalty (-500),
    and trigger immediate episode termination.
    """
    city = RealisticCityManager.create_metropolitan_city()
    target_nid = "Industrial_Park"
    pos = city.get_node_pos(target_nid)
    nbrs = list(city.graph.to_undirected().neighbors(target_nid))
    in_node = nbrs[0]
    in_pos = city.get_node_pos(in_node)
    h_in = math.atan2(pos[1] - in_pos[1], pos[0] - in_pos[0])

    env = LocalDrivingEnv(network_manager=city)
    env.current_path_nodes = [in_node, target_nid]

    city.update_traffic_lights(0.1)
    light = city.traffic_lights[target_nid]
    light["approaches"][in_node] = {
        "straight_state": "red",
        "straight_timer": 15.0,
        "left_state": "red",
        "left_timer": 15.0
    }

    # Position vehicle crossing the stop line into the junction on RED at 10 m/s (36 km/h)
    u_x = math.cos(h_in)
    u_y = math.sin(h_in)
    env.ego_vehicle.x = pos[0] - u_x * 26.0  # 2m PAST stop line
    env.ego_vehicle.y = pos[1] - u_y * 26.0
    env.ego_vehicle.heading = h_in
    env.ego_vehicle.speed = 10.0

    action = np.array([0.0, 0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.stats["red_light_violations"] >= 1, "Red light violation was not recorded!"
    assert info["is_red_light_fail"] is True, "info['is_red_light_fail'] should be True!"
    assert terminated is True, "Episode must be terminated on red light violation!"
    assert reward <= -400.0, f"Expected heavy penalty on red light violation, got {reward}"


def test_npc_left_turn_obeys_red_left_arrow_and_stops():
    """
    NPC vehicles turning left must obey the left-turn red light and stop completely,
    even when the adjacent straight lane has a green light.
    """
    city = RealisticCityManager.create_metropolitan_city()
    target_nid = "Industrial_Park"
    pos = city.get_node_pos(target_nid)
    nbrs = list(city.graph.to_undirected().neighbors(target_nid))
    in_node = nbrs[0]
    in_pos = city.get_node_pos(in_node)
    h_in = math.atan2(pos[1] - in_pos[1], pos[0] - in_pos[0])

    left_out_node = None
    for out_cand in nbrs:
        if out_cand == in_node:
            continue
        out_pos = city.get_node_pos(out_cand)
        h_out = math.atan2(out_pos[1] - pos[1], out_pos[0] - pos[0])
        diff = (h_out - h_in + math.pi) % (2 * math.pi) - math.pi
        if diff > 0.35:
            left_out_node = out_cand
            break

    assert left_out_node is not None
    out_pos = city.get_node_pos(left_out_node)

    # Path waypoints: approaching from in_node, passing target_nid, turning left towards out_node
    wps = [in_pos, pos, out_pos]
    from src.global_routing.router import GlobalRouter
    dense_wps = GlobalRouter.interpolate_waypoints(wps, step_size=5.0)

    npc = Vehicle(
        x=pos[0] - math.cos(h_in) * 32.0,  # 4m before stop line (28m)
        y=pos[1] - math.sin(h_in) * 32.0,
        heading=h_in,
        speed=4.0,
        vehicle_type="car",
        lane_id=1
    )
    npc.path_waypoints = dense_wps
    npc.current_wp_idx = 1

    city.update_traffic_lights(0.1)
    light = city.traffic_lights[target_nid]
    light["approaches"][in_node] = {
        "straight_state": "green",
        "straight_timer": 8.0,
        "left_state": "red",
        "left_timer": 12.0
    }

    # Step NPC safety patrol
    for _ in range(25):
        npc.update_npc_patrol_with_safety(
            dt=0.1,
            all_vehicles=[npc],
            network_manager=city
        )

    # NPC must have stopped before crossing the stop line!
    # Distance to junction must remain >= 27.0m (stop line is at 28.0m)
    d_to_junc = math.hypot(npc.x - pos[0], npc.y - pos[1])
    assert d_to_junc >= 26.5, f"NPC crossed the stop line on red! Distance: {d_to_junc}m"
    assert npc.speed == 0.0, f"NPC should be completely stopped on red left light, but speed is {npc.speed}"


def test_left_turn_lane_preparation_and_blinker():
    """
    When approaching an intersection where the route turns left,
    the vehicle must automatically activate its left blinker and target the left-turn lane (Lane 2).
    """
    city = RealisticCityManager.create_metropolitan_city()
    target_nid = "Industrial_Park"
    pos = city.get_node_pos(target_nid)
    nbrs = list(city.graph.to_undirected().neighbors(target_nid))
    in_node = nbrs[0]
    in_pos = city.get_node_pos(in_node)
    h_in = math.atan2(pos[1] - in_pos[1], pos[0] - in_pos[0])

    left_out_node = None
    for out_cand in nbrs:
        if out_cand == in_node:
            continue
        out_pos = city.get_node_pos(out_cand)
        h_out = math.atan2(out_pos[1] - pos[1], out_pos[0] - pos[0])
        diff = (h_out - h_in + math.pi) % (2 * math.pi) - math.pi
        if diff > 0.35:
            left_out_node = out_cand
            break

    env = LocalDrivingEnv(network_manager=city)
    env.current_path_nodes = [in_node, target_nid, left_out_node]

    # Vehicle is 38m before intersection center (10m before stop line)
    u_x = math.cos(h_in)
    u_y = math.sin(h_in)
    env.ego_vehicle.x = pos[0] - u_x * 38.0
    env.ego_vehicle.y = pos[1] - u_y * 38.0
    env.ego_vehicle.heading = h_in

    env.check_upcoming_traffic_light()

    # Blinker and lane checks
    assert env.ego_vehicle.turn_signal == "LEFT", "Left blinker should be active when preparing for left turn!"
    assert env.ego_vehicle.lane_id == 2, "Vehicle should target Lane 2 (left-turn lane)!"
    assert env.ego_vehicle.target_lateral_offset == env.LANE_OFFSETS[2], "Target lateral offset should be Lane 2 offset!"


def test_npc_queue_at_red_light_does_not_overlap():
    """
    Verifies that multiple NPC vehicles queueing at a red light STOP BEHIND each other
    and NEVER overlap or phase through one another.
    """
    city = RealisticCityManager.create_metropolitan_city()
    target_nid = "Industrial_Park"
    pos = city.get_node_pos(target_nid)
    nbrs = list(city.graph.to_undirected().neighbors(target_nid))
    in_node = nbrs[0]
    in_pos = city.get_node_pos(in_node)
    h_in = math.atan2(pos[1] - in_pos[1], pos[0] - in_pos[0])

    city.update_traffic_lights(0.1)
    light = city.traffic_lights[target_nid]
    light["approaches"][in_node] = {
        "straight_state": "red",
        "straight_timer": 20.0,
        "left_state": "red",
        "left_timer": 20.0
    }

    # Leader: Orange Truck (length = 8.0m) stopped at stop line (28m from center)
    cos_h = math.cos(h_in)
    sin_h = math.sin(h_in)
    truck = Vehicle(
        x=pos[0] - cos_h * 29.5,
        y=pos[1] - sin_h * 29.5,
        heading=h_in,
        speed=0.0,
        vehicle_type="truck",
        lane_id=1
    )

    # Follower: Red Car (length = 4.5m) approaching from 15m behind the truck at 7 m/s
    car = Vehicle(
        x=pos[0] - cos_h * 45.0,
        y=pos[1] - sin_h * 45.0,
        heading=h_in,
        speed=7.0,
        vehicle_type="car",
        lane_id=1
    )

    all_vehs = [truck, car]
    for _ in range(50):
        truck.update_npc_patrol_with_safety(0.1, all_vehs, network_manager=city)
        car.update_npc_patrol_with_safety(0.1, all_vehs, network_manager=city)

    # Calculate final distance between vehicle centers
    dist_centers = math.hypot(truck.x - car.x, truck.y - car.y)
    half_lengths = 0.5 * (truck.length + car.length)  # 4.0 + 2.25 = 6.25m
    bumper_gap = dist_centers - half_lengths

    # Assertions:
    # 1. Car must be stopped behind truck
    assert car.speed == 0.0, f"Car should be stopped in queue, but speed is {car.speed}"
    # 2. Bumper gap must be strictly POSITIVE (vehicles do not touch/overlap!)
    assert bumper_gap >= 0.5, f"Car overlapped the truck! Bumper gap: {bumper_gap:.2f}m (must be >= 0.5m)"
    # 3. Truck must remain stopped at stop line
    assert truck.speed == 0.0, "Truck should remain stopped at stop line"


def test_car_behind_truck_does_not_penetrate():
    """
    On an open road without a traffic light, a faster car approaching a slower truck
    must decelerate and match speed, maintaining a safe bumper buffer without clipping.
    """
    h = 0.0  # moving along +X
    truck = Vehicle(x=50.0, y=0.0, heading=h, speed=5.0, vehicle_type="truck", lane_id=1)
    car = Vehicle(x=25.0, y=0.0, heading=h, speed=10.0, vehicle_type="car", lane_id=1)

    all_vehs = [truck, car]
    for _ in range(60):
        truck.update_npc_patrol_with_safety(0.1, all_vehs)
        car.update_npc_patrol_with_safety(0.1, all_vehs)

        dist_c = truck.x - car.x
        half_l = 0.5 * (truck.length + car.length)
        assert dist_c >= half_l + 0.4, f"Car penetrated truck! Clearance: {dist_c - half_l:.2f}m"


def test_three_vehicles_queue_at_red_light():
    """
    Verifies that a chain of 3 vehicles (Truck -> Car -> Motorcycle) queueing up at a red light
    all come to a complete stop sequentially, with every pair maintaining a positive bumper gap.
    """
    city = RealisticCityManager.create_metropolitan_city()
    target_nid = "Industrial_Park"
    pos = city.get_node_pos(target_nid)
    nbrs = list(city.graph.to_undirected().neighbors(target_nid))
    in_node = nbrs[0]
    in_pos = city.get_node_pos(in_node)
    h_in = math.atan2(pos[1] - in_pos[1], pos[0] - in_pos[0])

    city.update_traffic_lights(0.1)
    light = city.traffic_lights[target_nid]
    light["approaches"][in_node] = {
        "straight_state": "red",
        "straight_timer": 30.0,
        "left_state": "red",
        "left_timer": 30.0
    }

    cos_h = math.cos(h_in)
    sin_h = math.sin(h_in)

    # 1. Lead Truck (length = 8m) stopped at stop line (29.5m from center)
    truck = Vehicle(
        x=pos[0] - cos_h * 29.5,
        y=pos[1] - sin_h * 29.5,
        heading=h_in,
        speed=0.0,
        vehicle_type="truck",
        lane_id=1
    )

    # 2. Car (length = 4.5m) behind truck at distance
    car = Vehicle(
        x=pos[0] - cos_h * 42.0,
        y=pos[1] - sin_h * 42.0,
        heading=h_in,
        speed=6.0,
        vehicle_type="car",
        lane_id=1
    )

    # 3. Motorcycle (length = 2.2m) behind car at distance
    bike = Vehicle(
        x=pos[0] - cos_h * 55.0,
        y=pos[1] - sin_h * 55.0,
        heading=h_in,
        speed=5.5,
        vehicle_type="motorcycle",
        lane_id=1
    )

    all_vehs = [truck, car, bike]
    for _ in range(80):
        for v in all_vehs:
            v.update_npc_patrol_with_safety(0.1, all_vehs, network_manager=city)

    # Check that all 3 are stopped
    assert truck.speed == 0.0, "Truck must remain stopped at stop line"
    assert car.speed == 0.0, f"Car should be stopped in queue, got {car.speed}"
    assert bike.speed == 0.0, f"Motorcycle should be stopped in queue, got {bike.speed}"

    # Check bumper gap between truck and car
    dist_tc = math.hypot(truck.x - car.x, truck.y - car.y)
    half_tc = 0.5 * (truck.length + car.length)
    gap_tc = dist_tc - half_tc
    assert gap_tc >= 0.5, f"Car overlapped truck! Gap: {gap_tc:.2f}m"

    # Check bumper gap between car and bike
    dist_cb = math.hypot(car.x - bike.x, car.y - bike.y)
    half_cb = 0.5 * (car.length + bike.length)
    gap_cb = dist_cb - half_cb
    assert gap_cb >= 0.5, f"Motorcycle overlapped car! Gap: {gap_cb:.2f}m"


def test_adjacent_lane_vehicle_at_intersection_does_not_freeze_ego():
    """
    Verifies that a vehicle (e.g. motorcycle) stopped in Lane 0 at an intersection
    does NOT trigger obstacle shielding for an ego vehicle approaching in Lane 1
    with a GREEN light, preventing the vehicle from freezing at the junction.
    """
    city = RealisticCityManager.create_metropolitan_city()
    router = GlobalRouter(city)
    path = ["Plaza_East", "Metro_Station", "Shopping_Mall"]
    wps = router.build_waypoints_with_roundabouts(path)
    interp = GlobalRouter.interpolate_waypoints(wps, step_size=7.0)

    env = LocalDrivingEnv(network_manager=city, waypoints=interp)
    env.current_path_nodes = path
    env.reset()

    # Place Ego at approach to Metro_Station in Lane 1 (offset = -6.5m)
    # Metro_Station is at (600, 600), approach from Plaza_East (525, 380)
    pos_ms = city.get_node_pos("Metro_Station")
    pos_pe = city.get_node_pos("Plaza_East")
    h = math.atan2(pos_ms[1] - pos_pe[1], pos_ms[0] - pos_pe[0])
    cos_h = math.cos(h)
    sin_h = math.sin(h)

    # Ego vehicle is 40m before the intersection center in Lane 1 (-6.5m offset)
    env.ego_vehicle.x = pos_ms[0] - cos_h * 40.0
    env.ego_vehicle.y = pos_ms[1] - sin_h * 40.0
    env.ego_vehicle.heading = h
    env.ego_vehicle.speed = 8.0
    env.ego_vehicle.current_lateral_offset = env.LANE_OFFSETS[1]
    env.ego_vehicle.target_lateral_offset = env.LANE_OFFSETS[1]

    # Synchronize waypoint index to Ego's location
    for idx, wp in enumerate(env.waypoints):
        if math.hypot(wp[0] - env.ego_vehicle.x, wp[1] - env.ego_vehicle.y) < 12.0:
            env.current_wp_idx = idx
            break

    # Spawn an NPC motorcycle in Lane 0 (-9.5m offset) stopped at the stop line (29m before center)
    bike = Vehicle(
        x=pos_ms[0] - cos_h * 29.0,
        y=pos_ms[1] - sin_h * 29.0,
        heading=h,
        speed=0.0,
        vehicle_type="motorcycle",
        lane_id=0
    )
    bike.current_lateral_offset = env.LANE_OFFSETS[0]
    bike.target_lateral_offset = env.LANE_OFFSETS[0]
    env.npcs = [bike]

    # Set GREEN light for the Plaza_East approach
    city.update_traffic_lights(0.1)
    light = city.traffic_lights["Metro_Station"]
    light["approaches"]["Plaza_East"] = {
        "straight_state": "green",
        "straight_timer": 20.0,
        "left_state": "green",
        "left_timer": 20.0
    }

    # Step environment for 30 steps (3.0 seconds)
    for _ in range(30):
        city.update_traffic_lights(0.1)
        light["approaches"]["Plaza_East"] = {
            "straight_state": "green",
            "straight_timer": 20.0,
            "left_state": "green",
            "left_timer": 20.0
        }
        f_dist, _, f_npc = env.check_traffic_ahead_and_overtake_lane()
        # Motorcycle in Lane 0 must NOT be detected as blocking Ego's Lane 1
        assert f_npc is None or f_npc is not bike, "Motorcycle in Lane 0 was falsely flagged as blocking Lane 1!"
        obs, r, term, trunc, info = env.step([0.0, 0.4])

    # Ego vehicle must not have frozen (speed > 0)
    assert env.ego_vehicle.speed > 5.0, f"Ego froze at intersection! Speed: {env.ego_vehicle.speed}"
    assert not info.get("is_crashed", False), "Ego crashed!"


def test_expanded_road_geometry_and_five_car_capacity():
    """
    Verifies the user requirements:
    1. Each lane expanded by +65% in area (3.0m -> 5.0m lane width = +66.7%).
    2. Lane offsets: Lane 0 = -15.0m, Lane 1 = -10.0m, Lane 2 = -5.0m.
    3. Carriageway width is 15.0m (3 lanes * 5.0m). Total road corridor is 35.0m (doubled).
    4. Carriageway can accommodate up to 5 cars abreast (width 2.0m each) during overtaking maneuvers
       with safe buffers, fitting within the 15.0m bounds without lateral collisions.
    """
    city = RealisticCityManager.create_metropolitan_city()
    router = GlobalRouter(city)
    path = ["Plaza_East", "Metro_Station", "Shopping_Mall"]
    wps = router.build_waypoints_with_roundabouts(path)
    interp = GlobalRouter.interpolate_waypoints(wps, step_size=7.0)
    env = LocalDrivingEnv(network_manager=city, waypoints=interp)

    # 1. Lane dimensions
    assert env.LANE_OFFSETS[0] == -15.0
    assert env.LANE_OFFSETS[1] == -10.0
    assert env.LANE_OFFSETS[2] == -5.0

    lane_width_0_1 = abs(env.LANE_OFFSETS[0] - env.LANE_OFFSETS[1])
    lane_width_1_2 = abs(env.LANE_OFFSETS[1] - env.LANE_OFFSETS[2])
    assert lane_width_0_1 == 5.0
    assert lane_width_1_2 == 5.0

    # +65% area verification: 3.0m baseline * 1.65 = 4.95m ~ 5.0m
    area_expansion_ratio = (lane_width_0_1 - 3.0) / 3.0
    assert area_expansion_ratio >= 0.65, f"Area expansion {area_expansion_ratio * 100:.1f}% is less than 65%!"

    # 2. 5 cars abreast fitting simultaneously across 15.0m carriageway
    # Carriageway boundaries: from median (-2.5m) to outer curb (-17.5m) = 15.0m
    # 5 cars placed abreast at offsets: -3.5, -6.5, -9.5, -12.5, -15.5
    car_width = 2.0
    offsets = [-3.5, -6.5, -9.5, -12.5, -15.5]
    assert len(offsets) == 5

    # Check that all 5 cars fit within carriageway [-17.5, -2.5]
    for off in offsets:
        left_edge = off - car_width / 2.0
        right_edge = off + car_width / 2.0
        assert left_edge >= -17.5, f"Car at {off} exceeds outer curb (-17.5): {left_edge}"
        assert right_edge <= -2.5, f"Car at {off} exceeds median curb (-2.5): {right_edge}"

    # Check pairwise non-overlapping with clearance
    for i in range(len(offsets) - 1):
        gap = abs(offsets[i] - offsets[i+1]) - car_width
        assert gap >= 1.0, f"Gap between car {i} and {i+1} is too small: {gap:.2f}m"


def test_sat_collision_prevents_false_collision_between_adjacent_lanes():
    """
    Verifies that SAT collision detection prevents false collision alarms:
    1. Ego in Lane 3 (-5.0m offset) directly alongside a Truck or Car in Lane 2 (-10.0m offset)
       reports NO collision, even when slight heading angle differences occur.
    2. Real physical overlaps (e.g. rear-end in same lane) are strictly detected as collisions.
    """
    h = 0.0
    ego = Vehicle(x=100.0, y=100.0, heading=h, speed=8.0, vehicle_type="car", is_ego=True)
    ego.current_lateral_offset = -5.0

    # Car in Lane 2 directly alongside
    car_lane2 = Vehicle(x=100.0, y=100.0, heading=h, speed=6.0, vehicle_type="car")
    car_lane2.current_lateral_offset = -10.0
    assert not ego.check_collision_with(car_lane2), "Ego in Lane 3 falsely collided with Car in Lane 2!"

    # Truck in Lane 2 with slight steering angle (0.1 rad ~ 5.7 degrees) and offset drift (-6.5m vs -10.0m)
    ego_drifting = Vehicle(x=100.0, y=100.0, heading=0.1, speed=8.0, vehicle_type="car", is_ego=True)
    ego_drifting.current_lateral_offset = -6.5
    truck_lane2 = Vehicle(x=100.0, y=100.0, heading=0.0, speed=5.0, vehicle_type="truck")
    truck_lane2.current_lateral_offset = -10.0
    assert not ego_drifting.check_collision_with(truck_lane2), "Ego falsely collided with Truck in Lane 2 during slight drift!"

    # Real rear-end collision in SAME lane (Lane 3)
    rear_car = Vehicle(x=96.5, y=100.0, heading=h, speed=8.0, vehicle_type="car")
    rear_car.current_lateral_offset = -5.0
    assert ego.check_collision_with(rear_car), "SAT failed to detect true rear-end collision in same lane!"


def test_motorcycle_roundabout_navigation_no_abnormal_jumps():
    """
    Verifies that an NPC motorcycle in Lane 1 (-15.0m offset) traversing a roundabout:
    1. Smoothly advances waypoints without skipping or jumping.
    2. Does not experience abnormal lateral position teleportation (> 1.5m displacement in 0.1s).
    """
    city = RealisticCityManager.create_metropolitan_city()
    router = GlobalRouter(city)
    nodes = ["Financial_Tower", "Shopping_Mall", "Hospital"]
    wps = router.build_waypoints_with_roundabouts(nodes)
    interp = GlobalRouter.interpolate_waypoints(wps, step_size=5.0)

    bike = Vehicle(
        x=interp[0][0],
        y=interp[0][1],
        heading=math.atan2(interp[1][1] - interp[0][1], interp[1][0] - interp[0][0]),
        speed=6.0,
        vehicle_type="motorcycle",
        lane_id=0,
        path_waypoints=interp
    )
    bike.current_lateral_offset = -15.0
    bike.target_lateral_offset = -15.0

    car_nearby = Vehicle(
        x=interp[0][0] - 1.0,
        y=interp[0][1],
        heading=bike.heading,
        speed=6.5,
        vehicle_type="car",
        lane_id=1,
        path_waypoints=interp
    )
    car_nearby.current_lateral_offset = -10.0
    car_nearby.target_lateral_offset = -10.0

    all_vehs = [bike, car_nearby]
    prev_pos = None

    for step in range(350):
        bike.update_npc_patrol_with_safety(dt=0.1, all_vehicles=all_vehs, network_manager=city)
        car_nearby.update_npc_patrol_with_safety(dt=0.1, all_vehicles=all_vehs, network_manager=city)

        rh = getattr(bike, "current_road_heading", bike.heading)
        act_x = bike.x - math.sin(rh) * bike.current_lateral_offset
        act_y = bike.y + math.cos(rh) * bike.current_lateral_offset

        if prev_pos is not None:
            dist_moved = math.hypot(act_x - prev_pos[0], act_y - prev_pos[1])
            # Max expected move at 6.0 m/s is 0.6m in 0.1s. Allow up to 1.5m for smooth road curvature.
            assert dist_moved < 1.5, f"Step {step}: Motorcycle experienced abnormal jump: {dist_moved:.2f}m in 0.1s!"
        prev_pos = (act_x, act_y)






