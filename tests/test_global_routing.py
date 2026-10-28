"""
Unit and integration tests for Global Routing module (A*, cost function, graph manager).
"""

import pytest
import math
from src.global_routing.cost_function import calculate_edge_cost
from src.global_routing.graph_manager import RoadNetworkManager
from src.global_routing.router import GlobalRouter


def test_cost_function_monotonicity():
    """Cost must increase strictly as congestion density increases."""
    length = 100.0
    speed = 10.0
    cost_free = calculate_edge_cost(length, speed, density=0.0)
    cost_moderate = calculate_edge_cost(length, speed, density=0.4)
    cost_heavy = calculate_edge_cost(length, speed, density=0.85)

    assert cost_free == pytest.approx(10.0, rel=1e-2)
    assert cost_moderate > cost_free
    assert cost_heavy > cost_moderate


def test_grid_city_creation():
    """Verify grid city dimensions and connectivity."""
    manager = RoadNetworkManager.create_grid_city(rows=3, cols=3, block_size=50.0)
    graph = manager.graph
    
    assert graph.number_of_nodes() == 9
    # Each inner node has 4 edges, perimeter nodes 3 or 2
    assert graph.has_edge("N_0_0", "N_0_1")
    assert graph.has_edge("N_0_1", "N_0_0")
    assert manager.get_node_pos("N_2_2") == (100.0, 100.0)


def test_astar_detours_around_congestion():
    """
    Test scenario:
    Path 1: N_0_0 -> N_0_1 -> N_1_1 (Shortest Euclidean, but N_0_0 -> N_0_1 is heavily congested)
    Path 2: N_0_0 -> N_1_0 -> N_1_1 (Alternative path with zero congestion)
    A* must choose Path 2!
    """
    manager = RoadNetworkManager.create_grid_city(rows=2, cols=2, block_size=100.0)
    
    # Jam the east road N_0_0 -> N_0_1
    manager.update_density("N_0_0", "N_0_1", density=0.98)
    # North road is clear
    manager.update_density("N_0_0", "N_1_0", density=0.0)

    router = GlobalRouter(manager)
    res_astar = router.find_path_astar("N_0_0", "N_1_1")
    
    assert res_astar["nodes"][1] == "N_1_0", "A* should detour through N_1_0 to avoid heavy jam"


def test_waypoint_interpolation():
    """Check that coarse waypoints are properly densified."""
    coarse = [(0.0, 0.0), (0.0, 50.0)]
    dense = GlobalRouter.interpolate_waypoints(coarse, step_size=10.0)
    
    assert len(dense) == 6
    assert dense[0] == (0.0, 0.0)
    assert dense[-1] == (0.0, 50.0)


def test_along_road_point_selection():
    """Verify that arbitrary click along a road segment correctly identifies edge and carriageway direction."""
    manager = RoadNetworkManager.create_grid_city(rows=2, cols=2, block_size=100.0)
    router = GlobalRouter(manager)

    # Click along road N_0_0 -> N_0_1 at x=40, y=-10 (forward side)
    pt_fwd = router.find_nearest_road_edge_point(click_x=40.0, click_y=-10.0)
    assert pt_fwd is not None
    assert pt_fwd["u"] == "N_0_0"
    assert pt_fwd["v"] == "N_0_1"
    assert 0.35 <= pt_fwd["t"] <= 0.45

    # Click on the opposite carriageway side at x=40, y=+10 (where N_0_1 -> N_0_0 is forward)
    pt_rev = router.find_nearest_road_edge_point(click_x=40.0, click_y=10.0)
    assert pt_rev is not None
    assert pt_rev["u"] == "N_0_1"
    assert pt_rev["v"] == "N_0_0"


def test_uturn_at_intersection_for_opposite_lane():
    """
    Verify that when start is on carriageway A -> B and destination is on opposite carriageway B -> A,
    vehicle does NOT reverse or cross the solid median, but travels forward to intersection B,
    executes a U-turn loop, and proceeds along B -> A to destination.
    """
    manager = RoadNetworkManager.create_grid_city(rows=2, cols=2, block_size=100.0)
    router = GlobalRouter(manager)

    # Start: halfway along N_0_0 -> N_0_1 (x=30, y=-10)
    start_pt = router.find_nearest_road_edge_point(click_x=30.0, click_y=-10.0)
    # Goal: on opposite carriageway N_0_1 -> N_0_0 (x=20, y=+10)
    goal_pt = router.find_nearest_road_edge_point(click_x=20.0, click_y=10.0)

    assert start_pt["u"] == "N_0_0" and start_pt["v"] == "N_0_1"
    assert goal_pt["u"] == "N_0_1" and goal_pt["v"] == "N_0_0"

    proposals = router.find_all_route_proposals_for_points(start_pt, goal_pt)
    assert len(proposals) >= 1
    selected = proposals[0]

    # Path must traverse to intersection N_0_1 and loop back to N_0_0
    assert selected["nodes"] == ["N_0_0", "N_0_1", "N_0_0"]
    assert selected["tag"] == "QUAY ĐẦU"

    # Waypoints must exist and form a continuous trajectory through junction N_0_1
    wps = selected["waypoints"]
    assert len(wps) >= 8
    # Turn instructions must indicate U-turn at intersection
    instructions = selected["instructions"]
    assert any("U-Turn" in inst["action"] or "quay đầu" in inst["action"].lower() for inst in instructions)

