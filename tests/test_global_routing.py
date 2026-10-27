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
