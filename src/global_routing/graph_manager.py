"""
Graph manager for global road network representation using NetworkX.
Supports grid city network creation, traffic density simulation, and serialization.
"""

from typing import Dict, Tuple, Optional, Any
import math
import random
import networkx as nx
from src.global_routing.cost_function import calculate_edge_cost


class RoadNetworkManager:
    """Manages the road graph, node positions, edge attributes, and traffic densities."""

    def __init__(self, is_directed: bool = True):
        self.is_directed = is_directed
        self.graph = nx.DiGraph() if is_directed else nx.Graph()

    def add_intersection(self, node_id: str, x: float, y: float) -> None:
        """Add an intersection node with 2D coordinates."""
        self.graph.add_node(node_id, pos=(float(x), float(y)))

    def add_road(
        self,
        u: str,
        v: str,
        length: Optional[float] = None,
        max_speed: float = 13.89,  # ~50 km/h in m/s
        lanes: int = 2,
        density: float = 0.0,
        bidirectional: bool = False
    ) -> None:
        """
        Add a road edge between nodes u and v.
        If length is None, Euclidean distance between node coordinates is used.
        """
        if u not in self.graph.nodes or v not in self.graph.nodes:
            raise KeyError(f"Nodes {u} and {v} must exist before connecting with road.")

        if length is None:
            pos_u = self.graph.nodes[u]["pos"]
            pos_v = self.graph.nodes[v]["pos"]
            length = math.hypot(pos_v[0] - pos_u[0], pos_v[1] - pos_u[1])

        cost = calculate_edge_cost(length=length, max_speed=max_speed, density=density)
        
        edge_data = {
            "length": float(length),
            "max_speed": float(max_speed),
            "lanes": int(lanes),
            "density": float(density),
            "cost": float(cost),
        }

        self.graph.add_edge(u, v, **edge_data)
        if bidirectional:
            self.graph.add_edge(v, u, **edge_data)

    def update_density(self, u: str, v: str, density: float) -> None:
        """Update traffic density on an edge and recalculate dynamic cost."""
        if not self.graph.has_edge(u, v):
            raise KeyError(f"Edge ({u}, {v}) does not exist.")

        data = self.graph[u][v]
        data["density"] = max(0.0, min(1.0, float(density)))
        data["cost"] = calculate_edge_cost(
            length=data["length"],
            max_speed=data["max_speed"],
            density=data["density"]
        )

    def simulate_random_traffic_jam(self, congested_ratio: float = 0.25, seed: Optional[int] = None) -> None:
        """Simulate realistic stochastic traffic congestion across a subset of roads."""
        rng = random.Random(seed)
        edges = list(self.graph.edges())
        num_congested = int(len(edges) * congested_ratio)
        congested_edges = rng.sample(edges, num_congested) if edges else []

        for u, v in edges:
            if (u, v) in congested_edges:
                # Heavy traffic density: 0.65 to 0.95
                d = rng.uniform(0.65, 0.95)
            else:
                # Light/Moderate traffic density: 0.05 to 0.35
                d = rng.uniform(0.05, 0.35)
            self.update_density(u, v, d)

    @classmethod
    def create_grid_city(
        cls,
        rows: int = 4,
        cols: int = 4,
        block_size: float = 100.0,
        default_speed: float = 13.89,  # 50 km/h
        bidirectional: bool = True
    ) -> "RoadNetworkManager":
        """
        Helper factory to generate a synthetic Manhattan-style grid city network.
        Nodes are named 'N_r_c' (e.g. N_0_0, N_0_1, ...).
        """
        manager = cls(is_directed=True)
        # 1. Add nodes
        for r in range(rows):
            for c in range(cols):
                node_id = f"N_{r}_{c}"
                x = c * block_size
                y = r * block_size
                manager.add_intersection(node_id, x, y)

        # 2. Add horizontal and vertical edges
        for r in range(rows):
            for c in range(cols):
                curr = f"N_{r}_{c}"
                # Horizontal neighbor (East)
                if c + 1 < cols:
                    east = f"N_{r}_{c+1}"
                    manager.add_road(curr, east, length=block_size, max_speed=default_speed, bidirectional=bidirectional)
                # Vertical neighbor (North)
                if r + 1 < rows:
                    north = f"N_{r+1}_{c}"
                    manager.add_road(curr, north, length=block_size, max_speed=default_speed, bidirectional=bidirectional)

        return manager

    def get_node_pos(self, node_id: str) -> Tuple[float, float]:
        """Return 2D coordinates (x, y) of a node."""
        return self.graph.nodes[node_id]["pos"]
