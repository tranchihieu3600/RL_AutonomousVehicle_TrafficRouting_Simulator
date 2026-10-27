"""
Plotting and visualization utilities for global graph networks and paths.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import networkx as nx
from src.global_routing.graph_manager import RoadNetworkManager


def plot_road_network(
    network_manager: RoadNetworkManager,
    path_nodes: Optional[List[str]] = None,
    title: str = "Road Network Traffic Congestion Map",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize road network with edges color-coded by traffic density.
    Highlight calculated A* path if provided.
    """
    G = network_manager.graph
    pos = {n: G.nodes[n]["pos"] for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine edge colors based on density (green: clear, yellow: moderate, red: jammed)
    edge_densities = [G[u][v].get("density", 0.0) for u, v in G.edges()]
    
    # Draw base edges
    edges = nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_densities,
        edge_cmap=plt.cm.RdYlGn_r,
        edge_vmin=0.0,
        edge_vmax=1.0,
        width=3.0,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.05"
    )

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=250, node_color="#4A90E2", alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")

    # Highlight optimal path if provided
    if path_nodes and len(path_nodes) > 1:
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            ax=ax,
            edge_color="#00E676",  # Vibrant neon green
            width=5.0,
            arrows=True,
            arrowsize=16
        )
        # Highlight start and goal
        start, goal = path_nodes[0], path_nodes[-1]
        nx.draw_networkx_nodes(G, pos, nodelist=[start], ax=ax, node_size=400, node_color="#00E676")
        nx.draw_networkx_nodes(G, pos, nodelist=[goal], ax=ax, node_size=400, node_color="#E91E63")

    # Colorbar for traffic density
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.04)
    cbar.set_label("Traffic Congestion Density (0 = Free flow, 1 = Jammed)", fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Map visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()
