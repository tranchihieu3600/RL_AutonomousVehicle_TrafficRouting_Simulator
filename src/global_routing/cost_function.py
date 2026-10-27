"""
Dynamic edge cost calculation for traffic routing.
Cost combines free-flow travel time and density-based congestion penalty.
"""

def calculate_edge_cost(
    length: float,
    max_speed: float,
    density: float,
    beta: float = 1.5,
    epsilon: float = 0.05
) -> float:
    """
    Calculate dynamic edge traversal cost.

    Formula:
        Cost = (Length / max_speed) * (1 + beta * (density / (1 - density + epsilon)))

    Args:
        length (float): Road length in meters.
        max_speed (float): Maximum allowed speed in m/s (must be > 0).
        density (float): Traffic density between 0.0 (empty) and 1.0 (gridlock).
        beta (float): Congestion penalty factor (default 1.5).
        epsilon (float): Smoothing factor to avoid zero division.

    Returns:
        float: Calculated dynamic travel cost (time-equivalent units in seconds).
    """
    if max_speed <= 0:
        raise ValueError("max_speed must be positive and greater than 0.")
    
    # Clamp density to [0.0, 1.0]
    clamped_density = max(0.0, min(1.0, float(density)))
    
    free_flow_time = length / max_speed
    congestion_factor = 1.0 + beta * (clamped_density / (1.0 - clamped_density + epsilon))
    
    return free_flow_time * congestion_factor
