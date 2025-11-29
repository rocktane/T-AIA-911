"""Distance calculation utilities using Haversine formula."""

import math


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Uses the Haversine formula to compute the distance between two GPS coordinates.

    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)

    Returns:
        Distance in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def euclidean_approx(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Fast approximate distance using Euclidean formula.

    Only accurate for short distances. Uses ~111 km per degree approximation.

    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)

    Returns:
        Approximate distance in kilometers
    """
    # Average latitude for longitude correction
    avg_lat = (lat1 + lat2) / 2
    lon_correction = math.cos(math.radians(avg_lat))

    dx = (lon2 - lon1) * lon_correction * 111
    dy = (lat2 - lat1) * 111

    return math.sqrt(dx**2 + dy**2)
