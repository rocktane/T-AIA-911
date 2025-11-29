"""Dijkstra pathfinding for railway routes."""

from dataclasses import dataclass, field

import networkx as nx

from .graph import RailwayGraph


@dataclass
class SegmentInfo:
    """Information about a path segment."""

    from_station: str
    to_station: str
    duration_min: float | None
    distance_km: float
    train_type: str


@dataclass
class PathResult:
    """Result of pathfinding."""

    path: list[str]
    total_distance: float  # Total distance in km
    found: bool
    total_duration: float | None = None  # Total duration in minutes
    num_connections: int = 0  # Number of train changes (stops - 1)
    segments: list[SegmentInfo] = field(default_factory=list)  # Details per segment


class PathFinder:
    """
    Find optimal paths in the railway network using Dijkstra's algorithm.

    Uses NetworkX's optimized implementation of Dijkstra's algorithm
    for finding shortest paths based on travel time.
    """

    def __init__(self, graph: RailwayGraph):
        """
        Initialize pathfinder with a railway graph.

        Args:
            graph: RailwayGraph instance
        """
        self.graph = graph.graph
        self._railway_graph = graph

    def find_path(self, departure: str, destination: str) -> PathResult:
        """
        Find the shortest path between two stations.

        Args:
            departure: Starting station name
            destination: Ending station name

        Returns:
            PathResult with path, distance, duration, and success flag
        """
        # Handle same station
        if departure == destination:
            return PathResult(
                path=[departure],
                total_distance=0.0,
                found=True,
                total_duration=0.0,
                num_connections=0,
                segments=[],
            )

        # Check if stations exist
        if departure not in self.graph:
            return PathResult(path=[], total_distance=0.0, found=False)
        if destination not in self.graph:
            return PathResult(path=[], total_distance=0.0, found=False)

        try:
            path = nx.dijkstra_path(
                self.graph, departure, destination, weight="weight"
            )

            # Calculate totals and collect segment info
            total_distance = 0.0
            total_duration = 0.0
            segments = []

            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                # Use distance if available, otherwise fall back to weight
                distance = edge_data.get("distance") or edge_data.get("weight", 0) or 0
                duration = edge_data.get("duration") or edge_data.get("weight", 0)
                train_type = edge_data.get("train_type", "Train")

                total_distance += distance
                total_duration += duration

                segments.append(SegmentInfo(
                    from_station=path[i],
                    to_station=path[i + 1],
                    duration_min=duration,
                    distance_km=distance,
                    train_type=train_type,
                ))

            return PathResult(
                path=path,
                total_distance=round(total_distance, 1),
                found=True,
                total_duration=round(total_duration, 0),
                num_connections=len(path) - 1,
                segments=segments,
            )
        except nx.NetworkXNoPath:
            return PathResult(path=[], total_distance=0.0, found=False)
        except nx.NodeNotFound:
            return PathResult(path=[], total_distance=0.0, found=False)

    def find_path_with_waypoints(
        self, departure: str, destination: str, waypoints: list[str]
    ) -> PathResult:
        """
        Find path through specified waypoints.

        Args:
            departure: Starting station
            destination: Ending station
            waypoints: List of intermediate stations to pass through

        Returns:
            PathResult with complete path
        """
        all_points = [departure] + waypoints + [destination]
        full_path = []
        total_distance = 0.0
        total_duration = 0.0
        all_segments = []

        for i in range(len(all_points) - 1):
            result = self.find_path(all_points[i], all_points[i + 1])
            if not result.found:
                return PathResult(path=[], total_distance=0.0, found=False)

            # Avoid duplicating waypoints in the path
            if full_path:
                full_path.extend(result.path[1:])
            else:
                full_path.extend(result.path)

            total_distance += result.total_distance
            total_duration += result.total_duration or 0
            all_segments.extend(result.segments)

        return PathResult(
            path=full_path,
            total_distance=round(total_distance, 1),
            found=True,
            total_duration=round(total_duration, 0),
            num_connections=len(full_path) - 1,
            segments=all_segments,
        )

    def get_alternative_paths(
        self, departure: str, destination: str, k: int = 3
    ) -> list[PathResult]:
        """
        Find k shortest paths between two stations.

        Args:
            departure: Starting station
            destination: Ending station
            k: Number of paths to find

        Returns:
            List of PathResult, sorted by duration
        """
        if departure not in self.graph or destination not in self.graph:
            return []

        try:
            paths = list(
                nx.shortest_simple_paths(
                    self.graph, departure, destination, weight="weight"
                )
            )[:k]

            results = []
            for path in paths:
                # Calculate totals and collect segment info
                total_distance = 0.0
                total_duration = 0.0
                segments = []

                for i in range(len(path) - 1):
                    edge_data = self.graph[path[i]][path[i + 1]]
                    # Use distance if available, otherwise fall back to weight
                    distance = edge_data.get("distance") or edge_data.get("weight", 0) or 0
                    duration = edge_data.get("duration") or edge_data.get("weight", 0)
                    train_type = edge_data.get("train_type", "Train")

                    total_distance += distance
                    total_duration += duration

                    segments.append(SegmentInfo(
                        from_station=path[i],
                        to_station=path[i + 1],
                        duration_min=duration,
                        distance_km=distance,
                        train_type=train_type,
                    ))

                results.append(PathResult(
                    path=path,
                    total_distance=round(total_distance, 1),
                    found=True,
                    total_duration=round(total_duration, 0),
                    num_connections=len(path) - 1,
                    segments=segments,
                ))

            return results
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def find_best_path_multi(
        self, departures: list[str], destinations: list[str]
    ) -> PathResult:
        """
        Find the best path among all combinations of departure and destination stations.

        Useful when a city has multiple stations (e.g., Paris has Paris Est,
        Paris Nord, Paris Gare de Lyon, etc.).

        Prioritizes:
        1. Shortest travel time (primary)
        2. Fewer connections (secondary)
        3. Shorter distance (tertiary)

        Args:
            departures: List of possible departure station names
            destinations: List of possible destination station names

        Returns:
            Best PathResult among all combinations
        """
        best_result = PathResult(path=[], total_distance=0.0, found=False)
        best_score = float("inf")

        for dep in departures:
            if dep not in self.graph:
                continue
            for dest in destinations:
                if dest not in self.graph:
                    continue

                result = self.find_path(dep, dest)
                if result.found:
                    # Score: primarily by duration, then by number of connections
                    # Lower score is better
                    duration = result.total_duration or result.total_distance
                    # Add 10 minutes penalty per connection to prefer direct routes
                    score = duration + (result.num_connections - 1) * 10

                    if score < best_score:
                        best_score = score
                        best_result = result

        return best_result


def format_duration(minutes: float | None) -> str:
    """Format duration in minutes to human readable string."""
    if minutes is None:
        return ""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}h{mins:02d}"
    return f"{mins} min"
