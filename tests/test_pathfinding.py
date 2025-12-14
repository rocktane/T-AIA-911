"""Tests for pathfinding module."""

import pytest

from src.pathfinding.dijkstra import PathFinder
from src.pathfinding.graph import RailwayGraph


class TestRailwayGraph:
    """Tests for RailwayGraph."""

    @pytest.fixture
    def graph(self):
        g = RailwayGraph()
        # Add nodes manually for testing
        g.graph.add_node("Paris", lat=48.8566, lon=2.3522)
        g.graph.add_node("Lyon", lat=45.7640, lon=4.8357)
        g.graph.add_node("Marseille", lat=43.2965, lon=5.3698)
        g.graph.add_node("Bordeaux", lat=44.8378, lon=-0.5792)
        return g

    def test_has_station(self, graph):
        assert graph.has_station("Paris")
        assert graph.has_station("Lyon")
        assert not graph.has_station("Unknown")

    def test_get_stations(self, graph):
        stations = graph.get_stations()
        assert "Paris" in stations
        assert "Lyon" in stations
        assert len(stations) == 4

    def test_build_connections(self, graph):
        # Build connections with a large distance to connect all
        graph.build_connections_from_coordinates(max_distance_km=500)

        # Paris and Lyon should be connected
        assert graph.graph.has_edge("Paris", "Lyon")


class TestPathFinder:
    """Tests for PathFinder."""

    @pytest.fixture
    def pathfinder(self):
        g = RailwayGraph()
        g.graph.add_node("A", lat=0, lon=0)
        g.graph.add_node("B", lat=0, lon=1)
        g.graph.add_node("C", lat=0, lon=2)
        g.graph.add_node("D", lat=1, lon=1)

        g.graph.add_edge("A", "B", weight=100)
        g.graph.add_edge("B", "C", weight=100)
        g.graph.add_edge("A", "D", weight=150)
        g.graph.add_edge("D", "C", weight=150)

        return PathFinder(g)

    def test_direct_path(self, pathfinder):
        result = pathfinder.find_path("A", "B")
        assert result.found
        assert result.path == ["A", "B"]
        assert result.total_distance == 100

    def test_multi_hop_path(self, pathfinder):
        result = pathfinder.find_path("A", "C")
        assert result.found
        # Should take A -> B -> C (200) instead of A -> D -> C (300)
        assert result.path == ["A", "B", "C"]
        assert result.total_distance == 200

    def test_same_station(self, pathfinder):
        result = pathfinder.find_path("A", "A")
        assert result.found
        assert result.path == ["A"]
        assert result.total_distance == 0

    def test_no_path(self, pathfinder):
        # Add disconnected node
        pathfinder.graph.add_node("Z", lat=10, lon=10)
        result = pathfinder.find_path("A", "Z")
        assert not result.found
        assert result.path == []

    def test_unknown_station(self, pathfinder):
        result = pathfinder.find_path("A", "Unknown")
        assert not result.found


class TestPathFinderWithWaypoints:
    """Tests for PathFinder with waypoints (VIA support)."""

    @pytest.fixture
    def pathfinder(self):
        g = RailwayGraph()
        # Create a more complex graph for waypoint testing
        # A -- B -- C -- D
        # |    |    |
        # E -- F -- G
        g.graph.add_node("A", lat=0, lon=0)
        g.graph.add_node("B", lat=0, lon=1)
        g.graph.add_node("C", lat=0, lon=2)
        g.graph.add_node("D", lat=0, lon=3)
        g.graph.add_node("E", lat=1, lon=0)
        g.graph.add_node("F", lat=1, lon=1)
        g.graph.add_node("G", lat=1, lon=2)

        g.graph.add_edge("A", "B", weight=100, distance=100, duration=60)
        g.graph.add_edge("B", "C", weight=100, distance=100, duration=60)
        g.graph.add_edge("C", "D", weight=100, distance=100, duration=60)
        g.graph.add_edge("A", "E", weight=100, distance=100, duration=60)
        g.graph.add_edge("B", "F", weight=100, distance=100, duration=60)
        g.graph.add_edge("C", "G", weight=100, distance=100, duration=60)
        g.graph.add_edge("E", "F", weight=100, distance=100, duration=60)
        g.graph.add_edge("F", "G", weight=100, distance=100, duration=60)

        return PathFinder(g)

    def test_path_with_single_waypoint(self, pathfinder):
        """Test path from A to D via B."""
        result = pathfinder.find_path_with_waypoints("A", "D", ["B"])
        assert result.found
        assert result.path[0] == "A"
        assert "B" in result.path
        assert result.path[-1] == "D"

    def test_path_with_multiple_waypoints(self, pathfinder):
        """Test path from A to D via B and C."""
        result = pathfinder.find_path_with_waypoints("A", "D", ["B", "C"])
        assert result.found
        assert result.path == ["A", "B", "C", "D"]
        assert result.total_distance == 300

    def test_path_no_waypoints(self, pathfinder):
        """Test path without waypoints (should work like regular find_path)."""
        result = pathfinder.find_path_with_waypoints("A", "D", [])
        assert result.found
        assert result.path[0] == "A"
        assert result.path[-1] == "D"

    def test_best_path_multi_with_waypoints(self, pathfinder):
        """Test finding best path among multiple stations with waypoints."""
        result = pathfinder.find_best_path_multi_with_waypoints(
            ["A", "E"],  # departure options
            ["D", "G"],  # destination options
            [["B", "F"]]  # waypoint options
        )
        assert result.found
        # Should find a valid path through one of the waypoints

    def test_best_path_multi_with_multiple_waypoints(self, pathfinder):
        """Test with multiple waypoint sets."""
        result = pathfinder.find_best_path_multi_with_waypoints(
            ["A"],
            ["D"],
            [["B"], ["C"]]  # Two waypoints: first through B, then through C
        )
        assert result.found
        assert result.path == ["A", "B", "C", "D"]

    def test_best_path_multi_no_waypoints(self, pathfinder):
        """Test best path multi with empty waypoints list."""
        result = pathfinder.find_best_path_multi_with_waypoints(
            ["A", "E"],
            ["D", "G"],
            []  # No waypoints
        )
        assert result.found
        # Should find shortest path without waypoints

    def test_best_path_multi_invalid_waypoint(self, pathfinder):
        """Test with invalid waypoint station."""
        result = pathfinder.find_best_path_multi_with_waypoints(
            ["A"],
            ["D"],
            [["INVALID"]]  # Invalid waypoint
        )
        assert not result.found
