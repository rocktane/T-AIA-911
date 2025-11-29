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
