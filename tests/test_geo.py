"""Tests for geo module."""

import pytest

from src.geo.distance import euclidean_approx, haversine


class TestHaversine:
    """Tests for Haversine distance calculation."""

    def test_same_point(self):
        distance = haversine(48.8566, 2.3522, 48.8566, 2.3522)
        assert distance == pytest.approx(0.0, abs=0.001)

    def test_paris_lyon(self):
        # Paris: 48.8566, 2.3522
        # Lyon: 45.7640, 4.8357
        distance = haversine(48.8566, 2.3522, 45.7640, 4.8357)
        # Should be around 390-400 km
        assert 380 < distance < 420

    def test_paris_marseille(self):
        # Paris: 48.8566, 2.3522
        # Marseille: 43.2965, 5.3698
        distance = haversine(48.8566, 2.3522, 43.2965, 5.3698)
        # Should be around 660-680 km
        assert 650 < distance < 700

    def test_short_distance(self):
        # Paris to nearby suburb (about 10 km)
        distance = haversine(48.8566, 2.3522, 48.9, 2.4)
        assert 5 < distance < 15


class TestEuclideanApprox:
    """Tests for Euclidean approximation."""

    def test_short_distance_similar_to_haversine(self):
        # For short distances, should be similar to Haversine
        lat1, lon1 = 48.8566, 2.3522
        lat2, lon2 = 48.9, 2.4

        haversine_dist = haversine(lat1, lon1, lat2, lon2)
        euclidean_dist = euclidean_approx(lat1, lon1, lat2, lon2)

        # Should be within 10% for short distances
        assert abs(haversine_dist - euclidean_dist) / haversine_dist < 0.1
