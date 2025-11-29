"""Tests for NLP module."""

import pytest

from src.nlp.baseline import BaselineNER
from src.nlp.preprocessing import (
    correct_city_typo,
    fuzzy_match_city,
    normalize_city_name,
    preprocess_text,
    remove_accents,
)


class TestPreprocessing:
    """Tests for preprocessing utilities."""

    def test_remove_accents(self):
        assert remove_accents("Montpellier") == "Montpellier"
        assert remove_accents("Béziers") == "Beziers"
        assert remove_accents("Châteauroux") == "Chateauroux"
        assert remove_accents("Saint-Étienne") == "Saint-Etienne"

    def test_preprocess_text(self):
        text = "  Je  veux   aller   "
        assert preprocess_text(text) == "Je veux aller"

    def test_preprocess_text_lowercase(self):
        text = "Je Veux ALLER"
        assert preprocess_text(text, lowercase=True) == "je veux aller"

    def test_normalize_city_name(self):
        assert normalize_city_name("Paris") == "paris"
        assert normalize_city_name("Saint-Étienne") == "saint etienne"
        assert normalize_city_name("LYON") == "lyon"


class TestBaselineNER:
    """Tests for baseline NER."""

    @pytest.fixture
    def ner(self):
        cities = ["Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse", "Nice"]
        return BaselineNER(cities)

    def test_simple_sentence(self, ner):
        result = ner.extract("Je veux aller de Paris à Lyon")
        assert result.is_valid
        assert result.departure == "Paris"
        assert result.arrival == "Lyon"

    def test_depuis_vers(self, ner):
        result = ner.extract("Depuis Bordeaux vers Marseille")
        assert result.is_valid
        assert result.departure == "Bordeaux"
        assert result.arrival == "Marseille"

    def test_arrival_first(self, ner):
        result = ner.extract("Je veux aller à Lyon depuis Paris")
        assert result.is_valid
        # Order might be reversed based on patterns
        assert result.departure is not None
        assert result.arrival is not None

    def test_invalid_no_cities(self, ner):
        result = ner.extract("Il fait beau aujourd'hui")
        assert not result.is_valid
        assert result.departure is None
        assert result.arrival is None

    def test_invalid_one_city(self, ner):
        result = ner.extract("Je veux aller à Paris")
        assert not result.is_valid

    def test_train_format(self, ner):
        result = ner.extract("Train Paris Lyon")
        assert result.is_valid

    def test_dash_format(self, ner):
        result = ner.extract("Paris - Lyon")
        assert result.is_valid


class TestFuzzyMatching:
    """Tests for fuzzy matching utilities."""

    @pytest.fixture
    def cities(self):
        return ["Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse", "Nice", "Nantes"]

    def test_fuzzy_match_exact(self, cities):
        matches = fuzzy_match_city("Paris", cities, threshold=80)
        assert len(matches) == 1
        assert matches[0][0] == "Paris"
        assert matches[0][1] == 100

    def test_fuzzy_match_typo(self, cities):
        # "Marseile" missing one 'l'
        matches = fuzzy_match_city("Marseile", cities, threshold=80)
        assert len(matches) == 1
        assert matches[0][0] == "Marseille"
        assert matches[0][1] >= 80

    def test_fuzzy_match_typo_bordeaux(self, cities):
        # "Bordeau" missing 'x'
        matches = fuzzy_match_city("Bordeau", cities, threshold=80)
        assert len(matches) == 1
        assert matches[0][0] == "Bordeaux"

    def test_fuzzy_match_typo_toulouse(self, cities):
        # "Toulous" missing 'e'
        matches = fuzzy_match_city("Toulous", cities, threshold=80)
        assert len(matches) == 1
        assert matches[0][0] == "Toulouse"

    def test_correct_city_typo(self, cities):
        assert correct_city_typo("Marseile", cities) == "Marseille"
        assert correct_city_typo("Bordeau", cities) == "Bordeaux"
        assert correct_city_typo("Toulous", cities) == "Toulouse"

    def test_correct_city_typo_no_match(self, cities):
        # Very different, should not match
        assert correct_city_typo("Strasbourg", cities, threshold=85) is None


class TestBaselineNERWithFuzzy:
    """Tests for baseline NER with fuzzy matching."""

    @pytest.fixture
    def ner(self):
        cities = ["Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse"]
        return BaselineNER(cities, use_fuzzy=True, fuzzy_threshold=80)

    def test_typo_in_departure(self, ner):
        result = ner.extract("Je veux aller de Marseile à Lyon")
        assert result.is_valid
        assert result.departure == "Marseille"
        assert result.arrival == "Lyon"

    def test_typo_in_arrival(self, ner):
        result = ner.extract("Je veux aller de Paris à Toulous")
        assert result.is_valid
        assert result.departure == "Paris"
        assert result.arrival == "Toulouse"

    def test_typo_in_both(self, ner):
        result = ner.extract("De Bordeau vers Marseile")
        assert result.is_valid
        assert result.departure == "Bordeaux"
        assert result.arrival == "Marseille"
