"""
FastAPI web interface for Travel Order Resolver.

Minimal, performance-focused interface with multi-model support.
"""

import csv
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Form, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.geo import CityResolver
from src.nlp import SpacyNER, BaselineNER
from src.pathfinding import PathFinder, RailwayGraph

# Try to import CamemBERT (optional)
try:
    from src.nlp import CamemBERTNER
    CAMEMBERT_AVAILABLE = True
except ImportError:
    CAMEMBERT_AVAILABLE = False
    CamemBERTNER = None

# Paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
STATIONS_FILE = DATA_DIR / "gares-sncf.csv"
CONNECTIONS_FILE = DATA_DIR / "connexions.csv"
CITIES_FILE = DATA_DIR / "communes-france.csv"
SPACY_MODEL_PATH = MODELS_DIR / "spacy-ner" / "model-best"
CAMEMBERT_MODEL_PATH = MODELS_DIR / "camembert-ner"

# Initialize FastAPI
app = FastAPI(
    title="Travel Order Resolver",
    description="NLP system for French train travel",
    version="0.1.0",
)

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Jinja2 filters
def format_duration(minutes: float | None) -> str:
    """Format duration in minutes to human readable string."""
    if minutes is None:
        return ""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}h{mins:02d}"
    return f"{mins} min"


templates.env.filters["format_duration"] = format_duration


# Global instances (loaded on startup)
ner_spacy = None
ner_camembert = None
resolver = None
pathfinder = None
cities_list = None


def load_city_names(filepath: Path) -> list[str]:
    """Load city names from stations CSV."""
    names = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("nom", "").strip()
            if name:
                names.append(name)
    return names


def load_commune_names(filepath: Path) -> list[str]:
    """Load commune names from communes CSV."""
    names = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get the proper name (with accents and hyphens)
            name = row.get("nom_commune", "").strip()
            if name:
                names.append(name)
    return names


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global ner_spacy, ner_camembert, resolver, pathfinder, cities_list

    if STATIONS_FILE.exists():
        cities = load_city_names(STATIONS_FILE)
        cities_list = cities
        communes = load_commune_names(CITIES_FILE) if CITIES_FILE.exists() else None

        # Load spaCy model (custom trained or pretrained)
        if SPACY_MODEL_PATH.exists():
            print(f"Loading custom spaCy model from {SPACY_MODEL_PATH}")
            ner_spacy = SpacyNER(cities, communes=communes, model_path=SPACY_MODEL_PATH)
        else:
            print("Loading pretrained spaCy model (fr_core_news_md)")
            ner_spacy = SpacyNER(cities, communes=communes, use_pretrained=True)

        # Load CamemBERT model if available
        if CAMEMBERT_AVAILABLE and CAMEMBERT_MODEL_PATH.exists():
            print(f"Loading CamemBERT model from {CAMEMBERT_MODEL_PATH}")
            try:
                ner_camembert = CamemBERTNER(CAMEMBERT_MODEL_PATH, cities, communes=communes)
                print("CamemBERT model loaded successfully")
            except Exception as e:
                print(f"Failed to load CamemBERT model: {e}")
                ner_camembert = None
        else:
            if not CAMEMBERT_AVAILABLE:
                print("CamemBERT not available (transformers not installed)")
            else:
                print(f"CamemBERT model not found at {CAMEMBERT_MODEL_PATH}")
            ner_camembert = None

        resolver = CityResolver(
            STATIONS_FILE,
            CITIES_FILE if CITIES_FILE.exists() else None,
        )

        # Initialize pathfinder with real SNCF connections
        graph = RailwayGraph()
        graph.load_stations(STATIONS_FILE)
        if CONNECTIONS_FILE.exists():
            graph.load_connections(CONNECTIONS_FILE)
        else:
            # Fallback to geographic proximity if no connections file
            graph.build_connections_from_coordinates(max_distance_km=150)
        pathfinder = PathFinder(graph)


class DetectionInfo(BaseModel):
    """NER detection information."""

    raw_text: str  # The raw text extracted by NER
    matched_city: str | None = None  # The city matched in database
    start_pos: int | None = None  # Start position in sentence
    end_pos: int | None = None  # End position in sentence
    confidence: float = 0.0  # NER confidence score
    match_score: float = 1.0  # City matching score (1.0 = exact, <1.0 = fuzzy)
    is_fuzzy: bool = False  # True if fuzzy matched


class QueryRequest(BaseModel):
    """Request model for API."""

    sentence: str


class SegmentDisplay(BaseModel):
    """Display info for a route segment."""

    from_station: str
    to_station: str
    duration_min: float | None
    distance_km: float
    train_type: str


class QueryResponse(BaseModel):
    """Response model for API."""

    is_valid: bool
    model_name: str = "spacy"  # Model used for extraction
    departure: str | None = None
    departure_station: str | None = None
    arrival: str | None = None
    arrival_station: str | None = None
    route: list[str] | None = None
    distance_km: float | None = None
    duration_min: float | None = None  # Total duration in minutes
    num_connections: int = 0  # Number of train changes
    segments: list[SegmentDisplay] | None = None  # Segment details with train types
    error: str | None = None
    # Detection information
    departure_detection: DetectionInfo | None = None
    arrival_detection: DetectionInfo | None = None
    highlighted_sentence: str | None = None  # Sentence with HTML highlights


class ModelsStatusResponse(BaseModel):
    """Response for models status."""
    spacy_available: bool
    spacy_custom: bool
    camembert_available: bool


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render main page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "spacy_available": ner_spacy is not None,
            "camembert_available": ner_camembert is not None,
        },
    )


@app.post("/", response_class=HTMLResponse)
async def process_form(request: Request, sentence: str = Form(...), model: str = Form("spacy")):
    """Process form submission and return results."""
    result = process_sentence(sentence, model)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sentence": sentence,
            "result": result,
            "selected_model": model,
            "spacy_available": ner_spacy is not None,
            "camembert_available": ner_camembert is not None,
        },
    )


@app.get("/api/models", response_model=ModelsStatusResponse)
async def api_models_status() -> ModelsStatusResponse:
    """Get available models status."""
    return ModelsStatusResponse(
        spacy_available=ner_spacy is not None,
        spacy_custom=SPACY_MODEL_PATH.exists(),
        camembert_available=ner_camembert is not None,
    )


@app.post("/api/query", response_model=QueryResponse)
async def api_query(
    query: QueryRequest,
    model: Literal["spacy", "camembert"] = Query(default="spacy"),
) -> QueryResponse:
    """API endpoint for programmatic access."""
    return process_sentence(query.sentence, model)


def extract_city_prefix(station_name: str) -> str:
    """
    Extract the city name prefix from a station name.

    Examples:
        "Paris Est" -> "Paris"
        "Marseille Saint-Charles" -> "Marseille"
        "Paris - Gare de Lyon - Hall 1 & 2" -> "Paris"
        "Lyon Part Dieu" -> "Lyon"
    """
    # Handle names with " - " separator (e.g., "Paris - Gare de Lyon")
    if " - " in station_name:
        station_name = station_name.split(" - ")[0]

    # Take the first word as the city prefix
    parts = station_name.split()
    if parts:
        return parts[0]
    return station_name


def create_highlighted_sentence(
    sentence: str,
    dep_start: int | None,
    dep_end: int | None,
    arr_start: int | None,
    arr_end: int | None,
) -> str:
    """
    Create HTML with highlighted departure (green) and arrival (red).

    Handles overlapping or adjacent spans by processing from end to start.
    """
    if dep_start is None and arr_start is None:
        return sentence

    # Collect spans with their types
    spans = []
    if dep_start is not None and dep_end is not None:
        spans.append((dep_start, dep_end, "departure"))
    if arr_start is not None and arr_end is not None:
        spans.append((arr_start, arr_end, "arrival"))

    # Sort by start position descending (process from end to avoid offset issues)
    spans.sort(key=lambda x: x[0], reverse=True)

    result = sentence
    for start, end, span_type in spans:
        css_class = "highlight-departure" if span_type == "departure" else "highlight-arrival"
        highlighted = f'<span class="{css_class}">{result[start:end]}</span>'
        result = result[:start] + highlighted + result[end:]

    return result


def process_sentence(sentence: str, model: str = "spacy") -> QueryResponse:
    """Process a sentence and return structured response."""
    # Select the appropriate NER model
    if model == "camembert":
        ner = ner_camembert
        model_name = "CamemBERT"
    else:
        ner = ner_spacy
        model_name = "spaCy"

    if not ner or not resolver:
        return QueryResponse(
            is_valid=False,
            model_name=model_name,
            error=f"Le modèle {model_name} n'est pas disponible. Veuillez réessayer plus tard.",
        )

    # NLP extraction
    result = ner.extract(sentence)

    if not result.is_valid:
        return QueryResponse(is_valid=False, model_name=model_name, error="Impossible de détecter un trajet valide. Veuillez préciser une ville de départ et d'arrivée.")

    departure, arrival = result.departure, result.arrival

    # Get detection positions and confidence from NER result
    dep_start = getattr(result, "departure_start", None)
    dep_end = getattr(result, "departure_end", None)
    arr_start = getattr(result, "arrival_start", None)
    arr_end = getattr(result, "arrival_end", None)
    dep_confidence = getattr(result, "departure_confidence", 0.0)
    arr_confidence = getattr(result, "arrival_confidence", 0.0)

    # Resolve cities
    dep_resolved = resolver.resolve(departure) if departure else None
    arr_resolved = resolver.resolve(arrival) if arrival else None

    # Build detection info
    dep_detection = None
    arr_detection = None

    if departure:
        dep_detection = DetectionInfo(
            raw_text=departure,
            matched_city=dep_resolved.city_matched if dep_resolved else None,
            start_pos=dep_start,
            end_pos=dep_end,
            confidence=dep_confidence,
            match_score=dep_resolved.match_score if dep_resolved else 0.0,
            is_fuzzy=dep_resolved.match_score < 1.0 if dep_resolved else False,
        )

    if arrival:
        arr_detection = DetectionInfo(
            raw_text=arrival,
            matched_city=arr_resolved.city_matched if arr_resolved else None,
            start_pos=arr_start,
            end_pos=arr_end,
            confidence=arr_confidence,
            match_score=arr_resolved.match_score if arr_resolved else 0.0,
            is_fuzzy=arr_resolved.match_score < 1.0 if arr_resolved else False,
        )

    # Create highlighted sentence
    highlighted = create_highlighted_sentence(
        sentence, dep_start, dep_end, arr_start, arr_end
    )

    # Extract city prefixes to find all possible stations
    dep_prefix = extract_city_prefix(departure) if departure else None
    arr_prefix = extract_city_prefix(arrival) if arrival else None

    # Get all stations for each city
    dep_stations = resolver.get_all_stations_for_city(dep_prefix) if dep_prefix else []
    arr_stations = resolver.get_all_stations_for_city(arr_prefix) if arr_prefix else []

    if not dep_stations:
        return QueryResponse(
            is_valid=False,
            model_name=model_name,
            departure=departure,
            arrival=arrival,
            error=f"Ville de départ non reconnue : {departure}. Vérifiez l'orthographe ou essayez une ville proche.",
            departure_detection=dep_detection,
            arrival_detection=arr_detection,
            highlighted_sentence=highlighted,
        )

    if not arr_stations:
        return QueryResponse(
            is_valid=False,
            model_name=model_name,
            departure=departure,
            arrival=arrival,
            error=f"Ville d'arrivée non reconnue : {arrival}. Vérifiez l'orthographe ou essayez une ville proche.",
            departure_detection=dep_detection,
            arrival_detection=arr_detection,
            highlighted_sentence=highlighted,
        )

    # Find the best route among all station combinations
    route = None
    distance = None
    duration = None
    num_connections = 0
    segments = None
    dep_station = dep_stations[0]  # Default
    arr_station = arr_stations[0]  # Default

    if pathfinder:
        path_result = pathfinder.find_best_path_multi(dep_stations, arr_stations)
        if path_result.found:
            route = path_result.path
            distance = path_result.total_distance
            duration = path_result.total_duration
            num_connections = path_result.num_connections
            # Convert segments to display format
            if path_result.segments:
                segments = [
                    SegmentDisplay(
                        from_station=seg.from_station,
                        to_station=seg.to_station,
                        duration_min=seg.duration_min,
                        distance_km=seg.distance_km,
                        train_type=seg.train_type,
                    )
                    for seg in path_result.segments
                ]
            # Update station names from the actual path
            if route:
                dep_station = route[0]
                arr_station = route[-1]

    return QueryResponse(
        is_valid=True,
        model_name=model_name,
        departure=dep_prefix,
        departure_station=dep_station,
        arrival=arr_prefix,
        arrival_station=arr_station,
        route=route,
        distance_km=distance,
        duration_min=duration,
        num_connections=num_connections,
        segments=segments,
        departure_detection=dep_detection,
        arrival_detection=arr_detection,
        highlighted_sentence=highlighted,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "spacy_loaded": ner_spacy is not None,
        "camembert_loaded": ner_camembert is not None,
        "resolver_loaded": resolver is not None,
        "pathfinder_loaded": pathfinder is not None,
    }
