from __future__ import annotations
from typing import Any, Dict, List
import html
import requests
import urllib3
from mcp.server.fastmcp import FastMCP

import logging

logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)

# MCP is the "tool server" layer. The agent connects to this process over stdio.
mcp = FastMCP("WeekendWizard")

REQUEST_TIMEOUT = 20
USER_AGENT = "WeekendWizard/1.0"

# Some public APIs reject cert validation in this environment unless we skip it.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _get_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:

    # Small wrapper so every API call uses the same timeout, headers, and SSL behavior.
    # That keeps the individual tool functions short and easier to scan.
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}, verify=False,)
    response.raise_for_status()
    return response.json()


# ========= WEATHER CODE MAPPER ==========
def _weather_description(code: int | None) -> str:
    # Open-Meteo returns numeric weather codes, so we map them to human-readable text.
    descriptions = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "slight snow fall",
        73: "moderate snow fall",
        75: "heavy snow fall",
        80: "rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        95: "thunderstorm",
    }
    if code is None:
        return "unknown"
    return descriptions.get(code, f"weather code {code}")


def _wind_direction_cardinal(degrees: int | float | None) -> str:
    # Convert degrees into a simple compass label for easier reading.  For example, 194 degrees becomes something like SSW.
    if degrees is None:
        return "unknown"
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = int((float(degrees) + 11.25) // 22.5) % 16
    return directions[index]


def _dog_breed_path(breed: str) -> tuple[str, str | None]:
    # Dog CEO uses /breed/{breed}/{subbreed}/images/random for sub-breeds.
    # We only need a light mapping here, with a fallback for plain breed names.
    # This is enough for the assignment's French bulldog request.
    normalized = " ".join(breed.lower().strip().split())
    if normalized in {"french bulldog", "bulldog french"}:
        return "bulldog", "french"

    parts = normalized.split()
    if len(parts) >= 2:
        # Best-effort fallback for simple two-word requests like "labrador retriever".
        return parts[-1], parts[0]

    return normalized.replace(" ", "-"), None

# ========================================= WEATHER TOOL (Open Mateo) =========================================
@mcp.tool()
def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Return current weather for coordinates via Open-Meteo."""

    # Fetch the current weather snapshot for the supplied coordinates.  
    # We also request a short daily forecast so the agent can talk about tomorrow.
    data = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,wind_direction_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,wind_speed_10m_max,wind_direction_10m_dominant",
            "forecast_days": 2,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
        },
    )
    
    current = data.get("current", {})
    daily = data.get("daily", {})
    weather_code = current.get("weather_code")

    daily_rows: List[Dict[str, Any]] = []
    dates = daily.get("time", [])

    for index, date_value in enumerate(dates):
        # Open-Meteo returns parallel arrays for each daily field, so we repack them by index.
        daily_rows.append(
            {
                "date": date_value,
                "weather_code": (daily.get("weather_code") or [None])[index] if index < len(daily.get("weather_code") or []) else None,
                "weather_description": _weather_description((daily.get("weather_code") or [None])[index] if index < len(daily.get("weather_code") or []) else None),
                "temperature_2m_max": (daily.get("temperature_2m_max") or [None])[index] if index < len(daily.get("temperature_2m_max") or []) else None,
                "temperature_2m_min": (daily.get("temperature_2m_min") or [None])[index] if index < len(daily.get("temperature_2m_min") or []) else None,
                "apparent_temperature_max": (daily.get("apparent_temperature_max") or [None])[index] if index < len(daily.get("apparent_temperature_max") or []) else None,
                "apparent_temperature_min": (daily.get("apparent_temperature_min") or [None])[index] if index < len(daily.get("apparent_temperature_min") or []) else None,
                "wind_speed_10m_max": (daily.get("wind_speed_10m_max") or [None])[index] if index < len(daily.get("wind_speed_10m_max") or []) else None,
                "wind_direction_10m_dominant": (daily.get("wind_direction_10m_dominant") or [None])[index] if index < len(daily.get("wind_direction_10m_dominant") or []) else None,
                "wind_direction_cardinal": _wind_direction_cardinal((daily.get("wind_direction_10m_dominant") or [None])[index] if index < len(daily.get("wind_direction_10m_dominant") or []) else None),
            }
        )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": data.get("timezone"),
        "current": {
            "time": current.get("time"),
            "temperature_2m": current.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "weather_code": weather_code,
            "weather_description": _weather_description(weather_code),
            "wind_speed_10m": current.get("wind_speed_10m"),
            "wind_direction_10m": current.get("wind_direction_10m"),
            "wind_direction_cardinal": _wind_direction_cardinal(current.get("wind_direction_10m")),
        },
        "daily": daily_rows,
    }


@mcp.tool()
def city_to_coords(city: str, limit: int = 1) -> Dict[str, Any]:
    """Geocode a city name to coordinates via Open-Meteo."""

    # This is a convenience helper so the agent can work with city names too.
    # The main prompt often gives a place name rather than latitude/longitude.
    data = _get_json("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": limit, "language": "en", "format": "json"})

    results: List[Dict[str, Any]] = []
    
    for item in data.get("results", [])[:limit]:
        results.append({"name": item.get("name"), "admin1": item.get("admin1"), "country": item.get("country"), "latitude": item.get("latitude"), "longitude": item.get("longitude"), "timezone": item.get("timezone")})
    
    return {"query": city, "results": results}


# ========================================= BOOK TOOL (Open Library) =========================================
@mcp.tool()
def book_recs(topic: str, limit: int = 5) -> Dict[str, Any]:
    """Return book suggestions for a topic via Open Library."""

    # Search Open Library and reshape the response into a smaller, friendlier payload.
    # The goal is to return just the interesting parts for the final CLI answer.
    data = _get_json("https://openlibrary.org/search.json", params={"q": topic, "limit": limit},)

    picks: List[Dict[str, Any]] = []
    for doc in data.get("docs", [])[:limit]:
        author_names = doc.get("author_name") or []
        picks.append(
            {
                "title": doc.get("title"),
                "author": author_names[0] if author_names else "Unknown",
                "year": doc.get("first_publish_year"),
                "edition_count": doc.get("edition_count"),
                "key": doc.get("key"),
                "openlibrary_url": f"https://openlibrary.org{doc.get('key')}" if doc.get("key") else None,
            }
        )
    
    return {"topic": topic, "results": picks}


# ========================================= JOKE TOOL (JokeAPI) =========================================
@mcp.tool()
def random_joke() -> Dict[str, Any]:
    """Return one safe, single-line joke."""

    # JokeAPI already returns a single joke string when asked for safe-mode single jokes.
    # That makes it a good zero-key API for the capstone.
    data = _get_json("https://v2.jokeapi.dev/joke/Any?type=single&safe-mode")
    joke = data.get("joke", "")
    
    return {"joke": html.unescape(joke)}


# ========================================= DOG IMAGE TOOL (Dog.Ceo API) =========================================
@mcp.tool()
def random_dog(breed: str | None = None) -> Dict[str, Any]:
    """Return a random dog image URL."""

    # Dog CEO gives back a direct image URL, which is enough for the CLI demo.
    # When a breed is supplied, try to honor it instead of returning a totally random dog.
    # This is what lets "french bulldog" turn into a breed-specific URL.
    if breed:
        breed_name, sub_breed = _dog_breed_path(breed)
        if sub_breed:
            url = f"https://dog.ceo/api/breed/{breed_name}/{sub_breed}/images/random"
        else:
            url = f"https://dog.ceo/api/breed/{breed_name}/images/random"
    else:
        url = "https://dog.ceo/api/breeds/image/random"

    data = _get_json(url)
    
    return {"status": data.get("status"), "message": data.get("message"), "requested_breed": breed}


if __name__ == "__main__":
    mcp.run()
