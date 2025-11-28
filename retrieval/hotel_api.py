from typing import List, Dict, Optional
import json
import os


class HotelAPIClient:
    """
    Simple hotel "API" client.

    For now this loads hotels from a local JSON file (data/hotels.json),
    but the interface is written so that you can later replace the implementation
    with a real HTTP API (TripAdvisor, Yelp, etc.).
    """

    def __init__(self, data_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(base_dir, "data", "hotels.json")
        self.data_path = data_path or default_path
        self.hotels = self._load_hotels()

    def _load_hotels(self) -> List[Dict]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Hotel data file not found at {self.data_path}")
        with open(self.data_path, "r") as f:
            data = json.load(f)
        return data

    def search_hotels(
        self,
        location: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_price: Optional[float] = None,
        limit: int = 5,
    ) -> List[Dict]:
        """
        Very simple filter over the local hotels list.
        - location: substring match in city name
        - min_rating: filter by rating
        - max_price: filter by price bucket (1, 2, 3) interpreted from '$', '$$', '$$$'

        You can later replace this with a real HTTP request.
        """
        candidates = self.hotels

        if location:
            loc_lower = location.lower()
            candidates = [h for h in candidates if loc_lower in h["location"].lower()]

        if min_rating is not None:
            candidates = [h for h in candidates if h["rating"] >= min_rating]

        if max_price is not None:
            candidates = [h for h in candidates if h["price_numeric"] <= max_price]

        candidates = sorted(candidates, key=lambda h: (-h["rating"], h["price_numeric"]))

        return candidates[:limit]
