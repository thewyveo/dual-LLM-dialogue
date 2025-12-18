from typing import List, Dict, Optional
import json
import os


class HotelAPIClient:
    """
    Simple JSON-backed hotel API.
    (not an actual HTTP client, just retrieves from local data)
    """

    def __init__(self, data_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(base_dir, "data", "hotels_synth.json")
        self.data_path = data_path or default_path
        self.hotels = self._load_hotels()

    def _load_hotels(self) -> List[Dict]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Hotel data file not found at {self.data_path}"
            )
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def search_hotels(
        self,
        location: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_price: Optional[float] = None,
        limit: int = 5,
    ) -> List[Dict]:
        """
        Filter over the local hotels list.

        - location: substring match inside the "location" field
        - min_rating: rating >= min_rating
        - max_price: price_numeric <= max_price
        """

        candidates = self.hotels

        if location:
            loc_lower = location.lower()
            candidates = [
                h for h in candidates
                if loc_lower in h["location"].lower()
            ]

        if min_rating is not None:
            candidates = [h for h in candidates if h["rating"] >= min_rating]

        if max_price is not None:
            candidates = [
                h for h in candidates
                if h["price_numeric"] <= max_price
            ]

        # sort by rating DESC, then price ASC
        candidates = sorted(
            candidates,
            key=lambda h: (-h["rating"], h["price_numeric"])
        )

        return candidates[:limit]
