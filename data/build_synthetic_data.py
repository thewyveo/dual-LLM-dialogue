
import json
import random
import os

ADJECTIVES = [
    "Quiet", "Sunny", "Golden", "Canal", "Moonlight", "Royal", "Garden",
    "Central", "Cosy", "Elegant", "Budget", "Vintage", "Modern", "Charming",
    "Harbor", "Riverside", "Classic", "Urban", "Grand", "Maple", "Velvet",
    "Silver", "Emerald", "Copper", "Heritage", "Amber", "Nordic", "The Social",
    "The Urban", "The Cozy", "The Grand", "The Classic", "The Modern",
    "The Charming", "The Elegant", "The Vintage", "La Belle", "Le Grand",
    "Il Magnifico", "Das Gemütliche", "El Palacio", "Le Château", "Il Castello",
    "Das Schloss",
]

NOUNS = [
    "Inn", "Hotel", "Palace", "Retreat", "Suites", "Lodge", "House",
    "Residency", "View", "Canalhouse", "Hub", "Plaza", "Haven", "Stay",
    "Villa", "Terrace", "Courtyard", "Rooms", "Quarters", "Escape", "Dwell",
    "Abode", "Nook", "Sanctuary", "Oasis", "Harbor", "Residence", "Manor",
    "Chateau", "Bungalow", "Cottage", "Getaway", "Hideaway", "Boutique",
    "Establishment", "Outpost", "Yard"
]

def generate_hotel_name(existing):
    while True:
        name = f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
        if name not in existing:
            return name

REVIEW_TOPICS = {
    "wifi": [
        "Wi-Fi worked fine most of the time.",
        "Good Wi-Fi connection throughout the hotel.",
        "Wi-Fi was unreliable during peak hours.",
    ],
    "cleanliness": [
        "Rooms are clean and well-maintained.",
        "Clean bathrooms with good water pressure.",
        "Bathroom cleanliness could be improved.",
        "The room smelled musty and unpleasant.",
    ],
    "noise": [
        "Quiet neighborhood, ideal for a restful stay.",
        "The hotel is in a busy area, expect some street noise.",
        "Noisy street made it hard to sleep.",
        "Walls are thin, could hear neighbors at night."
    ],
    "location": [
        "Great location, ideal for exploring the city.",
        "Close to public transport and major attractions.",
        "The location is far from major attractions.",
        "Location is fairly convenient."
    ],
    "staff": [
        "Friendly staff and smooth check-in process.",
        "The staff went above and beyond to help us.",
        "Staff were polite but not particularly engaging.",
        "Unfriendly staff who seemed uninterested in helping."
    ],
    "food": [
        "Breakfast was decent with fresh pastries.",
        "The breakfast area can get crowded in the mornings.",
        "Breakfast options were okay, not too many choices.",
        "Breakfast was disappointing and lacked variety."
    ],
    "value": [
        "Excellent value for money.",
        "Reasonable stay for the price.",
        "Overpriced for what you get."
    ],
    "amenities": [
        "Spacious rooms with comfortable beds.",
        "The gym equipment was old and poorly maintained.",
        "The elevator is slow during peak hours.",
        "The pool was closed for maintenance during our stay."
    ],
}

# global usage tracking to reduce duplicates across hotels
GLOBAL_REVIEW_USE_COUNT = {
    topic: {review: 0 for review in REVIEWS}
    for topic, REVIEWS in REVIEW_TOPICS.items()
}

def choose_review_from_topic(topic, used):
    """
    Choose a review for this hotel from the given topic.
    - No duplicates inside hotel
    - Falls back gracefully if all weights = 0
    """

    all_reviews = REVIEW_TOPICS[topic]

    # filter out reviews already used by this hotel
    candidates = [r for r in all_reviews if r not in used]

    # if all reviews are already used (rare but possible), reset per-hotel uniqueness
    if not candidates:
        candidates = list(all_reviews)

    # build weights based on global usage
    weights = []
    for review in candidates:
        usage = GLOBAL_REVIEW_USE_COUNT[topic][review]
        # Lower usage = higher probability
        weights.append(1 / (1 + usage))

    # safety fallback: if weights sum to zero (should never happen but guard anyway)
    if sum(weights) == 0:
        weights = [1] * len(candidates)

    chosen = random.choices(candidates, weights=weights, k=1)[0]
    GLOBAL_REVIEW_USE_COUNT[topic][chosen] += 1
    return chosen

def generate_review_snippets():
    """
    Generate 2-6 review snippets drawn from a weighted mix of topics.
      - no duplicates within hotel
      - rare reviews get used more
      - sentiment influences rating later
    """
    num_reviews = random.randint(2, 6)
    used = set()
    snippets = []

    # topic weights for diversity
    topic_weights = {
        "wifi": 0.15,
        "cleanliness": 0.15,
        "noise": 0.10,
        "location": 0.20,
        "staff": 0.20,
        "food": 0.10,
        "value": 0.10,
        "amenities": 0.15
    }

    topics = list(topic_weights.keys())
    weights = list(topic_weights.values())

    while len(snippets) < num_reviews:
        topic = random.choices(topics, weights=weights, k=1)[0]
        review = choose_review_from_topic(topic, used)
        used.add(review)
        snippets.append(review)

    return snippets


NEIGHBORHOODS = [
    "Jordaan", "De Pijp", "Oud-West", "Centrum", "Zuidoost",
    "Westerpark", "Nieuwmarkt", "Grachtengordel", "Museumkwartier",
    "Oost", "Sloterdijk", "Osdorp", "Slotermeer", "Bos en Lommer",
    "Noord", "IJburg", "Zuid", "Amstelveld", "Plantage", "Haarlemmerbuurt",
]

AMENITIES_POOL = [
    "free_wifi", "breakfast_included", "air_conditioning",
    "elevator", "gym", "pet_friendly", "restaurant", "bar",
    "bike_rental", "airport_shuttle", "24h_front_desk",
    "room_service", "swimming_pool", "spa", "parking",
    "rooftop_terrace", "laundry_service", "family_rooms",
    "non_smoking_rooms", "luggage_storage", "free_billiards",
    "terrace", "garden", "sauna", "hot_tub"
]

def generate_amenities():
    n = random.randint(3, 7)
    return random.sample(AMENITIES_POOL, n)


def compute_rating(snippets):
    rating = random.uniform(3.0, 5.0)

    # topic/sentiment-based adjustments
    for s in snippets:
        if "musty" in s or "broken" in s or "dirty" in s or "unsafe" in s:
            rating -= random.uniform(0.3, 0.6)
        if "noisy" in s or "street" in s:
            rating -= random.uniform(0.1, 0.3)
        if "excellent" in s or "amazing" in s:
            rating += random.uniform(0.1, 0.3)

    return max(2.8, min(5.0, round(rating, 1)))


def main():
    hotels = []
    NUM_HOTELS = 100
    used_names = set()

    for i in range(1, NUM_HOTELS + 1):
        snippets = generate_review_snippets()
        rating = compute_rating(snippets)

        price_numeric = random.choice([1, 2, 3])
        price = "$" * price_numeric

        hotel = {
            "id": f"hotel_{i}",
            "name": generate_hotel_name(used_names),
            "location": "Amsterdam",
            "neighborhood": random.choice(NEIGHBORHOODS),
            "distance_to_center_km": round(random.uniform(0.4, 6.0), 1),
            "rating": rating,
            "price": price,
            "price_numeric": price_numeric,
            "amenities": generate_amenities(),
            "review_snippets": snippets
        }

        used_names.add(hotel["name"])
        hotels.append(hotel)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, "hotels_synth.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hotels, f, indent=2)

    print(f"Generated synthetic dataset, saved to: {out_path}")

if __name__ == "__main__":
    main()
