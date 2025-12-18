
"""
Pre-designed initial histories to start conversations.

Each entry:
- id:                 a short identifier for logging
- persona: "minimalist" or "explorer"
- location:           city to use (currently always "Amsterdam")
- messages:           list[
    {
        "role": "user" | "assistant",
        "content": str
    }
]
"""

INITIAL_HISTORIES = [
        {
        "id": "min_walkable_transport_hub",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I need a reasonably priced hotel in Amsterdam that's within walking distance "
                    "of public transport. Clean, quiet, and practical is all I care about."
                ),
            }
        ],
    },
    {
        "id": "min_short_stay_no_frills",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm staying just one or two nights in Amsterdam and want a no-frills hotel "
                    "that's reliable and not overpriced. Any suggestions?"
                ),
            }
        ],
    },
    {
        "id": "min_good_reviews_under_budget",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Can you recommend a hotel in Amsterdam with consistently good reviews "
                    "that stays within a reasonable budget? I don't need luxury."
                ),
            }
        ],
    },
    {
        "id": "min_simple_business_stopover",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm stopping in Amsterdam briefly for work and need a straightforward hotel "
                    "that's quiet, dependable, and close enough to get around easily."
                ),
            }
        ],
    },
    {
        "id": "min_sleep_only_priority",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I mostly just need a place to sleep in Amsterdam. As long as it's safe, clean, "
                    "and not too expensive, that's fine."
                ),
            }
        ],
    },
    {
        "id": "min_central_budget_wifi_3n",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I need a hotel in central Amsterdam for 3 nights next month, "
                    "under 150 euros per night, with good reviews and reliable Wi-Fi. "
                    "What would you recommend?"
                ),
            }
        ],
    },
    {
        "id": "min_quiet_near_center",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Please recommend a quiet hotel in Amsterdam within walking distance "
                    "of the city center, with at least a 4.0 rating and not too expensive."
                ),
            }
        ],
    },
    {
        "id": "min_one_night_late_arrival",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm arriving late tonight and just need a clean, safe hotel in Amsterdam "
                    "for one night, reasonably close to the center. Any simple, good-value options?"
                ),
            }
        ],
    },
    {
        "id": "min_business_trip",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm in Amsterdam for a short business trip and need a hotel with good Wi-Fi "
                    "and a desk, not too fancy and not too far from the center. What do you suggest?"
                ),
            }
        ],
    },
    {
        "id": "min_just_the_cheapest_ok",
        "persona": "minimalist",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Just give me the cheapest acceptable hotel in Amsterdam with decent reviews. "
                    "I only care that it's not terrible and I can sleep."
                ),
            }
        ],
    },

    {
        "id": "exp_artsy_creative_vibe",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm looking for a hotel in Amsterdam with a creative or artsy feel, somewhere "
                    "that has character and good reviews rather than a generic chain vibe."
                ),
            }
        ],
    },
    {
        "id": "exp_green_relaxed_area",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Can you suggest a hotel in Amsterdam that feels calm and green, maybe with "
                    "a garden or relaxed atmosphere, but still connected to the city?"
                ),
            }
        ],
    },
    {
        "id": "exp_unique_small_hotel",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I prefer smaller, unique hotels over big ones. Is there a well-reviewed "
                    "hotel in Amsterdam that feels more personal and distinctive?"
                ),
            }
        ],
    },
    {
        "id": "exp_neighborhood_exploration_base",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I want a hotel in Amsterdam that's a good base for exploring different "
                    "neighborhoods and getting a feel for local life. Any recommendations?"
                ),
            }
        ],
    },
    {
        "id": "exp_comfort_plus_atmosphere",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm looking for a hotel in Amsterdam that balances comfort with atmosphere — "
                    "somewhere pleasant to come back to after a full day of exploring."
                ),
            }
        ],
    },
    {
        "id": "exp_canal_view_atmosphere",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm looking for a small, atmospheric hotel in Amsterdam with a nice canal view "
                    "and a cozy vibe. I care more about character than price as long as it's not extreme. "
                    "What would you recommend?"
                ),
            }
        ],
    },
    {
        "id": "exp_local_neighborhood_feel",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Can you recommend a hotel in Amsterdam in a neighborhood that feels local and "
                    "not too touristy, with good reviews and some interesting cafés or bars nearby?"
                ),
            }
        ],
    },
    {
        "id": "exp_romantic_weekend",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm planning a romantic weekend in Amsterdam and would like a hotel with a nice "
                    "view, comfortable rooms, and a special atmosphere. Price is moderate-to-high, "
                    "but I want it to feel worth it. Any ideas?"
                ),
            }
        ],
    },
    {
        "id": "exp_spa_relax_trip",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'd love a hotel in Amsterdam where I can really relax, ideally with a garden or "
                    "spa feeling, away from the loudest streets but still not too far from the center. "
                    "What would you suggest?"
                ),
            }
        ],
    },
    {
        "id": "exp_social_budget_hostel_vibe",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm a social traveler on a budget and I don't mind smaller rooms. Can you suggest "
                    "a place in Amsterdam with a friendly, social atmosphere and good reviews?"
                ),
            }
        ],
    },
    {
        "id": "exp_foodie_hotel_plus_restaurants",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm a foodie visiting Amsterdam and I want a hotel in an area with great restaurants "
                    "and cafés within walking distance. The hotel should be well-reviewed and not the absolute "
                    "cheapest. Any recommendations?"
                ),
            }
        ],
    },
    {
        "id": "exp_long_stay_explore_city",
        "persona": "explorer",
        "location": "Amsterdam",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm staying in Amsterdam for a week and want a comfortable base to explore the city. "
                    "I'd like a nice atmosphere, good reviews, and easy access to public transport. "
                    "Which hotel would fit that?"
                ),
            }
        ],
    },
]