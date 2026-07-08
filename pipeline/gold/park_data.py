"""
pipeline/gold/park_data.py
===========================
Static stadium reference data for the HR-prop Gold layer:

    - HR park factors (100 = neutral), overall and by batter handedness
    - Stadium coordinates (weather ingestion grid point)
    - Center-field compass bearing from home plate (wind-out projection)
    - Roof type (weather features are damped for retractable/dome parks)

⚠ SEED DATA: values are approximations of published 3-year Statcast park
factors and stadium geometry, good enough to develop and validate the
pipeline. Refresh from a verified source (Baseball Savant park factors,
surveyed stadium bearings) before betting real money on model output.

Keying is by Statcast `home_team` abbreviation plus season, because two
franchises changed venues in 2025 (Athletics → Sutter Health Park as "ATH",
Rays → George M. Steinbrenner Field after the Tropicana hurricane damage).
"""

from __future__ import annotations

import pandas as pd

# roof: "open" | "retractable" | "dome"
# cf_bearing_deg: compass bearing (deg from true north) home plate → center field
_PARK_FIELDS = ["team_abbr", "venue_name", "lat", "lon", "cf_bearing_deg", "roof",
                "park_hr_factor", "park_hr_factor_lhb", "park_hr_factor_rhb"]

_PARKS = [
    ("AZ",  "Chase Field",                33.45, -112.07,   0, "retractable", 101, 100, 102),
    ("ATL", "Truist Park",                33.89,  -84.47, 135, "open",        104, 102, 106),
    ("BAL", "Oriole Park at Camden Yards",39.28,  -76.62,  31, "open",         96, 104,  88),
    ("BOS", "Fenway Park",                42.35,  -71.10,  52, "open",         90,  96,  86),
    ("CHC", "Wrigley Field",              41.95,  -87.66,  35, "open",        102, 100, 104),
    ("CWS", "Rate Field",                 41.83,  -87.63, 127, "open",        110, 108, 112),
    ("CIN", "Great American Ball Park",   39.10,  -84.51, 120, "open",        128, 126, 130),
    ("CLE", "Progressive Field",          41.50,  -81.69,   0, "open",         95,  97,  93),
    ("COL", "Coors Field",                39.76, -104.99,   3, "open",        108, 106, 110),
    ("DET", "Comerica Park",              42.34,  -83.05, 150, "open",         88,  90,  86),
    ("HOU", "Daikin Park",                29.76,  -95.36, 345, "retractable", 106, 100, 112),
    ("KC",  "Kauffman Stadium",           39.05,  -94.48,  45, "open",         85,  84,  86),
    ("LAA", "Angel Stadium",              33.80, -117.88,  65, "open",        104, 102, 106),
    ("LAD", "Dodger Stadium",             34.07, -118.24,  26, "open",        114, 112, 116),
    ("MIA", "loanDepot park",             25.78,  -80.22,  40, "retractable",  88,  87,  89),
    ("MIL", "American Family Field",      43.03,  -87.97, 130, "retractable", 112, 114, 110),
    ("MIN", "Target Field",               44.98,  -93.28,  90, "open",         95,  94,  96),
    ("NYM", "Citi Field",                 40.76,  -73.85,  30, "open",         97,  96,  98),
    ("NYY", "Yankee Stadium",             40.83,  -73.93,  75, "open",        117, 128, 106),
    ("OAK", "Oakland Coliseum",           37.75, -122.20,  60, "open",         87,  86,  88),
    ("PHI", "Citizens Bank Park",         39.91,  -75.17,   9, "open",        112, 110, 114),
    ("PIT", "PNC Park",                   40.45,  -80.01, 115, "open",         88,  86,  90),
    ("SD",  "Petco Park",                 32.71, -117.16,   0, "open",         96,  94,  98),
    ("SEA", "T-Mobile Park",              47.59, -122.33,  49, "retractable", 104, 102, 106),
    ("SF",  "Oracle Park",                37.78, -122.39,  85, "open",         82,  75,  88),
    ("STL", "Busch Stadium",              38.62,  -90.19,  62, "open",         91,  90,  92),
    ("TB",  "Tropicana Field",            27.77,  -82.65,  45, "dome",         95,  94,  96),
    ("TEX", "Globe Life Field",           32.75,  -97.08,  15, "retractable", 103, 102, 104),
    ("TOR", "Rogers Centre",              43.64,  -79.39, 345, "retractable", 106, 108, 104),
    ("WSH", "Nationals Park",             38.87,  -77.01,  87, "open",        100,  99, 101),
]

# Season-specific venue overrides: (team_abbr, first_season, last_season, park row)
_OVERRIDES = [
    ("ATH", 2025, 2027,
     ("ATH", "Sutter Health Park",        38.58, -121.51,  85, "open",        105, 106, 104)),
    ("TB",  2025, 2027,
     ("TB",  "George M. Steinbrenner Field", 27.98, -82.51, 75, "open",       110, 118, 102)),
]


def get_parks_df(seasons: list[int]) -> pd.DataFrame:
    """
    One row per (team_abbr, season) so DuckDB can join on
    home_team + YEAR(game_date). Overrides replace the default venue for the
    seasons they cover; "ATH" only exists as an override (the franchise used
    "OAK" through 2024).
    """
    rows = []
    for season in seasons:
        for park in _PARKS:
            rows.append((*park, season))
        for abbr, first, last, park in _OVERRIDES:
            if first <= season <= last:
                rows = [r for r in rows
                        if not (r[0] == abbr and r[-1] == season)]
                rows.append((*park, season))
    return pd.DataFrame(rows, columns=[*_PARK_FIELDS, "season"])


def register_parks_table(con, seasons: list[int]) -> None:
    """Create/replace the DuckDB `parks` reference table."""
    parks_df = get_parks_df(seasons)  # noqa: F841 — referenced by DuckDB below
    con.execute("CREATE OR REPLACE TABLE parks AS SELECT * FROM parks_df")
