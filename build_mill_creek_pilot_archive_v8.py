#!/usr/bin/env python3
"""
build_mill_creek_pilot_archive_v8.py

One-file Mill Creek archive pilot builder, v8.

V8 focuses on corpus-quality improvements learned from Levels 9-13:
- richer title/source variation
- source_group labels for student ML labs
- event bursts around flood/redevelopment/memory years
- more displacement and school-memory artifacts
- fewer formulaic newspaper/photo/religion templates
- stronger manifest-side active-date protections

Default behavior:
- Makes pilot manifest files.
- Generates 10 artifacts in each story category:
    newspaper
    school
    religion
    business
    photo_caption
    oral_history
    council_minutes
- Writes detailed timing metrics and category summaries.
- Writes a simple gazetteer/map-data file without calling the model.

This is designed for a lunch-length or dinner-length timing run, not necessarily the final 5,000+ artifact build.

Example:

    cd /home/darin/PycharmProjects/HumantiesBook/MillCreek

    python build_mill_creek_pilot_archive.py

Useful options:

    python build_mill_creek_pilot_archive.py --per-category 10

    python build_mill_creek_pilot_archive.py --per-category 10 --max-retries 1 --retry-on-validation

    python build_mill_creek_pilot_archive.py --dry-run

    python build_mill_creek_pilot_archive.py --categories newspaper school religion

Notes:
- Requires Ollama running locally at http://localhost:11434.
- Uses gemma4:26b by default.
- Uses top-level "think": False to keep output in message.content.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_MODEL = "gemma4:26b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_OPTIONS = {
    "temperature": 0.62,
    "top_p": 0.88,
    "num_ctx": 8192,
    "num_predict": 1400,
    "repeat_penalty": 1.08,
}

CATEGORIES = [
    "newspaper",
    "school",
    "religion",
    "business",
    "photo_caption",
    "oral_history",
    "council_minutes",
]

FINAL_ARCHIVE_TARGETS = {
    "newspaper": 5000,
    "school": 500,
    "religion": 500,
    "business": 800,
    "photo_caption": 400,
    "oral_history": 150,
    "council_minutes": 300,
}


GLOBAL_CONTEXT = """
Mill Creek is a fictional prairie river town in Cobberland.

Core geography:
- Mill Creek flows into the Prairie River near Old Mill Bend.
- The main districts are Downtown/Main Street, the Depot District, North Orchard,
  South Flats, West Rows farms, and the Riverfront Trail.
- The railroad and grain elevator shaped early town life.
- Flooding is a recurring civic problem.

Core institutions:
- Mill Creek Chronicle: founded in 1904.
- Mill Creek Herald: later successor paper, used after the Chronicle era.
- St. Ansgar Lutheran Church.
- Sacred Heart Catholic Mission.
- Mill Creek Women’s Aid Society.
- Mill Creek Commercial Club.
- Mill Creek City Council.
- Mill Creek Public Schools.
- Mill Creek Historical Society.
- CobberTech Extension Center.
- Olotón Foods.
- Prairie River Clinic.
- Cobberland County Planning Commission.
- Cobberland Future Business Leaders Association.

Important recurring themes:
river and flood, downtown change, agricultural life, public memory,
church and civic service, schools, local business, heritage and belonging.

Do not mention that Mill Creek is fictional, simulated, generated, or part of a textbook.
"""


CATEGORY_INSTRUCTIONS = {
    "newspaper": """
Write a small-town newspaper artifact from the assigned year. Use plain newspaper prose,
not a historical essay or a short story. Avoid modern academic language.
""",
    "school": """
Write a school archive artifact. Depending on the source type, this may be a yearbook
caption, school newspaper note, sports summary, club report, program note, or student
profile. Keep it historically appropriate to the assigned year.
""",
    "religion": """
Write a religious-life archive artifact. Depending on the source type, this may be a
church bulletin notice, sermon excerpt, choir program note, service-project notice, or
community religious announcement. Keep it respectful, concrete, and appropriate to the year.
""",
    "business": """
Write a business or public-notice archive artifact. Depending on the source type, this may be
an advertisement, classified notice, market report, public notice, zoning notice, or local
business report. Keep it ordinary and archive-like.
""",
    "photo_caption": """
Write a photo-caption, cutline, object description, or exhibit-label artifact. It should be
concise, concrete, and useful as archive metadata.
""",
    "oral_history": """
Write an oral-history transcript excerpt. Use interviewer and speaker turns. It should sound
like remembered speech, not a polished essay. The speaker may be uncertain about small details.
""",
    "council_minutes": """
Write city council minutes. Use formal but plain municipal record language. Include call to
order, attendance, agenda/action items, motions or outcomes, and adjournment.
""",
}


FORBIDDEN_PHRASES = [
    "First Baptist",
    "First Methodist",
    "Lions Club",
    "stakeholders",
    "food insecurity",
    "underserved",
    "intentional community",
    "place-making",
    "data-driven",
    "language model",
    "fictional",
    "simulated",
    "as an artificial",
    "generated by",
    "as a model",
    "textbook",
    "Markdown",
    "###",
    "**",
]


# ---------------------------------------------------------------------
# Timeline / world rules
# ---------------------------------------------------------------------

ACTIVE_INSTITUTIONS: dict[str, tuple[int, int | None]] = {
    "Mill Creek Chronicle": (1904, 1987),
    "Mill Creek Herald": (1988, None),
    "Mill Creek City Council": (1874, None),
    "Mill Creek Public Schools": (1902, None),
    "Mill Creek High School": (1928, None),
    "St. Ansgar Lutheran Church": (1886, None),
    "Sacred Heart Catholic Mission": (1898, None),
    "Mill Creek Women’s Aid Society": (1908, 1965),
    "Mill Creek Commercial Club": (1910, 1980),
    "Prairie River Clinic": (1954, None),
    "Cobberland County Planning Commission": (1960, None),
    "Mill Creek Historical Society": (1988, None),
    "CobberTech Extension Center": (1998, None),
    "Cobberland Future Business Leaders Association": (1980, None),
    "Olotón Foods": (2001, None),
}

ACTIVE_LOCATIONS: dict[str, tuple[int, int | None]] = {
    "Old Mill Bend": (1874, None),
    "Prairie River bridge": (1893, None),
    "Downtown/Main Street": (1879, None),
    "Depot District": (1879, None),
    "South Flats": (1900, None),
    "North Orchard": (1948, None),
    "West Rows farms": (1874, None),
    "grain elevator": (1882, None),
    "Old Grange Hall": (1908, None),
    "Main Street reading room": (1915, 1945),
    "St. Ansgar Lutheran Church": (1886, None),
    "Sacred Heart Catholic Mission": (1898, None),
    "Mill Creek Public Schools": (1902, None),
    "Mill Creek High School": (1928, None),
    "Memorial Field": (1948, None),
    "City Hall": (1910, None),
    "Council Chambers, City Hall": (1910, None),
    "Fire Hall": (1900, 1960),
    "Prairie River Clinic": (1954, None),
    "Mill Creek Historical Society": (1988, None),
    "restored depot room": (1988, None),
    "Olotón Foods": (2001, None),
    "CobberTech Extension Center": (1998, None),
    "Riverfront Trail": (2022, None),
}


# These are not biographies. They are conservative "may be referenced in an archive
# artifact from this year" windows. The goal is to prevent obvious problems such as
# Maizey appearing as an adult farm/business figure before her public 1996 student award.
#
# Names not listed here fall back to the era pools in people_for_year().
PERSON_REFERENCE_WINDOWS: dict[str, tuple[int, int | None]] = {
    "Mayor Nels Hovland": (1900, 1939),
    "Clerk Elsie Bratten": (1900, 1939),
    "Councilman Peter Lunde": (1900, 1939),
    "Councilman Martin Kvale": (1900, 1939),
    "Councilwoman Clara Hestvik": (1910, 1945),
    "Street Commissioner Olaf Rask": (1900, 1939),
    "Clara Hestvik": (1910, 1945),
    "Martin Kvale": (1900, 1975),
    "Mrs. Anna Soren": (1910, 1945),

    "Rev. Anders Nygaard": (1910, 1968),
    "Father Thomas Berrigan": (1910, 1968),

    "Mayor Ingrid Lunde": (1940, 1965),
    "Clerk Ruth Ellingson": (1945, 1975),
    "Ruth Ellingson": (1945, 1995),
    "Principal Ruth Ellingson": (1960, 2000),
    "Dr. Helen Markham": (1954, 1995),
    "Public Works Foreman Carl Voss": (1945, 1975),
    "Coach Harold Bratten": (1940, 1995),
    "Councilwoman Lena Voss": (1950, 1989),
    "Lena Voss": (1950, 2022),
    "Councilman Peter Harlan": (1950, 1980),
    "Peter Harlan": (1950, 2000),

    "Mayor Edwin Rask": (1966, 1989),
    "Clerk Janine Roberts": (1975, 2010),
    "Janine Roberts": (1975, 2015),
    "County Engineer Paul Decker": (1960, 1985),
    "Professor Carrel Englekorn": (1975, 2026),
    "Councilman David Harlan": (1988, 2026),
    "David Harlan": (1988, 2026),
    "Councilwoman Asha Patel": (2000, 2026),
    "Asha Patel": (2000, 2026),
    "Mayor Naomi Reyes": (1990, 2026),
    "Clerk Asha Patel": (2011, 2026),
    "Rosa Martinez": (2011, 2026),

    # Maizey is public as a student in 1996, then as an entrepreneur from 2001 onward.
    # More detailed role logic belongs in the manifest rows, but this prevents 1990 Maizey.
    "Maizey Olotón": (1996, 2026),

    # Nora should only appear in contemporary digitization/student archive contexts.
    "Nora Reyes": (2022, 2026),
}


def newspaper_for_year(year: int) -> str:
    return "Mill Creek Chronicle" if year < 1988 else "Mill Creek Herald"


def is_active(name: str, year: int, active_map: dict[str, tuple[int, int | None]]) -> bool:
    if name not in active_map:
        return True
    start, end = active_map[name]
    return year >= start and (end is None or year <= end)


def active_filter(names: list[str], year: int, active_map: dict[str, tuple[int, int | None]]) -> list[str]:
    out: list[str] = []
    for name in names:
        if is_active(name, year, active_map) and name not in out:
            out.append(name)
    return out


def person_reference_allowed(name: str, year: int) -> bool:
    """Return True if this named person can plausibly be referenced in an artifact from year."""
    if name not in PERSON_REFERENCE_WINDOWS:
        return True
    start, end = PERSON_REFERENCE_WINDOWS[name]
    return year >= start and (end is None or year <= end)


def era_for_year(year: int) -> str:
    if year <= 1939:
        return "early_civic"
    if year <= 1965:
        return "war_postwar"
    if year <= 1989:
        return "transition"
    if year <= 2010:
        return "new_voices"
    return "contemporary"


def people_for_year(year: int) -> list[str]:
    """Return a conservative pool of people who can be used in artifacts from this year.

    This is intentionally not a full biography engine. Specific manifest rows can still
    supply particular people, but row() will filter those through PERSON_REFERENCE_WINDOWS.
    """
    people: list[str] = []

    if year <= 1939:
        people += [
            "Mayor Nels Hovland", "Clerk Elsie Bratten", "Councilman Peter Lunde",
            "Councilman Martin Kvale", "Councilwoman Clara Hestvik",
            "Street Commissioner Olaf Rask", "Clara Hestvik", "Martin Kvale",
            "Mrs. Anna Soren", "Rev. Anders Nygaard", "Father Thomas Berrigan",
        ]

    if 1930 <= year <= 1965:
        people += [
            "Mayor Ingrid Lunde", "Clerk Ruth Ellingson", "Ruth Ellingson",
            "Dr. Helen Markham", "Public Works Foreman Carl Voss",
            "Coach Harold Bratten", "Councilwoman Lena Voss", "Lena Voss",
            "Councilman Peter Harlan", "Peter Harlan",
            "Rev. Anders Nygaard", "Father Thomas Berrigan",
        ]

    if 1966 <= year <= 1989:
        people += [
            "Mayor Edwin Rask", "Clerk Ruth Ellingson", "Ruth Ellingson",
            "Clerk Janine Roberts", "Janine Roberts", "County Engineer Paul Decker",
            "Lena Voss", "Professor Carrel Englekorn", "Councilman David Harlan",
            "David Harlan", "Councilwoman Lena Voss", "Martin Kvale",
        ]

    if 1990 <= year <= 2010:
        people += [
            "Mayor Naomi Reyes", "Clerk Janine Roberts", "Janine Roberts",
            "Councilman David Harlan", "David Harlan", "Councilwoman Asha Patel",
            "Asha Patel", "Professor Carrel Englekorn", "Lena Voss",
            "Principal Ruth Ellingson", "Ruth Ellingson", "Rosa Martinez",
        ]
        if year >= 1996:
            people.append("Maizey Olotón")

    if year >= 2011:
        people += [
            "Mayor Naomi Reyes", "Clerk Asha Patel", "Councilman David Harlan",
            "David Harlan", "Councilwoman Lena Voss", "Professor Carrel Englekorn",
            "Asha Patel", "Lena Voss", "Maizey Olotón", "Rosa Martinez",
        ]
        if year >= 2022:
            people.append("Nora Reyes")

    out: list[str] = []
    seen: set[str] = set()
    for p in people:
        if p not in seen and person_reference_allowed(p, year):
            out.append(p)
            seen.add(p)
    return out




def source_group_for(category: str, source_type: str) -> str:
    """Broader source grouping for student-facing ML labs."""
    st = (source_type or "").lower()
    cat = (category or "").lower()
    if cat == "oral_history" or "oral" in st or "transcript" in st:
        return "oral_history"
    if cat == "photo_caption" or "photo" in st or "caption" in st or "exhibit" in st:
        return "photo_archive"
    if cat == "council_minutes" or "minutes" in st:
        return "minutes_government"
    if cat == "religion" or "sermon" in st or "church" in st or "service" in st:
        return "religious_community"
    if cat == "school" or "school" in st or "club" in st or "sports" in st or "yearbook" in st:
        return "school_record"
    if "notice" in st or "classified" in st or "advertisement" in st or "public" in st:
        return "notice_ad"
    if "market" in st or "business" in st or "price" in st:
        return "business_record"
    if cat == "newspaper" or "feature" in st or "report" in st or "article" in st:
        return "newspaper_article"
    return "other"


def row(
    *,
    category: str,
    artifact_id: str,
    year: int,
    date_label: str,
    collection: str,
    source_type: str,
    title_seed: str,
    topic: str,
    location: str,
    institutions: list[str],
    people: list[str],
    locations: list[str],
    required_details: list[str],
    tone: str,
    word_count_min: int,
    word_count_max: int,
    min_institutions: int = 1,
    min_people: int = 1,
    min_locations: int = 1,
) -> dict[str, Any]:
    newspaper = newspaper_for_year(year)

    # Only newspaper artifacts automatically include the newspaper as an allowed institution.
    # Other categories should not be nudged into mentioning the Herald/Chronicle unless
    # the row explicitly supplies it. This prevents oral histories about pre-1945 memories
    # from casually inserting the later Herald simply because the interview date is 1988.
    inst_seed = ([newspaper] + institutions) if category == "newspaper" else list(institutions)
    inst = active_filter(inst_seed, year, ACTIVE_INSTITUTIONS)
    locs = active_filter(locations + [location], year, ACTIVE_LOCATIONS)

    if not is_active(location, year, ACTIVE_LOCATIONS):
        location = locs[0] if locs else "Downtown/Main Street"

    if not inst:
        inst = [newspaper] if category == "newspaper" else ["Mill Creek City Council"]
        inst = active_filter(inst, year, ACTIVE_INSTITUTIONS)
    if location not in locs:
        locs.insert(0, location)

    allowed_people = []

    # Preserve explicitly supplied people. For most artifacts they must be
    # plausible in the artifact year. For oral histories, however, explicitly
    # supplied names may be remembered historical figures from an earlier period.
    for p in people:
        if (category == "oral_history" or person_reference_allowed(p, year)) and p not in allowed_people:
            allowed_people.append(p)

    # Then fill out the prompt with a few era-appropriate names.
    for p in people_for_year(year):
        if len(allowed_people) >= 4:
            break
        if p not in allowed_people:
            allowed_people.append(p)

    return {
        "artifact_id": artifact_id,
        "category": category,
        "artifact_kind": category,
        "year": year,
        "era": era_for_year(year),
        "date": date_label,
        "collection": collection,
        "source_type": source_type,
        "source_group": source_group_for(category, source_type),
        "title_seed": title_seed,
        "topic": topic,
        "location": location,
        "allowed_institutions": inst,
        "allowed_people": allowed_people,
        "allowed_locations": locs,
        "required_details": required_details,
        "tone": tone,
        "word_count_min": word_count_min,
        "word_count_max": word_count_max,
        "min_institutions": min_institutions,
        "min_people": min_people,
        "min_locations": min_locations,
    }


# ---------------------------------------------------------------------
# Pilot manifests: 10 per category
# ---------------------------------------------------------------------

def newspaper_rows() -> list[dict[str, Any]]:
    return [
        row(category="newspaper", artifact_id="NEWS_1922_001", year=1922, date_label="Thursday, June 8, 1922",
            collection="Mill Creek Newspaper Archive", source_type="local news article",
            title_seed="Women’s Aid Society Opens Reading Room Fund Drive",
            topic="reading room fund drive", location="Main Street reading room",
            institutions=["Mill Creek Women’s Aid Society", "Mill Creek Commercial Club"],
            people=["Clara Hestvik", "Martin Kvale", "Mrs. Anna Soren"],
            locations=["Main Street reading room", "Downtown/Main Street"],
            required_details=["fund drive for books or furnishings", "specific dollar amount or donation", "public meeting date or time"],
            tone="plain civic reporting with modest optimism", word_count_min=250, word_count_max=480),

        row(category="newspaper", artifact_id="NEWS_1934_001", year=1934, date_label="Thursday, October 18, 1934",
            collection="Mill Creek Newspaper Archive", source_type="community notice with reporting",
            title_seed="Churches and Aid Society Plan Harvest Relief Supper",
            topic="Depression relief supper", location="Old Grange Hall",
            institutions=["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Mill Creek Women’s Aid Society"],
            people=["Rev. Anders Nygaard", "Father Thomas Berrigan", "Clara Hestvik"],
            locations=["Old Grange Hall", "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission"],
            required_details=["Old Grange Hall", "requested supplies such as flour, sugar, canned goods, potatoes, or dry goods", "ticket price or donation instruction"],
            tone="small-town Depression-era article, concrete and practical", word_count_min=300, word_count_max=520),

        row(category="newspaper", artifact_id="NEWS_1948_001", year=1948, date_label="Friday, September 10, 1948",
            collection="Mill Creek Newspaper Archive", source_type="dedication report",
            title_seed="Memorial Field Dedicated Before Opening Game",
            topic="postwar public memory and school identity", location="Memorial Field",
            institutions=["Mill Creek Public Schools", "Mill Creek City Council"],
            people=["Mayor Ingrid Lunde", "Coach Harold Bratten", "Rev. Anders Nygaard"],
            locations=["Memorial Field", "Mill Creek Public Schools"],
            required_details=["dedication before an opening game", "plaque, flag, band, seating, or ceremony detail", "Memorial Field"],
            tone="respectful postwar civic reporting", word_count_min=280, word_count_max=500),

        row(category="newspaper", artifact_id="NEWS_1957_001", year=1957, date_label="Monday, April 22, 1957",
            collection="Mill Creek Newspaper Archive", source_type="emergency report",
            title_seed="River Crests Below 1893 Mark; Council Reviews Levee Question",
            topic="1957 flood and flood-control debate", location="Prairie River bridge",
            institutions=["Mill Creek City Council", "Prairie River Clinic", "Mill Creek Public Schools"],
            people=["Mayor Ingrid Lunde", "Dr. Helen Markham", "Public Works Foreman Carl Voss"],
            locations=["Prairie River bridge", "South Flats", "Prairie River Clinic"],
            required_details=["Prairie River bridge", "South Flats or bridge approaches", "boil-water, clinic, sandbags, inspection, or levee estimate"],
            tone="measured emergency reporting, practical and local", word_count_min=300, word_count_max=540),

        row(category="newspaper", artifact_id="NEWS_1968_001", year=1968, date_label="Wednesday, April 17, 1968",
            collection="Mill Creek Newspaper Archive", source_type="planning report",
            title_seed="Bypass Route Draws Questions From Main Street Merchants",
            topic="highway bypass and downtown anxiety", location="Downtown/Main Street",
            institutions=["Mill Creek City Council", "Mill Creek Commercial Club", "Cobberland County Planning Commission"],
            people=["Mayor Edwin Rask", "Martin Kvale", "Ruth Ellingson", "County Engineer Paul Decker"],
            locations=["Downtown/Main Street", "City Hall"],
            required_details=["proposed highway bypass route", "Main Street merchants", "traffic count, public comment date, map, or county estimate"],
            tone="late-1960s local planning report, balanced but concrete", word_count_min=320, word_count_max=560),

        row(category="newspaper", artifact_id="NEWS_1988_001", year=1988, date_label="Sunday, May 15, 1988",
            collection="Mill Creek Newspaper Archive", source_type="community feature",
            title_seed="Historical Society Opens First Exhibit in Restored Depot Room",
            topic="public memory and historical preservation", location="Depot District",
            institutions=["Mill Creek Historical Society", "Mill Creek City Council", "Mill Creek Public Schools"],
            people=["Professor Carrel Englekorn", "Ruth Ellingson", "Lena Voss"],
            locations=["Depot District", "restored depot room", "Mill Creek Historical Society"],
            required_details=["restored depot room", "first exhibit", "rail lantern, school photograph, flood map, depot ledger, or grain ticket"],
            tone="1980s community feature, warm but not sentimental", word_count_min=320, word_count_max=580),

        row(category="newspaper", artifact_id="NEWS_1996_001", year=1996, date_label="Friday, May 3, 1996",
            collection="Mill Creek Newspaper Archive", source_type="student profile",
            title_seed="Local Student’s Big Idea Wins State Entrepreneurship Award",
            topic="Maizey Olotón entrepreneurship award", location="Mill Creek High School",
            institutions=["Mill Creek Public Schools", "Cobberland Future Business Leaders Association"],
            people=["Maizey Olotón", "David Harlan", "Principal Ruth Ellingson"],
            locations=["Mill Creek High School", "Mill Creek Public Schools"],
            required_details=["state entrepreneurship award", "school presentation or contest detail", "one restrained corn-related pun or no pun at all"],
            tone="local achievement reporting, grounded, with at most one light corn pun", word_count_min=260, word_count_max=560),

        row(category="newspaper", artifact_id="NEWS_2009_001", year=2009, date_label="Wednesday, September 23, 2009",
            collection="Mill Creek Newspaper Archive", source_type="redevelopment report",
            title_seed="Depot District Plan Released for Public Review",
            topic="Depot District redevelopment plan", location="Depot District",
            institutions=["Mill Creek City Council", "Mill Creek Historical Society", "Olotón Foods"],
            people=["Mayor Naomi Reyes", "Professor Carrel Englekorn", "Maizey Olotón"],
            locations=["Depot District", "Mill Creek Historical Society", "City Hall"],
            required_details=["Depot District redevelopment plan", "public hearing date or review period", "rail-era structure or preservation detail"],
            tone="modern municipal reporting, concrete but not jargon-heavy", word_count_min=320, word_count_max=620),

        row(category="newspaper", artifact_id="NEWS_2022_001", year=2022, date_label="Sunday, June 19, 2022",
            collection="Mill Creek Newspaper Archive", source_type="modern local report",
            title_seed="Riverfront Trail Opens Along Former Flood Buyout Land",
            topic="Riverfront Trail opening and memory of South Flats", location="Riverfront Trail",
            institutions=["Mill Creek City Council", "Mill Creek Historical Society", "CobberTech Extension Center"],
            people=["Mayor Naomi Reyes", "Professor Carrel Englekorn", "Lena Voss", "Asha Patel"],
            locations=["Riverfront Trail", "South Flats", "Mill Creek Historical Society", "CobberTech Extension Center"],
            required_details=["former South Flats buyout land", "Riverfront Trail opening", "historical sign, photo digitization, creek access, or public ceremony detail"],
            tone="modern local journalism, concrete and restrained", word_count_min=330, word_count_max=600),

        row(category="newspaper", artifact_id="NEWS_2025_001", year=2025, date_label="Friday, February 14, 2025",
            collection="Mill Creek Newspaper Archive", source_type="digital archive report",
            title_seed="CobberTech Students Help Historical Society Scan Newspaper Clippings",
            topic="digital archive lab", location="CobberTech Extension Center",
            institutions=["CobberTech Extension Center", "Mill Creek Historical Society"],
            people=["Nora Reyes", "Professor Carrel Englekorn", "Asha Patel"],
            locations=["CobberTech Extension Center", "Mill Creek Historical Society"],
            required_details=["scanning photographs or newspaper clippings", "metadata, indexing, or search detail", "digital tools help but do not interpret by themselves"],
            tone="student-centered archive reporting, concrete and thoughtful", word_count_min=300, word_count_max=650),
    ]


def school_rows() -> list[dict[str, Any]]:
    return [
        row(category="school", artifact_id="SCH_1922_001", year=1922, date_label="Friday, May 19, 1922",
            collection="Mill Creek Public Schools Archive", source_type="school newspaper note",
            title_seed="Public School Pupils Give Spring Program",
            topic="spring program at public school", location="Mill Creek Public Schools",
            institutions=["Mill Creek Public Schools"], people=["Clara Hestvik", "Clerk Elsie Bratten"],
            locations=["Mill Creek Public Schools"], required_details=["spring program", "classroom or assembly room detail", "student recitation, song, or program order"],
            tone="early school note, plain and local", word_count_min=160, word_count_max=420, min_people=0),

        row(category="school", artifact_id="SCH_1929_001", year=1929, date_label="Thursday, October 10, 1929",
            collection="Mill Creek Public Schools Archive", source_type="school sports summary",
            title_seed="High School Team Opens Season on School Grounds",
            topic="early high school sports", location="Mill Creek High School",
            institutions=["Mill Creek Public Schools", "Mill Creek High School"], people=["Coach Harold Bratten", "Clerk Ruth Ellingson"],
            locations=["Mill Creek High School", "Mill Creek Public Schools"], required_details=["school grounds, school gym, or athletic field", "score, practice, or opponent detail", "no Memorial Field reference"],
            tone="plain school sports reporting", word_count_min=170, word_count_max=420, min_people=0),

        row(category="school", artifact_id="SCH_1948_001", year=1948, date_label="Friday, September 10, 1948",
            collection="Mill Creek Public Schools Archive", source_type="program note",
            title_seed="Band Plays at Memorial Field Dedication",
            topic="band and dedication ceremony", location="Memorial Field",
            institutions=["Mill Creek Public Schools"], people=["Coach Harold Bratten", "Mayor Ingrid Lunde"],
            locations=["Memorial Field", "Mill Creek Public Schools"], required_details=["band performance", "dedication ceremony", "Memorial Field"],
            tone="postwar school program note", word_count_min=160, word_count_max=420),

        row(category="school", artifact_id="SCH_1957_001", year=1957, date_label="Tuesday, April 23, 1957",
            collection="Mill Creek Public Schools Archive", source_type="student service note",
            title_seed="Students Help Fill Sandbags After Classes",
            topic="students and flood response", location="Mill Creek Public Schools",
            institutions=["Mill Creek Public Schools", "Mill Creek City Council"], people=["Public Works Foreman Carl Voss", "Coach Harold Bratten"],
            locations=["Mill Creek Public Schools", "South Flats", "Prairie River bridge"], required_details=["sandbags", "after classes or school schedule", "South Flats or bridge approach"],
            tone="school service report, practical and concrete", word_count_min=180, word_count_max=450),

        row(category="school", artifact_id="SCH_1968_001", year=1968, date_label="Friday, October 4, 1968",
            collection="Mill Creek Public Schools Archive", source_type="student editorial note",
            title_seed="Students Ask What Bypass Will Mean for Main Street",
            topic="student response to bypass discussion", location="Mill Creek High School",
            institutions=["Mill Creek Public Schools", "Mill Creek City Council"], people=["County Engineer Paul Decker", "Ruth Ellingson"],
            locations=["Mill Creek High School", "Downtown/Main Street"], required_details=["bypass discussion", "Main Street", "student classroom or assembly detail"],
            tone="student newspaper note, thoughtful but age-appropriate", word_count_min=180, word_count_max=460),

        row(category="school", artifact_id="SCH_1988_001", year=1988, date_label="Thursday, May 12, 1988",
            collection="Mill Creek Public Schools Archive", source_type="history club report",
            title_seed="History Club Helps Label Photographs for Depot Exhibit",
            topic="students and Historical Society exhibit", location="Mill Creek High School",
            institutions=["Mill Creek Public Schools", "Mill Creek Historical Society"], people=["Professor Carrel Englekorn", "Lena Voss"],
            locations=["Mill Creek High School", "Mill Creek Historical Society", "restored depot room"], required_details=["labeling photographs", "depot exhibit", "student history club work"],
            tone="school archive report, concrete and local", word_count_min=180, word_count_max=460),

        row(category="school", artifact_id="SCH_1996_001", year=1996, date_label="Friday, May 3, 1996",
            collection="Mill Creek Public Schools Archive", source_type="student profile",
            title_seed="Maizey Olotón Receives State Entrepreneurship Award",
            topic="student award and entrepreneurship", location="Mill Creek High School",
            institutions=["Mill Creek Public Schools", "Cobberland Future Business Leaders Association"], people=["Maizey Olotón", "Principal Ruth Ellingson", "David Harlan"],
            locations=["Mill Creek High School"], required_details=["state entrepreneurship award", "student business idea", "school recognition"],
            tone="student profile, proud but grounded", word_count_min=200, word_count_max=500),

        row(category="school", artifact_id="SCH_2009_001", year=2009, date_label="Thursday, October 1, 2009",
            collection="Mill Creek Public Schools Archive", source_type="service-learning note",
            title_seed="Students Study Depot District Plan in Civics Class",
            topic="civics class and redevelopment", location="Mill Creek High School",
            institutions=["Mill Creek Public Schools", "Mill Creek City Council", "Mill Creek Historical Society"], people=["Professor Carrel Englekorn", "Asha Patel"],
            locations=["Mill Creek High School", "Depot District", "City Hall"], required_details=["Depot District plan", "civics class", "public hearing or map detail"],
            tone="modern school archive note", word_count_min=180, word_count_max=480),

        row(category="school", artifact_id="SCH_2022_001", year=2022, date_label="Monday, June 20, 2022",
            collection="Mill Creek Public Schools Archive", source_type="student volunteer note",
            title_seed="Students Gather Trail Opening Photographs for Archive",
            topic="Riverfront Trail and student archive work", location="Riverfront Trail",
            institutions=["Mill Creek Public Schools", "Mill Creek Historical Society", "CobberTech Extension Center"], people=["Nora Reyes", "Asha Patel"],
            locations=["Riverfront Trail", "CobberTech Extension Center", "Mill Creek Historical Society"], required_details=["Riverfront Trail opening", "photographs or captions", "student archive work"],
            tone="student-centered contemporary archive note", word_count_min=180, word_count_max=480),

        row(category="school", artifact_id="SCH_2025_001", year=2025, date_label="Friday, February 14, 2025",
            collection="Mill Creek Public Schools Archive", source_type="digital humanities lab note",
            title_seed="Class Uses Newspaper Clippings to Study Mill Creek Memory",
            topic="students using digital archive", location="CobberTech Extension Center",
            institutions=["Mill Creek Public Schools", "CobberTech Extension Center", "Mill Creek Historical Society"], people=["Nora Reyes", "Professor Carrel Englekorn"],
            locations=["CobberTech Extension Center", "Mill Creek Historical Society"], required_details=["newspaper clippings", "metadata or scanning", "digital tools help but do not interpret by themselves"],
            tone="contemporary school lab note, reflective and concrete", word_count_min=220, word_count_max=520),
    ]


def religion_rows() -> list[dict[str, Any]]:
    return [
        row(category="religion", artifact_id="REL_1922_001", year=1922, date_label="Sunday, March 12, 1922",
            collection="Mill Creek Religious and Community Life Archive", source_type="church bulletin notice",
            title_seed="St. Ansgar Announces Lenten Supper",
            topic="Lenten supper and community collection", location="St. Ansgar Lutheran Church",
            institutions=["St. Ansgar Lutheran Church", "Mill Creek Women’s Aid Society"], people=["Rev. Anders Nygaard", "Clara Hestvik"],
            locations=["St. Ansgar Lutheran Church", "Downtown/Main Street"], required_details=["Lenten supper in March", "food or donation detail", "church basement or meeting-room detail"],
            tone="brief church bulletin notice, concrete and respectful", word_count_min=120, word_count_max=320, min_people=0),

        row(category="religion", artifact_id="REL_1934_001", year=1934, date_label="Sunday, October 14, 1934",
            collection="Mill Creek Religious and Community Life Archive", source_type="service project notice",
            title_seed="Churches Ask for Harvest Relief Supper Supplies",
            topic="Depression relief collection", location="Old Grange Hall",
            institutions=["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Mill Creek Women’s Aid Society"], people=["Rev. Anders Nygaard", "Father Thomas Berrigan", "Clara Hestvik"],
            locations=["Old Grange Hall", "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission"], required_details=["harvest relief supper", "flour, sugar, canned goods, potatoes, or dry goods", "donation instruction"],
            tone="practical church-community notice", word_count_min=140, word_count_max=360),

        row(category="religion", artifact_id="REL_1948_001", year=1948, date_label="Sunday, September 5, 1948",
            collection="Mill Creek Religious and Community Life Archive", source_type="choir program note",
            title_seed="Church Choirs Prepare Hymn for Memorial Field Dedication",
            topic="religious music and civic memory", location="Memorial Field",
            institutions=["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Mill Creek Public Schools"], people=["Rev. Anders Nygaard", "Father Thomas Berrigan", "Coach Harold Bratten"],
            locations=["Memorial Field", "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission"], required_details=["choir or hymn", "Memorial Field dedication", "public ceremony detail"],
            tone="respectful religious/civic program note", word_count_min=140, word_count_max=360),

        row(category="religion", artifact_id="REL_1957_001", year=1957, date_label="Sunday, April 21, 1957",
            collection="Mill Creek Religious and Community Life Archive", source_type="service project notice",
            title_seed="Church Volunteers Asked to Report for Flood Cleanup",
            topic="flood volunteer response", location="South Flats",
            institutions=["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Prairie River Clinic"], people=["Dr. Helen Markham", "Rev. Anders Nygaard", "Father Thomas Berrigan"],
            locations=["South Flats", "Prairie River bridge", "Prairie River Clinic"], required_details=["flood cleanup", "sandbag or bridge approach detail", "volunteer reporting time"],
            tone="practical emergency service notice", word_count_min=140, word_count_max=380),

        row(category="religion", artifact_id="REL_1968_001", year=1968, date_label="Sunday, November 24, 1968",
            collection="Mill Creek Religious and Community Life Archive", source_type="sermon_excerpt",
            title_seed="Sermon Excerpt on Roads, Neighbors, and Main Street",
            topic="bypass anxieties and neighborliness", location="St. Ansgar Lutheran Church",
            institutions=["St. Ansgar Lutheran Church", "Mill Creek City Council"], people=["Mayor Edwin Rask", "Ruth Ellingson"],
            locations=["St. Ansgar Lutheran Church", "Downtown/Main Street"], required_details=["bypass discussion", "Main Street", "neighborliness or responsibility"],
            tone="brief sermon excerpt, thoughtful but not grandiose", word_count_min=260, word_count_max=650),

        row(category="religion", artifact_id="REL_1988_001", year=1988, date_label="Sunday, May 22, 1988",
            collection="Mill Creek Religious and Community Life Archive", source_type="community announcement",
            title_seed="Churches Encourage Members to Visit Depot Exhibit",
            topic="churches and public memory", location="restored depot room",
            institutions=["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Mill Creek Historical Society"], people=["Professor Carrel Englekorn", "Lena Voss"],
            locations=["restored depot room", "Mill Creek Historical Society", "Depot District"], required_details=["depot exhibit", "old photograph, flood map, or rail lantern", "Sunday afternoon or visiting-hour detail"],
            tone="community religious announcement, warm and concrete", word_count_min=140, word_count_max=360),

        row(category="religion", artifact_id="REL_1996_001", year=1996, date_label="Sunday, May 5, 1996",
            collection="Mill Creek Religious and Community Life Archive", source_type="youth recognition note",
            title_seed="Congregations Congratulate Student Award Winner",
            topic="student achievement and community recognition", location="Downtown/Main Street",
            institutions=["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Mill Creek Public Schools"], people=["Maizey Olotón", "Asha Patel"],
            locations=["Downtown/Main Street", "Mill Creek High School"], required_details=["student award", "community congratulations", "school or church notice detail"],
            tone="brief community recognition note", word_count_min=120, word_count_max=320),

        row(category="religion", artifact_id="REL_2009_001", year=2009, date_label="Sunday, October 4, 2009",
            collection="Mill Creek Religious and Community Life Archive", source_type="community forum notice",
            title_seed="Church Basement Forum to Discuss Depot District Plan",
            topic="public forum and redevelopment", location="St. Ansgar Lutheran Church",
            institutions=["St. Ansgar Lutheran Church", "Mill Creek City Council", "Mill Creek Historical Society"], people=["Mayor Naomi Reyes", "Professor Carrel Englekorn"],
            locations=["St. Ansgar Lutheran Church", "Depot District", "City Hall"], required_details=["Depot District plan", "forum time or meeting detail", "preservation or redevelopment question"],
            tone="modern church-community notice", word_count_min=150, word_count_max=380),

        row(category="religion", artifact_id="REL_2022_001", year=2022, date_label="Sunday, June 19, 2022",
            collection="Mill Creek Religious and Community Life Archive", source_type="community blessing note",
            title_seed="Short Blessing Offered at Riverfront Trail Opening",
            topic="public ritual at riverfront opening", location="Riverfront Trail",
            institutions=["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Mill Creek Historical Society"], people=["Mayor Naomi Reyes", "Asha Patel"],
            locations=["Riverfront Trail", "South Flats"], required_details=["Riverfront Trail opening", "former South Flats buyout land", "brief blessing or public words"],
            tone="brief contemporary community religious note", word_count_min=130, word_count_max=340),

        row(category="religion", artifact_id="REL_2025_001", year=2025, date_label="Sunday, February 16, 2025",
            collection="Mill Creek Religious and Community Life Archive", source_type="interfaith archive notice",
            title_seed="Community Groups Add Worship Bulletins to Digital Archive",
            topic="religious records in digital archive", location="CobberTech Extension Center",
            institutions=["CobberTech Extension Center", "Mill Creek Historical Society", "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission"], people=["Nora Reyes", "Professor Carrel Englekorn", "Asha Patel"],
            locations=["CobberTech Extension Center", "Mill Creek Historical Society"], required_details=["scanning bulletins or program notes", "metadata or indexing detail", "what records preserve and what they miss"],
            tone="contemporary archive notice, respectful and concrete", word_count_min=180, word_count_max=480),
    ]


def business_rows() -> list[dict[str, Any]]:
    return [
        row(category="business", artifact_id="BUS_1922_001", year=1922, date_label="Thursday, April 13, 1922",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="advertisement",
            title_seed="Main Street Store Announces Spring Seed Orders",
            topic="spring seed orders", location="Downtown/Main Street",
            institutions=["Mill Creek Commercial Club"], people=["Martin Kvale"],
            locations=["Downtown/Main Street", "West Rows farms"], required_details=["seed orders", "price, deadline, or pickup detail", "Main Street"],
            tone="ordinary 1920s advertisement", word_count_min=60, word_count_max=180, min_people=0),

        row(category="business", artifact_id="BUS_1934_001", year=1934, date_label="Thursday, October 11, 1934",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="classified notice",
            title_seed="Dry Goods Requested for Relief Supper",
            topic="relief supply notice", location="Old Grange Hall",
            institutions=["Mill Creek Women’s Aid Society"], people=["Clara Hestvik"],
            locations=["Old Grange Hall", "Downtown/Main Street"], required_details=["dry goods", "drop-off time", "Old Grange Hall"],
            tone="brief Depression-era classified notice", word_count_min=60, word_count_max=200, min_people=0),

        row(category="business", artifact_id="BUS_1948_001", year=1948, date_label="Friday, September 3, 1948",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="advertisement",
            title_seed="Merchants Welcome Fans to Memorial Field Opening",
            topic="downtown merchants and field dedication", location="Downtown/Main Street",
            institutions=["Mill Creek Commercial Club", "Mill Creek Public Schools"], people=["Coach Harold Bratten"],
            locations=["Downtown/Main Street", "Memorial Field"], required_details=["Memorial Field opening", "store hours, sale, or refreshment detail", "opening game"],
            tone="postwar merchant advertisement", word_count_min=70, word_count_max=220, min_people=0),

        row(category="business", artifact_id="BUS_1957_001", year=1957, date_label="Monday, April 22, 1957",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="public notice",
            title_seed="Notice: Water Boiling Advised for South Flats",
            topic="flood public health notice", location="South Flats",
            institutions=["Mill Creek City Council", "Prairie River Clinic"], people=["Dr. Helen Markham", "Mayor Ingrid Lunde"],
            locations=["South Flats", "Prairie River Clinic", "Prairie River bridge"], required_details=["boil-water notice", "South Flats", "clinic or city instruction"],
            tone="brief official public notice", word_count_min=100, word_count_max=280),

        row(category="business", artifact_id="BUS_1968_001", year=1968, date_label="Wednesday, April 24, 1968",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="public hearing notice",
            title_seed="Public Hearing Set on Bypass Route",
            topic="bypass public hearing", location="City Hall",
            institutions=["Mill Creek City Council", "Cobberland County Planning Commission"], people=["Mayor Edwin Rask", "County Engineer Paul Decker"],
            locations=["City Hall", "Downtown/Main Street"], required_details=["public hearing", "bypass route", "time, room, or map review detail"],
            tone="official public notice", word_count_min=100, word_count_max=300),

        row(category="business", artifact_id="BUS_1988_001", year=1988, date_label="Thursday, May 19, 1988",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="directory entry",
            title_seed="Historical Society Lists Depot Exhibit Hours",
            topic="depot exhibit hours", location="Mill Creek Historical Society",
            institutions=["Mill Creek Historical Society"], people=["Professor Carrel Englekorn", "Lena Voss"],
            locations=["Mill Creek Historical Society", "restored depot room"], required_details=["depot exhibit hours", "object or display detail", "contact or visiting instruction"],
            tone="directory-style listing", word_count_min=70, word_count_max=220, min_people=0),

        row(category="business", artifact_id="BUS_1996_001", year=1996, date_label="Friday, May 10, 1996",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="business notice",
            title_seed="Local Student Offers Corn-Snack Samples After Award",
            topic="Maizey early entrepreneurship notice", location="Mill Creek High School",
            institutions=["Mill Creek Public Schools", "Cobberland Future Business Leaders Association"], people=["Maizey Olotón", "David Harlan"],
            locations=["Mill Creek High School"], required_details=["student entrepreneurship award", "sample, display, or demonstration detail", "restrained corn pun or no pun"],
            tone="small local business notice, light but grounded", word_count_min=100, word_count_max=280),

        row(category="business", artifact_id="BUS_2009_001", year=2009, date_label="Monday, September 28, 2009",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="redevelopment notice",
            title_seed="Depot District Comment Period Opens",
            topic="redevelopment comment period", location="Depot District",
            institutions=["Mill Creek City Council", "Mill Creek Historical Society", "Olotón Foods"], people=["Mayor Naomi Reyes", "Maizey Olotón"],
            locations=["Depot District", "City Hall"], required_details=["comment period", "Depot District", "review location or deadline"],
            tone="modern municipal/public notice", word_count_min=120, word_count_max=340),

        row(category="business", artifact_id="BUS_2022_001", year=2022, date_label="Monday, June 20, 2022",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="public notice",
            title_seed="Trail Maintenance Rules Posted for Riverfront Path",
            topic="Riverfront Trail public notice", location="Riverfront Trail",
            institutions=["Mill Creek City Council"], people=["Clerk Asha Patel", "Mayor Naomi Reyes"],
            locations=["Riverfront Trail", "South Flats"], required_details=["Riverfront Trail", "maintenance or use rules", "posted instruction"],
            tone="brief contemporary public notice", word_count_min=100, word_count_max=280),

        row(category="business", artifact_id="BUS_2025_001", year=2025, date_label="Friday, February 14, 2025",
            collection="Mill Creek Business, Notices, and Directories Archive", source_type="archive lab notice",
            title_seed="Digital Archive Lab Requests Business Advertisements",
            topic="business ads in digital archive", location="CobberTech Extension Center",
            institutions=["CobberTech Extension Center", "Mill Creek Historical Society"], people=["Nora Reyes", "Professor Carrel Englekorn"],
            locations=["CobberTech Extension Center", "Mill Creek Historical Society"], required_details=["business advertisements", "scanning or metadata detail", "drop-off or contact instruction"],
            tone="contemporary archive notice", word_count_min=120, word_count_max=340),
    ]


def photo_rows() -> list[dict[str, Any]]:
    return [
        row(category="photo_caption", artifact_id="IMG_1922_001", year=1922, date_label="circa 1922",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="photo_caption",
            title_seed="Reading Room Table on Main Street",
            topic="reading room photograph", location="Main Street reading room",
            institutions=["Mill Creek Women’s Aid Society"], people=["Clara Hestvik"],
            locations=["Main Street reading room", "Downtown/Main Street"], required_details=["reading room table", "books, chairs, ledger, or donation box", "archival uncertainty or donor note"],
            tone="concise archive caption", word_count_min=50, word_count_max=180, min_institutions=0, min_people=0),

        row(category="photo_caption", artifact_id="IMG_1934_001", year=1934, date_label="October 1934",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="photo_caption",
            title_seed="Harvest Relief Supper at Old Grange Hall",
            topic="relief supper photo", location="Old Grange Hall",
            institutions=["Mill Creek Women’s Aid Society"], people=["Clara Hestvik"],
            locations=["Old Grange Hall"], required_details=["Old Grange Hall", "tables, supplies, coats, or serving line", "caption or donor note"],
            tone="concise archive caption", word_count_min=50, word_count_max=180, min_institutions=0, min_people=0),

        row(category="photo_caption", artifact_id="IMG_1948_001", year=1948, date_label="September 1948",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="photo_caption",
            title_seed="Memorial Field Dedication Ceremony",
            topic="dedication photograph", location="Memorial Field",
            institutions=["Mill Creek Public Schools"], people=["Coach Harold Bratten", "Mayor Ingrid Lunde"],
            locations=["Memorial Field"], required_details=["Memorial Field", "flag, band, plaque, seating, or crowd detail", "date estimate"],
            tone="concise exhibit caption", word_count_min=60, word_count_max=190, min_institutions=0, min_people=0),

        row(category="photo_caption", artifact_id="IMG_1957_001", year=1957, date_label="April 1957",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="newspaper_photo_cutline",
            title_seed="Floodwater Near Prairie River Bridge",
            topic="flood photo", location="Prairie River bridge",
            institutions=["Mill Creek City Council"], people=["Public Works Foreman Carl Voss"],
            locations=["Prairie River bridge", "South Flats"], required_details=["floodwater", "bridge approach or sandbags", "South Flats or river level"],
            tone="newspaper cutline, concrete and brief", word_count_min=50, word_count_max=160, min_institutions=0, min_people=0),

        row(category="photo_caption", artifact_id="IMG_1968_001", year=1968, date_label="April 1968",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="map_caption",
            title_seed="Proposed Bypass Route Map",
            topic="bypass map", location="Downtown/Main Street",
            institutions=["Cobberland County Planning Commission", "Mill Creek City Council"], people=["County Engineer Paul Decker"],
            locations=["Downtown/Main Street", "City Hall"], required_details=["bypass route", "Main Street", "map annotation or hearing note"],
            tone="map caption, concrete", word_count_min=60, word_count_max=180, min_institutions=0, min_people=0),

        row(category="photo_caption", artifact_id="IMG_1988_001", year=1988, date_label="May 1988",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="exhibit_label",
            title_seed="Rail Lantern in Restored Depot Room",
            topic="depot exhibit object", location="restored depot room",
            institutions=["Mill Creek Historical Society"], people=["Professor Carrel Englekorn", "Lena Voss"],
            locations=["restored depot room", "Mill Creek Historical Society"], required_details=["rail lantern", "restored depot room", "donor, label, or display case detail"],
            tone="museum-style exhibit label, concise", word_count_min=70, word_count_max=220, min_people=0),

        row(category="photo_caption", artifact_id="IMG_1996_001", year=1996, date_label="May 1996",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="photo_caption",
            title_seed="Maizey Olotón With Entrepreneurship Award",
            topic="student award photograph", location="Mill Creek High School",
            institutions=["Mill Creek Public Schools"], people=["Maizey Olotón", "Principal Ruth Ellingson"],
            locations=["Mill Creek High School"], required_details=["award certificate or display table", "school hallway, classroom, or auditorium", "Maizey Olotón"],
            tone="school archive photo caption", word_count_min=60, word_count_max=180, min_institutions=0, min_people=1),

        row(category="photo_caption", artifact_id="IMG_2009_001", year=2009, date_label="September 2009",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="map_caption",
            title_seed="Depot District Redevelopment Plan Map",
            topic="redevelopment map", location="Depot District",
            institutions=["Mill Creek City Council", "Mill Creek Historical Society"], people=["Mayor Naomi Reyes"],
            locations=["Depot District", "City Hall"], required_details=["Depot District", "redevelopment plan", "rail-era building or public hearing mark"],
            tone="map caption for public exhibit", word_count_min=60, word_count_max=190, min_people=0),

        row(category="photo_caption", artifact_id="IMG_2022_001", year=2022, date_label="June 2022",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="photo_caption",
            title_seed="Riverfront Trail Opening Ceremony",
            topic="trail opening photo", location="Riverfront Trail",
            institutions=["Mill Creek City Council", "Mill Creek Historical Society"], people=["Mayor Naomi Reyes", "Asha Patel"],
            locations=["Riverfront Trail", "South Flats"], required_details=["Riverfront Trail", "former South Flats buyout land", "ribbon, sign, crowd, or creekbank detail"],
            tone="contemporary archive caption", word_count_min=60, word_count_max=180, min_institutions=0, min_people=0),

        row(category="photo_caption", artifact_id="IMG_2025_001", year=2025, date_label="February 2025",
            collection="Mill Creek Historical Society Image and Object Archive", source_type="photo_caption",
            title_seed="CobberTech Students Scanning Newspaper Clippings",
            topic="digital archive lab photo", location="CobberTech Extension Center",
            institutions=["CobberTech Extension Center", "Mill Creek Historical Society"], people=["Nora Reyes", "Professor Carrel Englekorn"],
            locations=["CobberTech Extension Center", "Mill Creek Historical Society"], required_details=["scanning newspaper clippings", "computer, flatbed scanner, folders, or metadata worksheet", "digital archive lab"],
            tone="contemporary archive caption", word_count_min=60, word_count_max=180, min_institutions=0, min_people=0),
    ]


def oral_history_rows() -> list[dict[str, Any]]:
    return [
        row(category="oral_history", artifact_id="OH_1988_001", year=1988, date_label="May 21, 1988",
            collection="Mill Creek Historical Society Oral History Collection", source_type="oral_history_transcript",
            title_seed="Lena Voss Remembers the Reading Room and Depot",
            topic="public memory and reading room", location="Mill Creek Historical Society",
            institutions=["Mill Creek Historical Society", "Mill Creek Women’s Aid Society"], people=["Lena Voss", "Professor Carrel Englekorn", "Clara Hestvik"],
            locations=["Mill Creek Historical Society", "Main Street reading room", "restored depot room"], required_details=["Main Street reading room", "restored depot room object", "what the town chooses to remember"],
            tone="reflective elder interview, concrete, warm but not sentimental", word_count_min=650, word_count_max=1050),

        row(category="oral_history", artifact_id="OH_1994_001", year=1994, date_label="August 3, 1994",
            collection="Mill Creek Historical Society Oral History Collection", source_type="oral_history_transcript",
            title_seed="Peter Harlan on the 1957 Flood",
            topic="flood memory and public works", location="South Flats",
            institutions=["Mill Creek Historical Society", "Mill Creek City Council", "Prairie River Clinic"], people=["Peter Harlan", "Ruth Ellingson", "Dr. Helen Markham"],
            locations=["South Flats", "Prairie River bridge", "Old Mill Bend"], required_details=["1957 flood memory", "bridge approaches or sandbags", "public works or water detail"],
            tone="plainspoken public-works memory with uncertainty", word_count_min=650, word_count_max=1050),

        row(category="oral_history", artifact_id="OH_1990_001", year=1990, date_label="October 12, 1990",
            collection="Mill Creek Historical Society Oral History Collection", source_type="oral_history_transcript",
            title_seed="Ruth Ellingson on Memorial Field and School Records",
            topic="school records and postwar memory", location="Mill Creek Historical Society",
            institutions=["Mill Creek Historical Society", "Mill Creek Public Schools"], people=["Ruth Ellingson", "Professor Carrel Englekorn", "Coach Harold Bratten"],
            locations=["Mill Creek Historical Society", "Memorial Field", "Mill Creek Public Schools"], required_details=["Memorial Field dedication", "plaque or record-keeping detail", "school band, opening game, or public ceremony"],
            tone="careful institutional memory from someone who kept records", word_count_min=650, word_count_max=1050),

        row(category="oral_history", artifact_id="OH_2006_001", year=2006, date_label="November 18, 2006",
            collection="CobberTech Extension Center Community Voices Project", source_type="oral_history_transcript",
            title_seed="Asha Patel on Schools, Belonging, and Public Traditions",
            topic="belonging and changing public traditions", location="North Orchard",
            institutions=["CobberTech Extension Center", "Mill Creek Public Schools", "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission"], people=["Asha Patel", "Janine Roberts", "Professor Carrel Englekorn"],
            locations=["North Orchard", "Mill Creek Public Schools", "Downtown/Main Street"], required_details=["school as a place where families meet", "feeling included and sometimes watched", "public traditions changing"],
            tone="modern reflective interview, respectful and specific", word_count_min=650, word_count_max=1100),

        row(category="oral_history", artifact_id="OH_2010_001", year=2010, date_label="April 9, 2010",
            collection="CobberTech Extension Center Community Voices Project", source_type="oral_history_transcript",
            title_seed="Maizey Olotón on West Rows Farms and Olotón Foods",
            topic="agriculture, school, and entrepreneurship", location="West Rows farms",
            institutions=["CobberTech Extension Center", "Olotón Foods", "Mill Creek Public Schools"], people=["Maizey Olotón", "Asha Patel", "David Harlan"],
            locations=["West Rows farms", "Olotón Foods", "Mill Creek High School"], required_details=["1996 state entrepreneurship award", "West Rows farms", "Olotón Foods"],
            tone="local success interview, grounded, restrained humor", word_count_min=650, word_count_max=1100),

        row(category="oral_history", artifact_id="OH_2012_001", year=2012, date_label="September 7, 2012",
            collection="Mill Creek Historical Society Oral History Collection", source_type="oral_history_transcript",
            title_seed="Rosa Martinez on Leaving South Flats",
            topic="buyout land, loss, home, and memory", location="South Flats",
            institutions=["Mill Creek Historical Society", "Mill Creek City Council"], people=["Rosa Martinez", "Lena Voss", "Mayor Naomi Reyes"],
            locations=["South Flats", "Prairie River bridge"], required_details=["2011 buyout or leaving South Flats", "difference between flood control and home", "porch, alley, bridge, yard, or kitchen memory"],
            tone="personal memory of loss, restrained and respectful", word_count_min=700, word_count_max=1150),

        row(category="oral_history", artifact_id="OH_2022_001", year=2022, date_label="June 20, 2022",
            collection="CobberTech Extension Center Community Voices Project", source_type="oral_history_transcript",
            title_seed="Professor Englekorn on Local History and Partial Archives",
            topic="local history, archive bias, and digitization", location="Mill Creek Historical Society",
            institutions=["Mill Creek Historical Society", "CobberTech Extension Center"], people=["Professor Carrel Englekorn", "Asha Patel", "Lena Voss"],
            locations=["Mill Creek Historical Society", "Riverfront Trail", "South Flats"], required_details=["digitization of photographs or records", "even good history is partial", "Riverfront Trail or South Flats"],
            tone="professional historian reflecting with humility", word_count_min=700, word_count_max=1150),

        row(category="oral_history", artifact_id="OH_2025_001", year=2025, date_label="February 14, 2025",
            collection="CobberTech Extension Center Digital Archive Lab", source_type="oral_history_transcript",
            title_seed="Nora Reyes on Scanning Photographs and Noticing Absences",
            topic="student digitization work and missing records", location="CobberTech Extension Center",
            institutions=["CobberTech Extension Center", "Mill Creek Historical Society"], people=["Nora Reyes", "Professor Carrel Englekorn", "Asha Patel"],
            locations=["CobberTech Extension Center", "Mill Creek Historical Society"], required_details=["scanning photographs or newspaper clippings", "noticing what is missing from the archive", "digital tools help but do not interpret by themselves"],
            tone="student interview, thoughtful, direct, not too polished", word_count_min=700, word_count_max=1150),

        row(category="oral_history", artifact_id="OH_1996_002", year=1996, date_label="May 17, 1996",
            collection="Mill Creek Historical Society Oral History Collection", source_type="oral_history_transcript",
            title_seed="David Harlan on Coaching, Students, and Maizey’s Award",
            topic="school mentorship and student entrepreneurship", location="Mill Creek High School",
            institutions=["Mill Creek Public Schools", "Cobberland Future Business Leaders Association"], people=["David Harlan", "Maizey Olotón", "Principal Ruth Ellingson"],
            locations=["Mill Creek High School", "Mill Creek Public Schools"], required_details=["Maizey’s award", "teacher or mentor memory", "school hallway, classroom, or assembly detail"],
            tone="teacher memory, concrete and modest", word_count_min=600, word_count_max=1000),

        row(category="oral_history", artifact_id="OH_2009_002", year=2009, date_label="October 2, 2009",
            collection="Mill Creek Historical Society Oral History Collection", source_type="oral_history_transcript",
            title_seed="Mayor Naomi Reyes on the Depot District Plan",
            topic="redevelopment and memory", location="City Hall",
            institutions=["Mill Creek City Council", "Mill Creek Historical Society", "Olotón Foods"], people=["Mayor Naomi Reyes", "Professor Carrel Englekorn", "Maizey Olotón"],
            locations=["City Hall", "Depot District", "Mill Creek Historical Society"], required_details=["Depot District plan", "public hearing or map detail", "preservation and economic development tension"],
            tone="civic interview, careful and grounded", word_count_min=650, word_count_max=1050),
    ]


def council_minutes_rows() -> list[dict[str, Any]]:
    return [
        row(category="council_minutes", artifact_id="MIN_1922_001", year=1922, date_label="Monday, June 12, 1922",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes on Reading Room Fund Drive and Plank Walk",
            topic="reading room and street report", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Mill Creek Women’s Aid Society"], people=["Mayor Nels Hovland", "Clerk Elsie Bratten", "Clara Hestvik"],
            locations=["Council Chambers, City Hall", "Main Street reading room", "Downtown/Main Street"], required_details=["call to order and attendance", "reading room fund drive", "plank walk or street report"],
            tone="formal early-20th-century municipal minutes", word_count_min=300, word_count_max=700),

        row(category="council_minutes", artifact_id="MIN_1934_001", year=1934, date_label="Monday, October 15, 1934",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes on Harvest Relief Supper and Street Grading",
            topic="Depression relief and street grading", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Mill Creek Women’s Aid Society"], people=["Mayor Nels Hovland", "Clerk Elsie Bratten", "Clara Hestvik"],
            locations=["Council Chambers, City Hall", "Old Grange Hall"], required_details=["Old Grange Hall", "Harvest Relief Supper", "street grading delayed or tabled"],
            tone="formal municipal minutes, practical and concise", word_count_min=300, word_count_max=700),

        row(category="council_minutes", artifact_id="MIN_1948_001", year=1948, date_label="Monday, August 30, 1948",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes on Memorial Field Dedication",
            topic="dedication arrangements", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Mill Creek Public Schools"], people=["Mayor Ingrid Lunde", "Clerk Ruth Ellingson", "Coach Harold Bratten"],
            locations=["Council Chambers, City Hall", "Memorial Field"], required_details=["Memorial Field dedication", "temporary seating or plaque", "opening game or band arrangement"],
            tone="formal postwar municipal minutes", word_count_min=300, word_count_max=700),

        row(category="council_minutes", artifact_id="MIN_1957_001", year=1957, date_label="Monday, April 22, 1957",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes on Flood Crest and Bridge Approaches",
            topic="flood response", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Prairie River Clinic"], people=["Mayor Ingrid Lunde", "Clerk Ruth Ellingson", "Dr. Helen Markham", "Public Works Foreman Carl Voss"],
            locations=["Council Chambers, City Hall", "Prairie River bridge", "South Flats"], required_details=["flood crest", "bridge approaches or sandbags", "boil-water or clinic notice"],
            tone="emergency municipal minutes, plain and factual", word_count_min=320, word_count_max=750),

        row(category="council_minutes", artifact_id="MIN_1968_001", year=1968, date_label="Monday, April 15, 1968",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes on Highway Bypass Questions",
            topic="bypass planning", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Mill Creek Commercial Club", "Cobberland County Planning Commission"], people=["Mayor Edwin Rask", "Clerk Ruth Ellingson", "County Engineer Paul Decker"],
            locations=["Council Chambers, City Hall", "Downtown/Main Street"], required_details=["highway bypass", "merchant concern or Commercial Club", "traffic estimate, map, or public hearing"],
            tone="late-1960s planning minutes, balanced and concrete", word_count_min=320, word_count_max=760),

        row(category="council_minutes", artifact_id="MIN_1988_001", year=1988, date_label="Monday, March 7, 1988",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes on Historical Society Depot Exhibit",
            topic="Historical Society exhibit and depot room", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Mill Creek Historical Society"], people=["Mayor Edwin Rask", "Clerk Janine Roberts", "Professor Carrel Englekorn", "Lena Voss"],
            locations=["Council Chambers, City Hall", "restored depot room", "Mill Creek Historical Society"], required_details=["restored depot room", "insurance or key access", "student volunteers or exhibit objects"],
            tone="formal municipal minutes, preservation focus", word_count_min=320, word_count_max=760),

        row(category="council_minutes", artifact_id="MIN_1996_001", year=1996, date_label="Monday, May 6, 1996",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes Recognizing Student Entrepreneurship Award",
            topic="student recognition and local business", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Mill Creek Public Schools"], people=["Mayor Naomi Reyes", "Clerk Janine Roberts", "Maizey Olotón"],
            locations=["Council Chambers, City Hall", "Mill Creek High School"], required_details=["recognition of student award", "Maizey Olotón", "school or entrepreneurship detail"],
            tone="formal municipal minutes with brief recognition item", word_count_min=280, word_count_max=680),

        row(category="council_minutes", artifact_id="MIN_2009_001", year=2009, date_label="Monday, September 21, 2009",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes on Depot District Redevelopment",
            topic="redevelopment and preservation", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Mill Creek Historical Society", "Olotón Foods"], people=["Mayor Naomi Reyes", "Clerk Janine Roberts", "Professor Carrel Englekorn", "Maizey Olotón"],
            locations=["Council Chambers, City Hall", "Depot District", "Mill Creek Historical Society"], required_details=["Depot District redevelopment", "public hearing or review period", "rail-era preservation"],
            tone="modern municipal minutes, concrete and not jargon-heavy", word_count_min=330, word_count_max=800),

        row(category="council_minutes", artifact_id="MIN_2011_001", year=2011, date_label="Monday, August 8, 2011",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes on South Flats Buyout Notices",
            topic="South Flats buyout and floodplain", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Prairie River Clinic"], people=["Mayor Naomi Reyes", "Clerk Janine Roberts", "Rosa Martinez"],
            locations=["Council Chambers, City Hall", "South Flats", "Prairie River bridge"], required_details=["South Flats buyout", "floodplain or flood control", "resident comment or relocation concern"],
            tone="modern municipal minutes, restrained and respectful", word_count_min=330, word_count_max=800),

        row(category="council_minutes", artifact_id="MIN_2022_001", year=2022, date_label="Monday, May 16, 2022",
            collection="Mill Creek City Council Minutes Archive", source_type="city_council_minutes",
            title_seed="Council Minutes on Riverfront Trail Opening and Archive Pilot",
            topic="Riverfront Trail and digitization", location="Council Chambers, City Hall",
            institutions=["Mill Creek City Council", "Mill Creek Historical Society", "CobberTech Extension Center"], people=["Mayor Naomi Reyes", "Clerk Asha Patel", "Professor Carrel Englekorn", "Nora Reyes"],
            locations=["Council Chambers, City Hall", "Riverfront Trail", "South Flats", "CobberTech Extension Center"], required_details=["Riverfront Trail opening", "former South Flats buyout land", "CobberTech digitization pilot or historical sign"],
            tone="contemporary municipal minutes, concrete", word_count_min=330, word_count_max=800),
    ]


MANIFEST_BUILDERS = {
    "newspaper": newspaper_rows,
    "school": school_rows,
    "religion": religion_rows,
    "business": business_rows,
    "photo_caption": photo_rows,
    "oral_history": oral_history_rows,
    "council_minutes": council_minutes_rows,
}


# ---------------------------------------------------------------------
# Gazetteer
# ---------------------------------------------------------------------

PLACES = [
    {"place_id":"PLC_001","name":"Old Mill Bend","district":"river","x":0.0,"y":0.0,"active_from":1874,"active_to":"","themes":"origin story; river; mill; memory"},
    {"place_id":"PLC_002","name":"Prairie River bridge","district":"river","x":1.5,"y":-0.4,"active_from":1893,"active_to":"","themes":"flood; infrastructure; movement"},
    {"place_id":"PLC_003","name":"Downtown/Main Street","district":"downtown","x":0.5,"y":0.7,"active_from":1879,"active_to":"","themes":"commerce; civic ritual; decline; nostalgia"},
    {"place_id":"PLC_004","name":"Depot District","district":"depot","x":1.2,"y":1.0,"active_from":1879,"active_to":"","themes":"rail; labor; preservation; redevelopment"},
    {"place_id":"PLC_005","name":"South Flats","district":"river","x":1.0,"y":-1.1,"active_from":1900,"active_to":"","themes":"home; flood risk; buyout; loss"},
    {"place_id":"PLC_006","name":"Mill Creek Public Schools","district":"school","x":-0.2,"y":1.6,"active_from":1902,"active_to":"","themes":"youth; civic identity; memory"},
    {"place_id":"PLC_007","name":"Mill Creek High School","district":"school","x":-0.4,"y":1.8,"active_from":1928,"active_to":"","themes":"youth; sports; Maizey Olotón"},
    {"place_id":"PLC_008","name":"Memorial Field","district":"school","x":-0.7,"y":1.5,"active_from":1948,"active_to":"","themes":"war memory; sports; public ceremony"},
    {"place_id":"PLC_009","name":"West Rows farms","district":"agricultural","x":-2.0,"y":0.3,"active_from":1874,"active_to":"","themes":"agriculture; seed; Maizey Olotón"},
    {"place_id":"PLC_010","name":"Olotón Foods","district":"agricultural/business","x":-1.6,"y":0.0,"active_from":2001,"active_to":"","themes":"entrepreneurship; agriculture"},
    {"place_id":"PLC_011","name":"Mill Creek Historical Society","district":"depot","x":1.1,"y":1.05,"active_from":1988,"active_to":"","themes":"public memory; archive; preservation"},
    {"place_id":"PLC_012","name":"CobberTech Extension Center","district":"education/civic","x":0.2,"y":2.3,"active_from":1998,"active_to":"","themes":"digitization; students; digital humanities"},
    {"place_id":"PLC_013","name":"Riverfront Trail","district":"river","x":0.8,"y":-0.8,"active_from":2022,"active_to":"","themes":"buyout land; public memory; recreation"},
]


def write_gazetteer(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "mill_creek_gazetteer.csv"
    jsonl_path = out_dir / "mill_creek_gazetteer.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PLACES[0].keys()))
        writer.writeheader()
        writer.writerows(PLACES)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row_obj in PLACES:
            f.write(json.dumps(row_obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Prompting and Ollama
# ---------------------------------------------------------------------

def build_system_prompt(category: str) -> str:
    return f"""
You generate primary-source-like artifacts for a fictional digital humanities archive.

CRITICAL RULES:
1. Use only the supplied institutions, people, and locations.
2. Do not invent additional named people, churches, clubs, businesses, newspapers,
   schools, government bodies, counties, towns, projects, or committees.
3. Do not mention that Mill Creek is fictional, simulated, generated, or part of a textbook.
4. Do not use Markdown.
5. Keep prose historically appropriate to the assigned year and source type.
6. Prefer concrete details over vague civic language.
7. Return only the requested plain-text fields.

Category instruction:
{CATEGORY_INSTRUCTIONS[category]}

{GLOBAL_CONTEXT}
"""


def build_user_prompt(assignment: dict[str, Any], previous_errors: list[str] | None = None) -> str:
    errors = ""
    if previous_errors:
        errors = "\nPrevious output had problems. Correct these problems:\n"
        errors += "\n".join(f"- {e}" for e in previous_errors)
        errors += "\n"

    category = assignment["category"]

    category_specific = ""
    if category == "oral_history":
        category_specific = """
For BODY:
- Start directly with an interviewer question or a brief spoken setup.
- Use interviewer and speaker turns.
- Include 5 to 8 interviewer questions.
- Label turns only as "Interviewer:" and "Speaker:".
- Do not begin BODY with artifact_id, year, date, collection, source_type, or primary_location.
"""
    elif category == "council_minutes":
        category_specific = """
For BODY:
- Include call to order.
- Include attendance.
- Include at least two agenda/action items.
- Include motions or outcomes.
- Include adjournment.
"""
    elif category == "photo_caption":
        category_specific = """
For BODY:
- Write one concise caption or label.
- Include visible details and, if appropriate, a note of uncertainty or donor/context information.
"""
    else:
        category_specific = """
For BODY:
- Write plain text paragraphs appropriate to the source type.
"""

    return f"""
Write one Mill Creek archive artifact.

Assignment:
- artifact_id: {assignment["artifact_id"]}
- category: {assignment["category"]}
- year: {assignment["year"]}
- date: {assignment["date"]}
- collection: {assignment["collection"]}
- source_type: {assignment["source_type"]}
- source_group: {assignment.get("source_group", "") }
- title seed: {assignment["title_seed"]}
- topic: {assignment["topic"]}
- primary location: {assignment["location"]}
- tone: {assignment["tone"]}
- word count range for body: {assignment["word_count_min"]}-{assignment["word_count_max"]} words

Allowed institutions:
{json.dumps(assignment["allowed_institutions"], ensure_ascii=False, indent=2)}

Allowed people:
{json.dumps(assignment["allowed_people"], ensure_ascii=False, indent=2)}

Allowed locations:
{json.dumps(assignment["allowed_locations"], ensure_ascii=False, indent=2)}

Required concrete details:
{json.dumps(assignment["required_details"], ensure_ascii=False, indent=2)}
{forbidden_terms_prompt_line(assignment)}
Output format:
Return exactly this plain-text format:

TITLE: final title or headline, based on the seed but polished
BODY:
artifact body as plain text
END_BODY

Do not return JSON.
Do not use Markdown.
Do not include assignment metadata labels such as artifact_id, year, date, collection, source_type, or primary_location inside BODY.
Those metadata fields are already stored separately by the Python script.

Content requirements:
- Include all required concrete details in natural prose.
- Mention at least {assignment["min_institutions"]} allowed institution(s), unless the count is 0.
- Mention at least {assignment["min_people"]} allowed person/people, unless the count is 0.
- Mention at least {assignment["min_locations"]} allowed location(s), unless the count is 0.
- Use only supplied names, institutions, newspapers, locations, counties, and towns.
- Do not introduce other Mill Creek place names unless they are in the allowed locations list.
- Do not include assignment metadata labels such as artifact_id, category, collection, source_type, or required_details in the BODY.
- Keep the artifact focused on the assigned topic.
- Do not use modern academic language unless the assignment itself is contemporary and archive/digital-humanities related.
- Do not use em dashes excessively.
- Do not use corn puns unless the assignment explicitly allows it.
{category_specific}
{errors}
"""


def call_ollama(
    *,
    assignment: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    options: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": args.model,
        "stream": False,
        "keep_alive": "30m",
        "think": False,
        "messages": [
            {"role": "system", "content": build_system_prompt(assignment["category"])},
            {"role": "user", "content": prompt},
        ],
        "options": options,
    }

    t0 = time.perf_counter()
    response = requests.post(args.ollama_url, json=payload, timeout=args.timeout)
    wall = time.perf_counter() - t0
    response.raise_for_status()
    return response.json(), wall


def parse_response(api_data: dict[str, Any]) -> dict[str, str]:
    message = api_data.get("message", {})
    content = message.get("content", "")
    thinking = message.get("thinking", "")

    if not content.strip():
        if thinking.strip():
            raise ValueError('Empty message.content; model appears to have answered in message.thinking. Payload should include "think": False.')
        raise ValueError("Empty model response.")

    title_match = re.search(r"^TITLE:\s*(.*)$", content, flags=re.MULTILINE)
    body_match = re.search(r"^BODY:\s*(.*?)\s*^END_BODY\s*$", content, flags=re.MULTILINE | re.DOTALL)

    return {
        "title": title_match.group(1).strip() if title_match else "",
        "body": body_match.group(1).strip() if body_match else "",
    }


# ---------------------------------------------------------------------
# Validation and metrics
# ---------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’\-]+\b", text))


def contains_forbidden(text: str, phrase: str) -> bool:
    if phrase in {"###", "**"}:
        return phrase in text
    return re.search(r"\b" + re.escape(phrase) + r"\b", text, flags=re.IGNORECASE) is not None


def important_terms(text: str) -> list[str]:
    stop = {
        "about", "after", "along", "also", "before", "being", "could", "details",
        "first", "from", "have", "into", "least", "light", "local", "named",
        "public", "report", "required", "should", "such", "that", "their",
        "there", "these", "this", "with", "word", "would", "year",
        "allowed", "concrete", "detail", "details", "comment",
    }
    return [
        t.lower()
        for t in re.findall(r"\b[A-Za-z][A-Za-z’'\-]{4,}\b", text)
        if t.lower() not in stop
    ]


def detail_satisfied(detail: str, combined: str) -> bool:
    """Flexible check for a required detail.

    This is intentionally lightweight. It catches clear omissions without turning
    every phrase into a brittle exact-string validator.
    """
    low_detail = detail.lower()
    low = combined.lower()

    # Common archive wording variants.
    if "uncertainty" in low_detail and any(x in low for x in ["unclear", "uncertain", "appears", "may be", "probably", "likely"]):
        return True
    if "donor" in low_detail and any(x in low for x in ["donor", "donated", "gift", "given by"]):
        return True
    if "digital tools help" in low_detail and "interpret" in low:
        return True

    terms = important_terms(detail)
    if not terms:
        return True

    # If the detail offers alternatives with "or", one strong hit is enough.
    if " or " in low_detail:
        return any(t in low for t in terms[:8])

    return sum(1 for t in terms[:6] if t in low) > 0




def v8_timeline_warnings(combined: str, assignment: dict[str, Any]) -> list[str]:
    """Post-generation canon guardrails learned from the Level 13 audit."""
    year = int(assignment.get("year", 0))
    category = assignment.get("category", "")
    low = combined.lower()
    rules = [
        ("Olotón Foods", 2001, False),
        ("Memorial Field", 1948, False),
        ("Mill Creek Historical Society", 1988, category == "photo_caption"),
        ("CobberTech Extension Center", 1998, False),
        ("Riverfront Trail", 2022, False),
        ("Prairie River Clinic", 1954, False),
        ("Mill Creek High School", 1928, False),
        ("Maizey Olotón", 1996, False),
        ("Nora Reyes", 2022, False),
    ]
    out: list[str] = []
    for phrase, start_year, allow_photo_caption_soft in rules:
        if year < start_year and phrase.lower() in low:
            if allow_photo_caption_soft:
                out.append(f"Soft timeline warning: {phrase} appears before {start_year}; acceptable only as modern archive caption framing.")
            else:
                out.append(f"Hard timeline warning: {phrase} appears before {start_year}.")
    if year > 1987 and "mill creek chronicle" in low and assignment.get("category") == "newspaper":
        out.append("Hard timeline warning: Mill Creek Chronicle appears after 1987 in a newspaper artifact.")
    return out



def hard_validation_warnings(warnings: list[str]) -> list[str]:
    """Warnings that should trigger automatic regeneration even during timing runs."""
    hard_prefixes = [
        "Missing body.",
        "Hard timeline warning:",
        "Forbidden phrase appears:",
    ]
    out = []
    for w in warnings:
        if any(w.startswith(prefix) for prefix in hard_prefixes):
            out.append(w)
        elif "Body word count 0 outside" in w:
            out.append(w)
    return out


def forbidden_terms_for_year(year: int, category: str) -> list[str]:
    """Terms that must not appear before their active date, except soft modern photo-caption framing."""
    rules = [
        ("Olotón Foods", 2001, False),
        ("Memorial Field", 1948, False),
        ("Mill Creek Historical Society", 1988, category == "photo_caption"),
        ("CobberTech Extension Center", 1998, False),
        ("Riverfront Trail", 2022, False),
        ("Prairie River Clinic", 1954, False),
        ("Mill Creek High School", 1928, False),
        ("Maizey Olotón", 1996, False),
        ("Nora Reyes", 2022, False),
    ]
    return [phrase for phrase, start_year, allow_soft in rules if year < start_year and not allow_soft]


def forbidden_terms_prompt_line(assignment: dict[str, Any]) -> str:
    """Extra prompt guardrail to prevent hard active-date hallucinations."""
    year = int(assignment.get("year", 0))
    category = assignment.get("category", "")
    terms = forbidden_terms_for_year(year, category)
    if not terms:
        return ""
    return (
        "\nHard date restrictions for this artifact:\n"
        f"- Because the year is {year}, do NOT mention any of these terms: "
        + ", ".join(terms)
        + ".\n"
        "- If you need a similar idea, use only the allowed institutions, people, and locations supplied above.\n"
    )

def validate_generated(generated: dict[str, str], assignment: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    title = generated.get("title", "")
    body = generated.get("body", "")
    combined = f"{title}\n{body}"

    if not title.strip():
        warnings.append("Missing title.")
    if not body.strip():
        warnings.append("Missing body.")

    wc = word_count(body)
    min_wc = int(assignment["word_count_min"] * 0.90)
    max_wc = int(assignment["word_count_max"] * 1.10)
    if wc < min_wc or wc > max_wc:
        warnings.append(f"Body word count {wc} outside slack range {min_wc}-{max_wc}.")

    for phrase in FORBIDDEN_PHRASES:
        if contains_forbidden(combined, phrase):
            warnings.append(f"Forbidden phrase appears: {phrase}")

    if assignment["min_institutions"] > 0:
        hits = [x for x in assignment["allowed_institutions"] if x in combined]
        if len(hits) < assignment["min_institutions"]:
            warnings.append("Too few allowed institutions appear.")

    if assignment["min_people"] > 0:
        hits = [x for x in assignment["allowed_people"] if x in combined]
        if len(hits) < assignment["min_people"]:
            warnings.append("Too few allowed people appear.")

    if assignment["min_locations"] > 0:
        hits = [x for x in assignment["allowed_locations"] if x in combined]
        if len(hits) < assignment["min_locations"]:
            warnings.append("Too few allowed locations appear.")

    for detail in assignment["required_details"]:
        if not detail_satisfied(detail, combined):
            warnings.append(f"Required detail may be missing: {detail}")

    for phrase in [
        "enduring spirit", "true character of Mill Creek", "bonds of Cobberland",
        "palpable sense", "measure of hope", "noble endeavor", "came together as one",
    ]:
        if phrase in combined.lower():
            warnings.append(f"Generic LLM-style phrase appears: {phrase}")

    warnings.extend(v8_timeline_warnings(combined, assignment))

    return warnings


def ns_to_s(value: Any) -> float:
    try:
        return float(value) / 1_000_000_000.0
    except (TypeError, ValueError):
        return 0.0


def collect_metrics(
    api_data: dict[str, Any],
    wall: float,
    assignment: dict[str, Any],
    generated: dict[str, str],
    attempt: int,
    warnings: list[str],
    status: str,
    model: str,
) -> dict[str, Any]:
    eval_count = int(api_data.get("eval_count") or 0)
    eval_duration_s = ns_to_s(api_data.get("eval_duration"))
    prompt_eval_count = int(api_data.get("prompt_eval_count") or 0)
    prompt_eval_duration_s = ns_to_s(api_data.get("prompt_eval_duration"))
    total_duration_s = ns_to_s(api_data.get("total_duration"))
    load_duration_s = ns_to_s(api_data.get("load_duration"))

    return {
        "artifact_id": assignment["artifact_id"],
        "category": assignment["category"],
        "year": assignment["year"],
        "era": assignment["era"],
        "source_type": assignment["source_type"],
        "attempt": attempt,
        "status": status,
        "warning_count": len(warnings),
        "warnings_json": json.dumps(warnings, ensure_ascii=False),
        "wall_clock_s": round(wall, 3),
        "total_duration_s": round(total_duration_s, 3),
        "load_duration_s": round(load_duration_s, 3),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_s": round(prompt_eval_duration_s, 3),
        "eval_count": eval_count,
        "eval_duration_s": round(eval_duration_s, 3),
        "eval_tokens_per_s": round(eval_count / eval_duration_s, 2) if eval_duration_s > 0 else 0.0,
        "wall_tokens_per_s": round(eval_count / wall, 2) if wall > 0 else 0.0,
        "body_word_count": word_count(generated.get("body", "")),
        "model": model,
    }


def build_record(assignment: dict[str, Any], generated: dict[str, str], metrics: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    record = dict(assignment)
    record["title"] = generated.get("title", "")
    record["body"] = generated.get("body", "")
    record["_generation"] = {
        "model": metrics["model"],
        "attempt": metrics["attempt"],
        "status": metrics["status"],
        "warnings": warnings,
        "metrics": metrics,
    }
    return record


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row_obj in rows:
            f.write(json.dumps(row_obj, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def append_csv(path: Path, row_obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_obj.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row_obj)
        f.flush()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def generate_one(assignment: dict[str, Any], args: argparse.Namespace, options: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    last_warning_list: list[str] = []
    last_generated: dict[str, str] = {"title": "", "body": ""}

    for attempt in range(1, args.max_retries + 2):
        prompt = build_user_prompt(assignment, previous_errors=last_warning_list if args.retry_on_validation else None)

        try:
            api_data, wall = call_ollama(assignment=assignment, prompt=prompt, args=args, options=options)
            generated = parse_response(api_data)
        except Exception as exc:
            if attempt <= args.max_retries + 1:
                last_warning_list = [f"Exception: {exc}"]
                if attempt <= args.max_retries:
                    continue
            failure_metrics = {
                "artifact_id": assignment["artifact_id"],
                "category": assignment["category"],
                "year": assignment["year"],
                "era": assignment["era"],
                "source_type": assignment["source_type"],
                "attempt": attempt,
                "status": "FAIL",
                "warning_count": 1,
                "warnings_json": json.dumps([str(exc)], ensure_ascii=False),
                "wall_clock_s": 0.0,
                "total_duration_s": 0.0,
                "load_duration_s": 0.0,
                "prompt_eval_count": 0,
                "prompt_eval_duration_s": 0.0,
                "eval_count": 0,
                "eval_duration_s": 0.0,
                "eval_tokens_per_s": 0.0,
                "wall_tokens_per_s": 0.0,
                "body_word_count": 0,
                "model": args.model,
            }
            return None, failure_metrics

        warnings = validate_generated(generated, assignment)
        hard_warnings = hard_validation_warnings(warnings)

        # V8 guardrail:
        # Even if --retry-on-validation is off for timing purposes, automatically retry
        # empty bodies and hard canon violations a small number of times. This prevents
        # one blank artifact or one active-date hallucination from slipping into a long run.
        if hard_warnings and attempt <= args.hard_retries:
            last_warning_list = hard_warnings
            last_generated = generated
            continue

        status = "OK" if not warnings else "WARN"
        if hard_warnings:
            status = "FAIL"

        metrics = collect_metrics(api_data, wall, assignment, generated, attempt, warnings, status, args.model)

        last_warning_list = warnings
        last_generated = generated

        if (not warnings) or (not args.retry_on_validation):
            record = None if status == "FAIL" and args.drop_failed_hard_violations else build_record(assignment, generated, metrics, warnings)
            return record, metrics

    # Last attempt with warnings.
    dummy_api = {}
    final_hard_warnings = hard_validation_warnings(last_warning_list)
    final_status = "FAIL" if final_hard_warnings else "WARN"
    metrics = {
        "artifact_id": assignment["artifact_id"],
        "category": assignment["category"],
        "year": assignment["year"],
        "era": assignment["era"],
        "source_type": assignment["source_type"],
        "attempt": args.max_retries + 1,
        "status": final_status,
        "warning_count": len(last_warning_list),
        "warnings_json": json.dumps(last_warning_list, ensure_ascii=False),
        "wall_clock_s": 0.0,
        "total_duration_s": 0.0,
        "load_duration_s": 0.0,
        "prompt_eval_count": 0,
        "prompt_eval_duration_s": 0.0,
        "eval_count": 0,
        "eval_duration_s": 0.0,
        "eval_tokens_per_s": 0.0,
        "wall_tokens_per_s": 0.0,
        "body_word_count": word_count(last_generated.get("body", "")),
        "model": args.model,
    }
    record = None if final_status == "FAIL" and args.drop_failed_hard_violations else build_record(assignment, last_generated, metrics, last_warning_list)
    return record, metrics


def build_options(args: argparse.Namespace) -> dict[str, Any]:
    options = dict(DEFAULT_OPTIONS)
    for arg_name, opt_key in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("num_ctx", "num_ctx"),
        ("num_predict", "num_predict"),
        ("repeat_penalty", "repeat_penalty"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            options[opt_key] = value
    return options


def parse_category_counts(counts_text: str | None) -> dict[str, int]:
    """Parse --counts like 'newspaper=50,school=10,religion=10'.

    Categories not listed fall back to --per-category.
    """
    if not counts_text:
        return {}

    out: dict[str, int] = {}
    for piece in counts_text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Bad --counts entry {piece!r}; use category=count.")
        key, value = piece.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key not in CATEGORIES:
            raise ValueError(f"Unknown category in --counts: {key!r}. Valid categories: {', '.join(CATEGORIES)}")

        n = int(value)
        if n < 0:
            raise ValueError(f"Count for {key} must be >= 0.")
        out[key] = n

    return out



# ---------------------------------------------------------------------
# Medium-run manifest expansion
# ---------------------------------------------------------------------

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

CATEGORY_PREFIX = {
    "newspaper": "NEWS",
    "school": "SCH",
    "religion": "REL",
    "business": "BUS",
    "photo_caption": "IMG",
    "oral_history": "OH",
    "council_minutes": "MIN",
}

def date_label(year: int, month: int, day: int) -> str:
    return f"{MONTH_NAMES[month]} {day}, {year}"

def active_choices(names: list[str], year: int, active_map: dict[str, tuple[int, int | None]]) -> list[str]:
    return [name for name in names if is_active(name, year, active_map)]

def pick_active_location(rng: random.Random, year: int, candidates: list[str], fallback: str = "Downtown/Main Street") -> str:
    active = active_choices(candidates, year, ACTIVE_LOCATIONS)
    if active:
        return rng.choice(active)
    if is_active(fallback, year, ACTIVE_LOCATIONS):
        return fallback
    return "Old Mill Bend"

_FORCED_YEAR_QUEUE: list[int] = []


def pick_year(rng: random.Random, category: str) -> int:
    global _FORCED_YEAR_QUEUE
    if _FORCED_YEAR_QUEUE:
        return _FORCED_YEAR_QUEUE.pop(0)
    if category == "oral_history":
        return rng.choice([1988, 1990, 1994, 1996, 2006, 2009, 2010, 2012, 2022, 2025])
    if category == "photo_caption":
        return rng.choice([1922, 1934, 1948, 1957, 1968, 1975, 1988, 1993, 1996, 2009, 2011, 2022, 2025])
    if category == "council_minutes":
        return rng.choice([1922, 1934, 1948, 1957, 1968, 1975, 1988, 1993, 1996, 2001, 2009, 2011, 2022, 2025])
    # Balanced spread across the whole archive.
    return rng.choice([1922, 1927, 1934, 1941, 1948, 1957, 1962, 1968, 1975, 1982, 1988, 1993, 1996, 2001, 2006, 2009, 2011, 2016, 2022, 2025])

def make_medium_newspaper_row(rng: random.Random, idx: int) -> dict[str, Any]:
    year = pick_year(rng, "newspaper")
    templates = []

    templates.append(("local news article", "Local Notes: Community Supper and Street Repairs",
                      "community supper and street repairs", "local news",
                      ["Mill Creek City Council", "Mill Creek Public Schools", "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission"],
                      ["Downtown/Main Street", "City Hall", "Mill Creek Public Schools", "Old Mill Bend"],
                      ["community supper or public schedule", "street, school, weather, or repair detail", "ordinary local consequence or instruction"],
                      180, 460))

    templates.append(("city government report", "Council Reviews Bridge and Street Work",
                      "bridge and street work", "city",
                      ["Mill Creek City Council", "Cobberland County Planning Commission"],
                      ["City Hall", "Council Chambers, City Hall", "Prairie River bridge", "Downtown/Main Street"],
                      ["council review", "bridge, street, estimate, vote, or hearing detail", "public works or schedule detail"],
                      220, 520))

    templates.append(("school report", "Public School Program Draws Family Crowd",
                      "school program", "school",
                      ["Mill Creek Public Schools"],
                      ["Mill Creek Public Schools", "Mill Creek High School", "Memorial Field"],
                      ["student program, concert, class display, or sports event", "teacher, schedule, score, or classroom detail", "school location detail"],
                      180, 480))

    templates.append(("farm-business report", "West Rows Producers Report Harvest Figures",
                      "farm production and business", "farm and business",
                      ["Mill Creek Commercial Club", "Olotón Foods", "Mill Creek City Council"],
                      ["West Rows farms", "grain elevator", "Downtown/Main Street", "Olotón Foods"],
                      ["harvest, grain, seed, price, yield, or shipment detail", "West Rows farms or grain elevator", "meeting, order, or market detail"],
                      200, 500))

    if year >= 1954:
        templates.append(("weather and flood report", "River Watch Continues Near South Flats",
                          "river level and flood watch", "local news",
                          ["Mill Creek City Council", "Prairie River Clinic"],
                          ["Prairie River bridge", "South Flats", "Prairie River Clinic", "Old Mill Bend"],
                          ["river level or spring thaw", "South Flats or bridge approach", "sandbag, water notice, inspection, or clinic detail"],
                          220, 560))

    if year >= 1988:
        templates.append(("public memory feature", "Historical Society Adds Items to Depot Display",
                          "public memory and archive display", "community",
                          ["Mill Creek Historical Society", "Mill Creek City Council", "Mill Creek Public Schools"],
                          ["Mill Creek Historical Society", "restored depot room", "Depot District"],
                          ["Historical Society or depot display", "photograph, ledger, rail lantern, map, or yearbook", "public visiting or donation detail"],
                          240, 580))

    if year >= 1996:
        templates.append(("local achievement feature", "Maizey Olotón Talks School, Corn, and Business",
                          "Maizey Olotón and local entrepreneurship", "business",
                          ["Olotón Foods", "Mill Creek Public Schools", "Cobberland Future Business Leaders Association"],
                          ["Mill Creek High School", "West Rows farms", "Olotón Foods", "Downtown/Main Street"],
                          ["Maizey Olotón", "student award, Olotón Foods, or West Rows farms", "one restrained corn-related pun or no pun at all"],
                          260, 560))

    if year >= 2009:
        templates.append(("redevelopment report", "Depot District Plan Receives Public Comment",
                          "redevelopment and preservation", "city",
                          ["Mill Creek City Council", "Mill Creek Historical Society", "Olotón Foods"],
                          ["Depot District", "City Hall", "Mill Creek Historical Society"],
                          ["Depot District plan", "public hearing, map, deadline, or review period", "rail-era structure or preservation detail"],
                          300, 620))

    if year >= 2022:
        templates.append(("modern local report", "Riverfront Trail Opens Along Former Buyout Land",
                          "Riverfront Trail and South Flats memory", "community",
                          ["Mill Creek City Council", "Mill Creek Historical Society", "CobberTech Extension Center"],
                          ["Riverfront Trail", "South Flats", "Mill Creek Historical Society", "CobberTech Extension Center"],
                          ["Riverfront Trail", "former South Flats buyout land", "historical sign, creek access, photo digitization, or public ceremony"],
                          300, 620))

    source_type, title, topic, section, inst, locs, details, mn, mx = rng.choice(templates)
    location = pick_active_location(rng, year, locs)
    return row(
        category="newspaper",
        artifact_id=f"{CATEGORY_PREFIX['newspaper']}_MED_{idx:05d}",
        year=year,
        date_label=date_label(year, rng.choice([2, 3, 4, 5, 6, 9, 10, 11]), rng.randint(1, 28)),
        collection="Mill Creek Newspaper Archive",
        source_type=source_type,
        title_seed=title,
        topic=topic,
        location=location,
        institutions=inst,
        people=people_for_year(year)[:4],
        locations=locs,
        required_details=details,
        tone="plain local newspaper prose appropriate to the year; concrete and not essay-like",
        word_count_min=mn,
        word_count_max=mx,
        min_people=1,
    ) | {"section": section}

def make_medium_school_row(rng: random.Random, idx: int) -> dict[str, Any]:
    year = pick_year(rng, "school")
    templates = [
        ("school newspaper note", "Students Prepare Spring Program", "spring school program", ["Mill Creek Public Schools", "Mill Creek High School"], ["student program", "classroom, assembly, auditorium, or gym detail", "teacher, song, display, or schedule detail"], 150, 420),
        ("club report", "School Club Reports on Community Project", "club service project", ["Mill Creek Public Schools", "Mill Creek High School"], ["club project", "student work detail", "school building or meeting detail"], 150, 430),
        ("sports summary", "School Team Opens Season", "school sports", ["Mill Creek Public Schools", "Mill Creek High School", "Memorial Field"], ["game, meet, practice, or score detail", "opponent, crowd, coach, or schedule detail", "school field, gym, or Memorial Field when active"], 150, 420),
        ("yearbook caption", "Students Gather for Class Photograph", "yearbook photograph", ["Mill Creek Public Schools", "Mill Creek High School"], ["class photograph", "visible clothing, room, banner, desk, or yearbook detail", "date estimate or caption note"], 70, 220),
    ]
    if year >= 1996:
        templates.append(("student profile", "Maizey Olotón and the Student Business Award", "student entrepreneurship", ["Mill Creek High School", "Mill Creek Public Schools"], ["Maizey Olotón", "award, business idea, display table, or classroom detail", "school recognition"], 180, 480))
    if year >= 2022:
        templates.append(("digital humanities lab note", "Students Add Captions to Digital Archive", "student digital archive work", ["CobberTech Extension Center", "Mill Creek Historical Society", "Mill Creek Public Schools"], ["scanning, metadata, or captioning detail", "photograph or newspaper clipping", "digital tools help but do not interpret by themselves"], 180, 480))

    source_type, title, topic, locs, details, mn, mx = rng.choice(templates)
    location = pick_active_location(rng, year, locs, "Mill Creek Public Schools")
    return row(
        category="school",
        artifact_id=f"{CATEGORY_PREFIX['school']}_MED_{idx:05d}",
        year=year,
        date_label=date_label(year, rng.choice([2, 3, 4, 5, 9, 10, 11]), rng.randint(1, 28)),
        collection="Mill Creek Public Schools Archive",
        source_type=source_type,
        title_seed=title,
        topic=topic,
        location=location,
        institutions=["Mill Creek Public Schools", "Mill Creek High School", "Mill Creek Historical Society", "CobberTech Extension Center", "Cobberland Future Business Leaders Association"],
        people=people_for_year(year)[:4],
        locations=locs,
        required_details=details,
        tone="school archive language appropriate to the year and source type",
        word_count_min=mn,
        word_count_max=mx,
        min_people=0,
    )

def make_medium_religion_row(rng: random.Random, idx: int) -> dict[str, Any]:
    year = pick_year(rng, "religion")
    templates = [
        ("church bulletin notice", "Harvest Supper Notice", "harvest supper", 10, ["Old Grange Hall", "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission"], ["harvest supper", "meal, time, ticket, or donation detail", "church or hall location"], 110, 320),
        ("service project notice", "Church Volunteers Organize Relief Collection", "relief collection", 11, ["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Downtown/Main Street"], ["relief collection", "clothing, food, supplies, or volunteer detail", "drop-off time or instruction"], 120, 340),
        ("choir program note", "Community Choir Program Announced", "choir program", 12, ["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Mill Creek Public Schools"], ["choir or hymn", "program time or song detail", "church or school location"], 120, 340),
        ("sermon_excerpt", "Sermon Excerpt on Work, Water, and Neighbors", "sermon excerpt", 4, ["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission"], ["neighborliness, work, flood, river, or memory theme", "concrete local reference", "short excerpt style"], 240, 620),
    ]
    if year >= 1957:
        templates.append(("service project notice", "Volunteers Asked to Help Near South Flats", "flood volunteer service", 4, ["South Flats", "Prairie River bridge", "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission"], ["flood cleanup or sandbagging", "South Flats or bridge approach", "reporting time or supply detail"], 120, 360))
    if year >= 1988:
        templates.append(("community announcement", "Congregations Encourage Visit to Depot Exhibit", "churches and public memory", 5, ["restored depot room", "Mill Creek Historical Society", "Depot District"], ["depot exhibit", "photograph, map, rail lantern, or yearbook", "visiting-hour or community note"], 120, 360))
    if year >= 2022:
        templates.append(("community blessing note", "Short Blessing Offered at Riverfront Trail Opening", "public ritual at trail opening", 6, ["Riverfront Trail", "South Flats"], ["Riverfront Trail opening", "former South Flats buyout land", "brief public words or blessing"], 120, 340))

    source_type, title, topic, month, locs, details, mn, mx = rng.choice(templates)
    location = pick_active_location(rng, year, locs, "St. Ansgar Lutheran Church")
    return row(
        category="religion",
        artifact_id=f"{CATEGORY_PREFIX['religion']}_MED_{idx:05d}",
        year=year,
        date_label=date_label(year, month, rng.randint(1, 28)),
        collection="Mill Creek Religious and Community Life Archive",
        source_type=source_type,
        title_seed=title,
        topic=topic,
        location=location,
        institutions=["St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission", "Mill Creek Women’s Aid Society", "Mill Creek Historical Society", "CobberTech Extension Center"],
        people=people_for_year(year)[:4],
        locations=locs,
        required_details=details,
        tone="religious community record appropriate to source type and year; respectful and concrete",
        word_count_min=mn,
        word_count_max=mx,
        min_people=0 if source_type != "sermon_excerpt" else 1,
    )

def make_medium_business_row(rng: random.Random, idx: int) -> dict[str, Any]:
    year = pick_year(rng, "business")
    templates = [
        ("advertisement", "Main Street Store Announces Sale", "local advertisement", ["Downtown/Main Street", "West Rows farms", "grain elevator"], ["price, hours, order deadline, pickup, or display detail", "Main Street, farm, seed, grain, or store detail"], 60, 220),
        ("classified_notice", "Classified Notice", "classified notice", ["Downtown/Main Street", "City Hall"], ["item, room, sale, lost object, service, or contact detail", "brief classified style"], 50, 180),
        ("market_report", "Cream, Egg, and Grain Prices Posted", "market report", ["grain elevator", "West Rows farms", "Downtown/Main Street"], ["price, shipment, yield, or receipt detail", "farm, elevator, or market detail"], 80, 260),
        ("public_notice", "City Public Notice", "public notice", ["City Hall", "Council Chambers, City Hall", "Prairie River bridge"], ["date, time, office, street, bridge, or instruction detail", "official notice style"], 90, 300),
    ]
    if year >= 1968:
        templates.append(("public hearing notice", "Public Hearing Notice", "planning hearing", ["City Hall", "Downtown/Main Street", "Depot District"], ["public hearing", "map, comment date, permit, route, or zoning detail"], 100, 320))
    if year >= 2001:
        templates.append(("business notice", "Olotón Foods Posts Local Notice", "Olotón Foods business notice", ["Olotón Foods", "West Rows farms"], ["Olotón Foods", "job, production, sample, order, or schedule detail", "restrained corn pun or no pun"], 90, 300))
    if year >= 2022:
        templates.append(("archive lab notice", "Digital Archive Requests Business Advertisements", "business ads in digital archive", ["CobberTech Extension Center", "Mill Creek Historical Society"], ["scanning, metadata, clipping, advertisement, or drop-off detail", "archive contact or schedule detail"], 120, 340))

    source_type, title, topic, locs, details, mn, mx = rng.choice(templates)
    location = pick_active_location(rng, year, locs, "Downtown/Main Street")
    return row(
        category="business",
        artifact_id=f"{CATEGORY_PREFIX['business']}_MED_{idx:05d}",
        year=year,
        date_label=date_label(year, rng.choice([1, 2, 3, 4, 5, 9, 10, 11, 12]), rng.randint(1, 28)),
        collection="Mill Creek Business, Notices, and Directories Archive",
        source_type=source_type,
        title_seed=title,
        topic=topic,
        location=location,
        institutions=["Mill Creek Commercial Club", "Mill Creek City Council", "Olotón Foods", "Mill Creek Historical Society", "CobberTech Extension Center"],
        people=people_for_year(year)[:4],
        locations=locs,
        required_details=details,
        tone="ordinary business or public-notice prose appropriate to the year; concrete and not literary",
        word_count_min=mn,
        word_count_max=mx,
        min_people=0,
        min_institutions=0 if source_type in {"advertisement", "classified_notice", "market_report"} else 1,
    )

def make_medium_photo_row(rng: random.Random, idx: int) -> dict[str, Any]:
    year = pick_year(rng, "photo_caption")
    templates = [
        ("photo_caption", "Main Street Scene", "street photograph", ["Downtown/Main Street"], ["storefront, wagon, car, sign, sidewalk, window, or street detail", "date estimate or donor note"], 50, 180),
        ("photo_caption", "Prairie River Bridge Photograph", "bridge photograph", ["Prairie River bridge"], ["bridge, riverbank, water level, railing, or road detail", "caption note or uncertainty"], 50, 180),
        ("photo_caption", "West Rows Farm Scene", "farm photograph", ["West Rows farms", "grain elevator"], ["field, seed, wagon, truck, grain, elevator, or harvest detail", "date estimate or donor note"], 50, 180),
    ]
    if year >= 1948:
        templates.append(("photo_caption", "Memorial Field Ceremony", "Memorial Field photo", ["Memorial Field"], ["Memorial Field", "crowd, flag, band, bleachers, plaque, or team detail"], 50, 180))
    if year >= 1957:
        templates.append(("newspaper_photo_cutline", "Floodwater Near South Flats", "flood cutline", ["South Flats", "Prairie River bridge"], ["floodwater", "sandbags, bridge approach, waterline, or volunteers", "South Flats"], 45, 160))
    if year >= 1988:
        templates.append(("exhibit_label", "Object in Restored Depot Room", "depot exhibit label", ["restored depot room", "Mill Creek Historical Society"], ["rail lantern, ledger, map, photograph, ticket, or yearbook", "display case or donor note"], 60, 200))
    if year >= 2022:
        templates.append(("photo_caption", "Digital Archive Lab Photograph", "digitization photo", ["CobberTech Extension Center", "Mill Creek Historical Society"], ["scanner, computer, folders, clipping, photograph, or metadata worksheet", "digital archive lab"], 60, 180))

    source_type, title, topic, locs, details, mn, mx = rng.choice(templates)
    location = pick_active_location(rng, year, locs)
    return row(
        category="photo_caption",
        artifact_id=f"{CATEGORY_PREFIX['photo_caption']}_MED_{idx:05d}",
        year=year,
        date_label=f"circa {year}",
        collection="Mill Creek Historical Society Image and Object Archive",
        source_type=source_type,
        title_seed=title,
        topic=topic,
        location=location,
        institutions=["Mill Creek Historical Society", "Mill Creek City Council", "Mill Creek Public Schools", "CobberTech Extension Center"],
        people=people_for_year(year)[:4],
        locations=locs,
        required_details=details,
        tone="concise archive caption or exhibit-label prose; concrete and careful about uncertainty",
        word_count_min=mn,
        word_count_max=mx,
        min_institutions=0,
        min_people=0,
    )

def make_medium_oral_history_row(rng: random.Random, idx: int) -> dict[str, Any]:
    year = pick_year(rng, "oral_history")
    templates = [
        ("Lena Voss Remembers the Reading Room and Depot", "public memory and reading room", "Mill Creek Historical Society", ["Lena Voss", "Professor Carrel Englekorn", "Clara Hestvik"], ["Main Street reading room", "restored depot room", "Mill Creek Historical Society"], ["Main Street reading room", "restored depot room object", "what the town chooses to remember"]),
        ("Peter Harlan on Flood Work and South Flats", "flood memory and public works", "South Flats", ["Peter Harlan", "Ruth Ellingson", "Dr. Helen Markham"], ["South Flats", "Prairie River bridge", "Old Mill Bend"], ["1957 flood memory", "bridge approaches or sandbags", "public works or water detail"]),
        ("Ruth Ellingson on Memorial Field and Records", "school records and public ceremony", "Mill Creek Historical Society", ["Ruth Ellingson", "Professor Carrel Englekorn", "Coach Harold Bratten"], ["Mill Creek Historical Society", "Memorial Field", "Mill Creek Public Schools"], ["Memorial Field dedication", "plaque or record-keeping detail", "school band, opening game, or ceremony"]),
        ("Maizey Olotón on West Rows and Business", "agriculture, school, and entrepreneurship", "West Rows farms", ["Maizey Olotón", "Asha Patel", "David Harlan"], ["West Rows farms", "Olotón Foods", "Mill Creek High School"], ["1996 state entrepreneurship award", "West Rows farms", "Olotón Foods"]),
        ("Professor Englekorn on Local History and Partial Archives", "archive bias and digitization", "Mill Creek Historical Society", ["Professor Carrel Englekorn", "Asha Patel", "Lena Voss"], ["Mill Creek Historical Society", "Riverfront Trail", "South Flats"], ["digitization of photographs or records", "even good history is partial", "Riverfront Trail or South Flats"]),
        ("Nora Reyes on Scanning Photographs and Missing Records", "student digitization and missing records", "CobberTech Extension Center", ["Nora Reyes", "Professor Carrel Englekorn", "Asha Patel"], ["CobberTech Extension Center", "Mill Creek Historical Society"], ["scanning photographs or newspaper clippings", "noticing what is missing from the archive", "digital tools help but do not interpret by themselves"]),
    ]
    title, topic, location, people, locs, details = rng.choice(templates)
    location = pick_active_location(rng, year, locs, "Mill Creek Historical Society")
    return row(
        category="oral_history",
        artifact_id=f"{CATEGORY_PREFIX['oral_history']}_MED_{idx:05d}",
        year=year,
        date_label=date_label(year, rng.choice([2, 4, 5, 6, 8, 9, 10, 11]), rng.randint(1, 28)),
        collection="Mill Creek Oral History Collection",
        source_type="oral_history_transcript",
        title_seed=title,
        topic=topic,
        location=location,
        institutions=["Mill Creek Historical Society", "CobberTech Extension Center", "Mill Creek Public Schools", "Olotón Foods"],
        people=people,
        locations=locs,
        required_details=details,
        tone="oral history interview, concrete, reflective, and not too polished",
        word_count_min=600,
        word_count_max=1050,
        min_people=1,
    )

def make_medium_council_minutes_row(rng: random.Random, idx: int) -> dict[str, Any]:
    year = pick_year(rng, "council_minutes")
    templates = [
        ("Council Minutes on Streets and Public Works", "street and public works", ["City Hall", "Council Chambers, City Hall"], ["street grading, bridge inspection, budget, or water detail", "motion or vote", "attendance and adjournment"], 280, 700),
        ("Council Minutes on School and Civic Program", "school and civic program", ["City Hall", "Council Chambers, City Hall", "Mill Creek Public Schools"], ["school, ceremony, program, or public-use detail", "motion or outcome", "attendance and adjournment"], 280, 700),
        ("Council Minutes on Flood and River Conditions", "flood and river conditions", ["City Hall", "Council Chambers, City Hall", "Prairie River bridge", "South Flats"], ["river level, bridge, sandbags, water notice, or floodplain detail", "motion or action item", "attendance and adjournment"], 300, 760),
    ]
    if year >= 1968:
        templates.append(("Council Minutes on Planning and Downtown Questions", "planning and downtown", ["City Hall", "Council Chambers, City Hall", "Downtown/Main Street", "Depot District"], ["planning, bypass, redevelopment, parking, or public hearing detail", "motion or tabled item", "attendance and adjournment"], 300, 780))
    if year >= 1988:
        templates.append(("Council Minutes on Historical Society and Archive Work", "public memory and archive", ["Council Chambers, City Hall", "Mill Creek Historical Society", "restored depot room"], ["Historical Society, exhibit, photograph, map, ledger, or depot detail", "motion or authorization", "attendance and adjournment"], 300, 780))
    if year >= 2022:
        templates.append(("Council Minutes on Riverfront Trail and Digitization", "Riverfront Trail and digitization", ["Council Chambers, City Hall", "Riverfront Trail", "CobberTech Extension Center"], ["Riverfront Trail", "South Flats buyout land or historical sign", "CobberTech digitization pilot or archive detail"], 300, 800))
    title, topic, locs, details, mn, mx = rng.choice(templates)
    location = pick_active_location(rng, year, locs, "Council Chambers, City Hall")
    return row(
        category="council_minutes",
        artifact_id=f"{CATEGORY_PREFIX['council_minutes']}_MED_{idx:05d}",
        year=year,
        date_label=date_label(year, rng.choice([1, 3, 4, 5, 8, 9, 10, 11]), rng.choice([3, 7, 12, 16, 21, 24])),
        collection="Mill Creek City Council Minutes Archive",
        source_type="city_council_minutes",
        title_seed=title,
        topic=topic,
        location=location,
        institutions=["Mill Creek City Council", "Mill Creek Public Schools", "Prairie River Clinic", "Mill Creek Historical Society", "CobberTech Extension Center", "Olotón Foods"],
        people=people_for_year(year)[:4],
        locations=locs,
        required_details=details,
        tone="formal but plain municipal record language",
        word_count_min=mn,
        word_count_max=mx,
        min_people=1,
    )


# ---------------------------------------------------------------------
# V8 manifest diversification
# ---------------------------------------------------------------------

V8_EVENT_YEARS = [1957, 1993, 2011, 2022]

V8_TITLE_VARIANTS: dict[str, dict[str, list[str]]] = {
    "newspaper": {
        "bridge and street work": [
            "Bridge Committee Reviews Spring Repairs", "Street Grading Schedule Sent to Council",
            "Bridge Approach Inspection Draws Public Questions", "Council Hears Estimate for Road and Bridge Work",
            "Main Street Drainage Work Put on Calendar", "Prairie River Bridge Railing Repairs Discussed",
            "Public Works Crew Lists Week's Street Needs", "Council Tables Gravel Purchase After Debate",
            "Bridge Plank Report Sent to City Hall", "Street Lamp and Culvert Work Reviewed",
        ],
        "school program": [
            "School Program Fills Assembly Room", "Pupils Present Music and Class Displays",
            "Families Attend Public School Evening", "Student Recitation Program Draws Crowd",
            "School Exhibit Shows Maps, Copybooks, and Models", "High School Program Includes Band and Readings",
            "Parents Visit Classrooms During Open House", "Spring Program Features Choir and Student Work",
            "School Night Combines Music and Science Displays", "Students Host Community Program at the School",
        ],
        "farm production and business": [
            "Elevator Posts Week's Grain Receipts", "West Rows Farmers Compare Seed Results",
            "Cream and Grain Shipments Rise After Clear Weather", "Elevator Crew Reports Busy Harvest Week",
            "Farmers Discuss Prices at Main Street Counter", "West Rows Producers Watch Corn and Oat Yields",
            "Grain Tickets Show Strong Receipts from West Rows", "Seed Orders and Market Prices Mark Spring Trade",
            "Harvest Loads Keep Elevator Platform Busy", "Farm Report Notes Rain, Rust, and Rail Cars",
        ],
        "community supper and street repairs": [
            "Community Supper Planned as Street Work Continues", "Local Notes Include Supper, Repairs, and School News",
            "Volunteers Set Tables While City Crew Grades Street", "Neighborhood Supper Added to Busy Civic Week",
            "Church Hall Meal to Aid Street Fund", "Main Street Notes: Supper, Sidewalks, and Weather",
            "Repair Crew and Supper Committee Both Seek Help", "Local Calendar Lists Meal, Meeting, and Road Work",
            "Residents Asked to Note Detours Before Supper", "Supper Committee Announces Menu and Work Bee",
        ],
        "river level and flood watch": [
            "River Gauge Watched Near South Flats", "Sandbag Crew Called to Bridge Approach",
            "South Flats Families Watch Waterline", "Prairie River Inspection Continues After Rain",
            "Clinic and Council Share Flood Notice", "Water Nears Low Lots Along South Flats",
            "Bridge Road Checked as River Rises", "Flood Watch Brings Volunteers to Riverbank",
            "South Flats Residents Prepare for High Water", "River Notice Posted at City Hall and Clinic",
        ],
        "public memory and archive display": [
            "Depot Display Adds Ledger and Class Photograph", "Historical Society Labels New Rail Lantern Exhibit",
            "Old Map and Yearbook Added to Depot Room", "Archive Volunteers Identify Main Street Photograph",
            "Society Seeks Names for Depot Display", "Restored Depot Case Holds New Local Items",
            "Students Help Label Photographs for Historical Society", "Public Invited to Review Archive Table at Depot",
            "Ledger, Ticket Stub, and School Photo Join Display", "Historical Society Records Donor Notes for Exhibit",
        ],
        "Maizey Olotón and local entrepreneurship": [
            "Maizey Olotón Presents Student Business Project", "Student Corn Project Wins State Notice",
            "High School Senior Connects Farm Work and Business", "Maizey Olotón Discusses West Rows Enterprise Idea",
            "Student Award Honors Farm-Business Plan", "Future Business Group Recognizes Mill Creek Student",
            "Maizey Olotón Displays Crop-Marketing Project", "School and Farm Roots Shape Student Business Award",
        ],
        "redevelopment and preservation": [
            "Depot District Plan Draws Evening Comment", "Public Reviews Map for Depot District Work",
            "Redevelopment Hearing Balances Parking and Preservation", "City Receives Comments on Old Rail Blocks",
            "Depot Plan Pairs Storefront Repairs with Historic Signs", "Council Reviews Downtown Redevelopment Timetable",
            "Residents Ask About Traffic Near Depot District", "Plan for Depot Blocks Moves to Comment Period",
        ],
        "Riverfront Trail and South Flats memory": [
            "Riverfront Trail Opens on Former Buyout Land", "Trail Signs Recall South Flats Flood History",
            "New River Path Links Recreation and Memory", "Residents Walk Former Flats at Trail Opening",
            "Historical Markers Placed Along Riverfront Trail", "Trail Ceremony Notes Floodplain's New Public Use",
            "Students Scan Flood Photos for Trail Display", "Former South Flats Lots Become Public River Path",
        ],
    },
    "school": {
        "spring school program": ["Spring Program Lists Songs and Student Displays", "Classroom Exhibit Prepared for Family Night", "Students Rehearse Readings for Public Program", "School Program Notes Maps, Music, and Models", "Parents Invited to View Student Work"],
        "club service project": ["Student Club Plans Community Service Afternoon", "Club Members Sort Archive Clippings", "Students Prepare Signs for Civic Project", "Service Club Reports Work at Depot Room", "Class Committee Collects Flood Memories"],
        "school sports": ["School Team Opens Season on Athletic Field", "Baseball Squad Practices Behind School", "Track Team Prepares for County Meet", "Students Report Close Game at School Grounds", "Coach Notes Strong Turnout for Practice"],
        "yearbook photograph": ["Class Photograph Taken Near Assembly Room", "Yearbook Staff Labels Student Group Picture", "Students Pose Beside Classroom Banner", "Undated Class Photo Added to School File", "Yearbook Caption Notes Clothing and Desks"],
        "student digital archive work": ["Students Add Metadata to Archive Photographs", "Digital Lab Class Compares Captions and Evidence", "Student Archivists Scan Flood Clippings", "Class Builds Finding Aid for Local History", "Archive Lab Notes What the Photograph Does Not Show"],
    },
    "photo_caption": {
        "street photograph": ["Storefront Window on Main Street", "Wagon and Automobile Near Dry Goods Store", "Sidewalk View After Spring Rain", "Main Street Corner with Hand-Painted Sign", "Unidentified Clerk Outside Shop Door", "North-Looking View of Downtown Sidewalk"],
        "bridge photograph": ["Prairie River Bridge After High Water", "Bridge Railing and South Bank", "County Engineer's Bridge Approach Photograph", "Children Watching Water Near the Bridge", "Gravel Washed Along Bridge Road", "Undated Bridge View from East Bank"],
        "farm photograph": ["Wagon Loading at West Rows Field", "Grain Truck Beside Elevator Platform", "Threshing Crew Near North Edge of Field", "Seed Sacks Stacked by Elevator Door", "Late Summer Harvest View", "Farm Ledger Photograph with Grain Ticket"],
        "Memorial Field photo": ["Bleachers at Memorial Field Dedication", "Band and Flag at Memorial Field", "Team Photograph Near New Field Sign", "Plaque Table at Field Ceremony", "Crowd Along First-Base Line"],
        "flood cutline": ["Sandbags Along South Flats Road", "Waterline at Bridge Approach", "Volunteers Filling Bags Near Riverbank", "Flooded Yard at Edge of South Flats", "Clinic Notice Board During High Water"],
        "depot exhibit label": ["Rail Lantern in Restored Depot Case", "Ledger Open to Grain Receipt Page", "Depot Map with Pencil Notes", "Ticket Stub and Station Photograph", "Yearbook Page in Archive Display"],
        "digitization photo": ["Students Scanning Chronicle Clippings", "Metadata Worksheet Beside Photograph Box", "Archive Lab Table with Gloves and Folders", "Digital Camera Stand Over Ledger", "Volunteers Comparing Captions"],
    },
}


V8_TITLE_VARIANTS.update({
    "business": {
        "local advertisement": [
            "Main Street Store Lists Spring Specials", "Seed Orders Taken at Downtown Counter", "Hardware Window Displays New Farm Tools",
            "Grocer Announces Saturday Prices", "Dry Goods Shop Advertises School Week Sale", "Feed and Seed Notice Posted for West Rows Farmers",
            "Merchant Offers Delivery After Market Day", "Store Window Features Harvest Supplies", "Main Street Merchant Extends Evening Hours", "Farm Account Books Available at Local Shop",
        ],
        "classified notice": [
            "Classified: Lost Key Ring Near Depot", "Classified: Room Offered Near Main Street", "Classified: Wagon For Sale After Harvest",
            "Classified: Sewing Work Accepted", "Classified: Bicycle Found by Bridge", "Classified: Piano Lessons Offered After School",
            "Classified: Used Desk Wanted", "Classified: Farmhand Seeks Work", "Classified: Stove and Chairs For Sale", "Classified: Notice of Found Ledger",
        ],
        "market report": [
            "Elevator Posts Cream, Egg, and Grain Prices", "Market Board Lists Wheat and Egg Receipts", "Grain Ticket Totals Announced for Week",
            "Cream Station Reports Saturday Prices", "West Rows Market Figures Posted", "Elevator Notes Corn, Oats, and Cream Receipts",
            "Produce Prices Updated at Main Street Office", "Weekly Farm Prices Entered in Ledger", "Shipment Receipts Listed by Elevator Clerk", "Market Report Notes Rail Car Delay",
        ],
        "public notice": [
            "Public Notice: Bridge Repair Bids Requested", "City Notice Lists Street Work Date", "Public Notice Posted for Water Main Work",
            "Notice of Hearing at City Hall", "Bridge Approach Closure Notice", "City Clerk Announces Permit Deadline",
            "Public Notice on Sidewalk Assessment", "Zoning Notice Posted for Depot Blocks", "Council Notice Lists Map Review Hours", "Notice to Property Owners Near River",
        ],
        "planning hearing": [
            "Hearing Notice for Depot District Map", "Public Hearing Scheduled on Downtown Parking", "Planning Notice Lists Bypass Questions",
            "Zoning Hearing Set for Old Rail Lots", "Notice of Comment Period on Redevelopment Plan", "Depot Blocks Hearing Announced",
            "Planning Commission Seeks Written Comments", "Public Notice on Storefront Repair District", "City Hall Hearing to Review Route Map", "Redevelopment Notice Lists Inspection Hours",
        ],
        "Olotón Foods business notice": [
            "Olotón Foods Posts Harvest Schedule", "Olotón Foods Lists Sampling Hours", "Corn Kitchen Announces Local Hiring Notice",
            "Olotón Foods Requests West Rows Delivery Times", "Olotón Foods Posts Community Tour Notice", "Production Notice from Olotón Foods",
            "Olotón Foods Seeks Seasonal Help", "Notice to Growers from Olotón Foods", "Olotón Foods Announces Test Batch Day", "Factory Office Posts Order Deadline",
        ],
        "business ads in digital archive": [
            "Archive Lab Requests Old Business Advertisements", "Digital Project Seeks Main Street Clippings", "Historical Society Asks for Store Ledgers",
            "CobberTech Scan Day Announced for Business Records", "Archive Notice Seeks Classified Pages", "Old Advertisements Wanted for Digital Collection",
        ],
    },
    "religion": {
        "harvest supper": [
            "Harvest Supper Tickets Available After Service", "Church Hall Supper Lists Menu and Times", "Annual Supper Committee Seeks Pies",
            "St. Ansgar Announces Harvest Meal", "Sacred Heart Volunteers Prepare Supper Tables", "Harvest Supper Proceeds Marked for Relief Fund",
            "Supper Notice Lists Serving Hours", "Church Kitchen Plans Fall Meal", "Choir Members to Help at Harvest Supper", "Old Grange Hall Reserved for Church Supper",
        ],
        "relief collection": [
            "Church Volunteers Collect Coats and Flour", "Relief Boxes to Be Packed After Service", "Congregations Ask for Bedding and Canned Goods",
            "Volunteer Notice Lists Drop-Off Hours", "Church Basement Open for Relief Sorting", "Young People Asked to Carry Relief Parcels",
            "Relief Collection Set for Saturday Morning", "Donation Table Placed Near Church Door", "Service Committee Requests Clean Blankets", "Food and Clothing Drive Announced",
        ],
        "choir program": [
            "Community Choir Program Lists Hymns", "Choir Rehearsal Set for Thursday Evening", "Sacred Heart and St. Ansgar Plan Joint Music",
            "Church Choir to Sing at School Hall", "Advent Program Includes Student Voices", "Choir Notice Lists Practice Time",
            "Hymn Program Prepared for Public Evening", "Youth Choir Adds Two Selections", "Community Sing Announced at Church", "Choir Program to Benefit Relief Fund",
        ],
        "sermon excerpt": [
            "Sermon Excerpt on River, Work, and Neighbor", "Sermon Notes Stewardship During High Water", "Excerpt on Fields, Floods, and Mutual Care",
            "Sermon Passage Recalls Bridge and Neighbor Duty", "Pastor Reflects on Work and Water", "Sermon Excerpt on Memory and Service",
            "Homily Notes the River's Warning", "Sermon Text on Charity After Rain", "Excerpt on Public Burdens and Private Help", "Sunday Message on Labor and Mercy",
        ],
        "flood volunteer service": [
            "Church Volunteers Asked to Fill Sandbags", "South Flats Cleanup Crew to Meet at Church", "Congregations Organize High-Water Watch",
            "Relief Team Lists Flood Work Schedule", "Church Basement Open for Flood Supplies", "Volunteer Notice for Bridge Approach Sandbags",
            "Prayer and Work Crew Planned Near River", "Flood Cleanup Call Issued After Service", "Sandbag Team Seeks Trucks and Shovels", "South Flats Relief Table Set Up at Church",
        ],
        "churches and public memory": [
            "Congregations Encourage Visit to Depot Exhibit", "Church Bulletin Notes Historical Society Display", "Choir Program to Include Depot Memories",
            "Service Committee Donates Photograph to Archive", "Churches Invite Members to Identify Old Pictures", "Depot Exhibit Hours Listed in Bulletin",
            "Memory Table Set Up After Sunday Service", "Church Notice Seeks Names for Old Class Photo", "Congregations Share Records with Historical Society", "Archive Display Connects Church and Town Service",
        ],
        "public ritual at trail opening": [
            "Short Blessing Planned for Riverfront Trail", "Clergy Offer Words at Trail Opening", "Congregations Mark Former South Flats Path",
            "Trail Ceremony Includes Prayer and Flood Memory", "Church Volunteers Staff Riverfront Table", "Blessing Note Recalls Buyout Land",
        ],
    },
    "council_minutes": {
        "street and public works": ["Council Minutes on Street Grading and Bridge Inspection", "Council Minutes on Culvert, Gravel, and Road Work", "Minutes of Public Works Committee", "Council Record of Bridge Repair Estimate", "Minutes on Sidewalks, Drainage, and Street Lamps"],
        "school and civic program": ["Council Minutes on School Use and Public Program", "Minutes on Band Concert and School Grounds", "Council Record on Student Civic Program", "Minutes on School Request and Public Hall", "Council Notes on Youth Service Project"],
        "flood and river conditions": ["Council Minutes on River Stage and South Flats", "Minutes on Sandbags and Bridge Closure", "Council Record of Flood Watch Actions", "Special Minutes on High Water Response", "Minutes on South Flats Cleanup and Inspection"],
        "planning and downtown": ["Council Minutes on Depot District Planning", "Minutes on Downtown Parking and Storefronts", "Public Hearing Minutes on Bypass Effects", "Council Record on Redevelopment Map", "Minutes on Depot Blocks and Public Comment"],
        "public memory and archive": ["Council Minutes on Historical Society Request", "Minutes on Depot Exhibit Authorization", "Council Record on Archive Donation", "Minutes on Preservation and Public Display", "Historical Society Agenda Item in Council Minutes"],
        "Riverfront Trail and digitization": ["Council Minutes on Riverfront Trail Signs", "Minutes on Trail Opening and Archive Project", "Council Record on South Flats Buyout Land", "Minutes on CobberTech Digitization Pilot", "Riverfront Trail and Historical Marker Minutes"],
    },
})

V8_SOURCE_TYPE_VARIANTS: dict[str, list[str]] = {
    "newspaper": ["local news article", "brief local report", "front-page local item", "community column", "city desk report", "feature note"],
    "school": ["school newspaper note", "yearbook note", "club report", "program note", "sports summary", "archive worksheet"],
    "religion": ["church bulletin notice", "service project notice", "sermon excerpt", "choir program note", "community announcement"],
    "business": ["advertisement", "classified notice", "market report", "public notice", "business directory note", "meeting notice"],
    "photo_caption": ["photo caption", "archive caption", "newspaper photo cutline", "object label", "exhibit label", "catalog note"],
    "council_minutes": ["city_council_minutes", "committee_minutes", "public_hearing_minutes", "special_meeting_minutes"],
}

V8_ERA_TEXTURES = [
    "include a small material detail appropriate to the year, such as paper forms, wagons, automobiles, ledgers, radio notices, photocopies, computers, scanners, or printed handouts",
    "include one ordinary consequence for residents, students, clerks, farmers, church volunteers, or business owners",
    "include a concrete object, sign, room, street corner, table, ledger, photograph, tool, or posted notice",
]

V8_DISPLACEMENT_DETAILS = [
    "relocation, buyout, moved furniture, temporary housing, emptied lot, school absence, clinic record, or family leaving South Flats",
    "show displacement indirectly through an address change, missing classroom seat, church relief note, or photograph of a cleared lot",
]

V8_SCHOOL_MEMORY_DETAILS = [
    "student archive worksheet, yearbook note, exhibit label, oral-history question, caption debate, or missing name in a photograph",
    "show students comparing public memory with incomplete records",
]


def v8_year_sequence(category: str, requested: int) -> list[int]:
    """Even distribution plus deliberate event bursts for interpretation."""
    if category == "oral_history":
        base_years = [1988, 1990, 1994, 1996, 2006, 2009, 2010, 2012, 2022, 2025]
        eventish = [1988, 1993, 2009, 2011, 2012, 2022, 2025]
    else:
        base_years = list(range(1920, 2027))
        eventish = V8_EVENT_YEARS

    if requested <= 0:
        return []

    event_count = max(0, round(requested * 0.16)) if requested >= 40 else 0
    base_count = requested - event_count
    seq = even_year_sequence(base_years, base_count)

    if event_count:
        for i in range(event_count):
            seq.append(eventish[i % len(eventish)])
    seq.sort()
    return seq


def v8_pick_variant(rng: random.Random, category: str, topic: str, fallback: str) -> str:
    variants = V8_TITLE_VARIANTS.get(category, {}).get(topic)
    if variants:
        return rng.choice(variants)
    return fallback


def v8_active_cleanup(a: dict[str, Any]) -> dict[str, Any]:
    """Remove known anachronistic prompt material before generation."""
    year = int(a["year"])
    if year < 2001:
        for key in ["title_seed", "topic", "location"]:
            a[key] = str(a.get(key, "")).replace("Olotón Foods", "West Rows farms")
        a["allowed_institutions"] = [x for x in a["allowed_institutions"] if x != "Olotón Foods"]
        a["allowed_locations"] = [x if x != "Olotón Foods" else "West Rows farms" for x in a["allowed_locations"]]
        a["required_details"] = [str(d).replace("Olotón Foods", "student business project") for d in a["required_details"]]
    if year < 1948:
        for key in ["title_seed", "topic", "location"]:
            a[key] = str(a.get(key, "")).replace("Memorial Field", "school athletic field")
        a["allowed_locations"] = [x for x in a["allowed_locations"] if x != "Memorial Field"]
        if "Mill Creek Public Schools" not in a["allowed_locations"]:
            a["allowed_locations"].append("Mill Creek Public Schools")
        a["required_details"] = [str(d).replace("Memorial Field", "school athletic field") for d in a["required_details"]]
    if year < 1988 and a["category"] != "photo_caption":
        a["allowed_institutions"] = [x for x in a["allowed_institutions"] if x != "Mill Creek Historical Society"]
        a["allowed_locations"] = [x for x in a["allowed_locations"] if x != "Mill Creek Historical Society"]
    return a


def v8_diversify_assignment(a: dict[str, Any], rng: random.Random, idx: int, requested: int) -> dict[str, Any]:
    """V8 assignment polish without changing the core generator."""
    a = dict(a)
    category = a["category"]
    topic = a.get("topic", "")
    a["title_seed"] = v8_pick_variant(rng, category, topic, a.get("title_seed", ""))

    # More varied source_type labels while preserving source_group for ML.
    if category in V8_SOURCE_TYPE_VARIANTS:
        # Do not obscure oral-history source type; it is valuable as-is.
        if category != "oral_history" and rng.random() < 0.55:
            a["source_type"] = rng.choice(V8_SOURCE_TYPE_VARIANTS[category])

    # Enrich prompts with small material/era details.
    details = list(a.get("required_details", []))
    if rng.random() < 0.75:
        details.append(rng.choice(V8_ERA_TEXTURES))

    year = int(a["year"])
    low_topic = str(topic).lower()
    if year in {1957, 1993, 2011, 2022} and category in {"newspaper", "school", "religion", "photo_caption", "council_minutes", "oral_history"}:
        if rng.random() < 0.60:
            details.append(rng.choice(V8_DISPLACEMENT_DETAILS))
            if "South Flats" not in a["allowed_locations"] and is_active("South Flats", year, ACTIVE_LOCATIONS):
                a["allowed_locations"].append("South Flats")

    if category in {"school", "photo_caption", "newspaper", "oral_history"} and year >= 1988:
        if rng.random() < 0.35:
            details.append(rng.choice(V8_SCHOOL_MEMORY_DETAILS))
            if is_active("Mill Creek Historical Society", year, ACTIVE_INSTITUTIONS) and "Mill Creek Historical Society" not in a["allowed_institutions"]:
                a["allowed_institutions"].append("Mill Creek Historical Society")

    # If the row is about Maizey before the company exists, make it a student/farm-business story.
    if year < 2001 and "Maizey" in str(a.get("title_seed", "")):
        a["topic"] = "Maizey Olotón student entrepreneurship and West Rows farm roots"
        if year < 1996:
            a["title_seed"] = "Student Business Club Discusses Farm Marketing"
            a["topic"] = "student business club and farm marketing"
        details = [str(d).replace("Olotón Foods", "student business project") for d in details]

    # Add source group after source_type variation.
    a["source_group"] = source_group_for(a["category"], a["source_type"])
    a["required_details"] = details[:7]
    a = v8_active_cleanup(a)
    a["source_group"] = source_group_for(a["category"], a["source_type"])
    return a

MEDIUM_ROW_BUILDERS = {
    "newspaper": make_medium_newspaper_row,
    "school": make_medium_school_row,
    "religion": make_medium_religion_row,
    "business": make_medium_business_row,
    "photo_caption": make_medium_photo_row,
    "oral_history": make_medium_oral_history_row,
    "council_minutes": make_medium_council_minutes_row,
}

def even_year_sequence(years: list[int], requested: int) -> list[int]:
    """Return a deterministic, evenly distributed year sequence.

    For example, 300 council minutes over 1920-2026 gives each year
    either 2 or 3 artifacts, never a random pile-up in one year.
    """
    if requested <= 0:
        return []
    if not years:
        raise ValueError("years must not be empty")

    years = sorted(years)
    base, extra = divmod(requested, len(years))

    sequence: list[int] = []
    for year in years:
        sequence.extend([year] * base)

    if extra:
        if extra == 1:
            extra_indices = [len(years) // 2]
        else:
            extra_indices = [
                round(i * (len(years) - 1) / (extra - 1))
                for i in range(extra)
            ]
        for idx in extra_indices:
            sequence.append(years[idx])

    sequence.sort()
    return sequence


def years_for_medium_category(category: str) -> list[int]:
    """Years used by the medium-run builders.

    Council minutes are intentionally annual across the full 1920-2026 span.
    Oral histories are intentionally concentrated in the later archival/interview era.
    Other categories are spread across the full project period.
    """
    if category == "oral_history":
        return [1988, 1990, 1994, 1996, 2006, 2009, 2010, 2012, 2022, 2025]
    return list(range(1920, 2027))

def make_medium_manifest(category: str, requested: int, seed: int = 1776) -> list[dict[str, Any]]:
    rng = random.Random(seed + sum(ord(ch) for ch in category))
    builder = MEDIUM_ROW_BUILDERS[category]

    # Force an even year distribution for medium/large runs.
    # This prevents outcomes such as 32 council minutes from one year and none
    # from nearby years. For 300 council minutes over 1920-2026, each year gets
    # either 2 or 3 entries.
    global _FORCED_YEAR_QUEUE
    _FORCED_YEAR_QUEUE = v8_year_sequence(category, requested)

    rows = [v8_diversify_assignment(builder(rng, idx), rng, idx, requested) for idx in range(1, requested + 1)]
    _FORCED_YEAR_QUEUE = []

    return rows


def make_all_manifests(categories: list[str], per_category: int, counts_by_category: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    """Create manifests for the requested categories.

    For requested counts <= the curated pilot list length, this keeps using the
    hand-curated rows. For larger requests, it switches to the medium-run row
    builders, which create as many timeline-gated assignments as requested.
    """
    manifests: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        requested = counts_by_category.get(category, per_category)
        curated_rows = MANIFEST_BUILDERS[category]()
        if requested <= len(curated_rows):
            rng = random.Random(7777 + sum(ord(ch) for ch in category))
            manifests[category] = [v8_diversify_assignment(r, rng, i + 1, requested) for i, r in enumerate(curated_rows[:requested])]
        else:
            manifests[category] = make_medium_manifest(category, requested)
    return manifests


def summarize_metrics(metrics_rows: list[dict[str, Any]], total_wall_s: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_obj in metrics_rows:
        by_cat[row_obj["category"]].append(row_obj)

    category_summary: list[dict[str, Any]] = []
    projection: dict[str, Any] = {}

    for category, rows in sorted(by_cat.items()):
        generated = len(rows)
        ok = sum(1 for r in rows if r["status"] == "OK")
        warn = sum(1 for r in rows if r["status"] == "WARN")
        fail = sum(1 for r in rows if r["status"] == "FAIL")
        wall_sum = sum(float(r["wall_clock_s"]) for r in rows)
        eval_tokens = sum(int(r["eval_count"]) for r in rows)
        eval_duration = sum(float(r["eval_duration_s"]) for r in rows)
        words = sum(int(r["body_word_count"]) for r in rows)

        avg_wall = wall_sum / generated if generated else 0.0
        avg_tps = eval_tokens / eval_duration if eval_duration > 0 else 0.0
        avg_words = words / generated if generated else 0.0
        target = FINAL_ARCHIVE_TARGETS.get(category, 0)
        projected_seconds = avg_wall * target

        category_summary.append({
            "category": category,
            "generated": generated,
            "ok": ok,
            "warn": warn,
            "fail": fail,
            "wall_sum_s": round(wall_sum, 2),
            "avg_wall_s": round(avg_wall, 2),
            "eval_tokens": eval_tokens,
            "avg_eval_tokens_per_s": round(avg_tps, 2),
            "total_words": words,
            "avg_words": round(avg_words, 1),
            "final_archive_target": target,
            "projected_seconds_for_target": round(projected_seconds, 1),
            "projected_hours_for_target": round(projected_seconds / 3600, 2),
        })

        projection[category] = {
            "target": target,
            "avg_wall_s": avg_wall,
            "projected_seconds": projected_seconds,
            "projected_hours": projected_seconds / 3600,
        }

    total_projected_seconds = sum(x["projected_seconds"] for x in projection.values())
    overall = {
        "total_generated": len(metrics_rows),
        "total_wall_s": round(total_wall_s, 2),
        "total_wall_minutes": round(total_wall_s / 60, 2),
        "status_counts": dict(Counter(r["status"] for r in metrics_rows)),
        "final_archive_targets": FINAL_ARCHIVE_TARGETS,
        "projection_by_category": projection,
        "projected_full_archive_seconds": round(total_projected_seconds, 1),
        "projected_full_archive_hours": round(total_projected_seconds / 3600, 2),
    }

    return category_summary, overall


# ---------------------------------------------------------------------
# Args / main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a timed Mill Creek pilot archive with all major categories.")
    parser.add_argument("--per-category", type=int, default=10, help="Fallback count for categories not listed in --counts.")
    parser.add_argument(
        "--counts",
        default=None,
        help=(
            "Comma-separated category=count vector, e.g. "
            "'newspaper=50,school=10,religion=10,business=15,photo_caption=10,oral_history=3,council_minutes=8'. "
            "Categories not listed use --per-category."
        ),
    )
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES, help="Categories to include.")
    parser.add_argument("--out-root", type=Path, default=Path("generated/mill_creek_pilot_archive"))
    parser.add_argument("--run-label", default=None, help="Optional run folder name. Defaults to timestamp.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=0, help="Retries for exceptions or, with --retry-on-validation, validation warnings.")
    parser.add_argument("--retry-on-validation", action="store_true", help="Retry artifacts with validation warnings. Off by default for timing runs.")
    parser.add_argument("--hard-retries", type=int, default=2, help="Automatic retries for empty bodies and hard timeline/canon violations, even when --retry-on-validation is off.")
    parser.add_argument("--drop-failed-hard-violations", action="store_true", help="If hard violations persist after retries, omit that artifact from all_artifacts.jsonl instead of saving it as FAIL.")
    parser.add_argument("--dry-run", action="store_true", help="Create manifests and print plan, but do not call Ollama.")
    parser.add_argument("--sleep", type=float, default=0.0)

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = build_options(args)

    run_label = args.run_label or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.out_root / run_label

    manifest_dir = run_dir / "manifests"
    artifact_dir = run_dir / "artifacts"
    metrics_dir = run_dir / "metrics"
    gazetteer_dir = run_dir / "gazetteer"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    write_gazetteer(gazetteer_dir)

    counts_by_category = parse_category_counts(args.counts)
    manifests = make_all_manifests(args.categories, args.per_category, counts_by_category)

    all_assignments: list[dict[str, Any]] = []
    for category, rows in manifests.items():
        manifest_path = manifest_dir / f"{category}_manifest_{len(rows)}.jsonl"
        write_jsonl(manifest_path, rows)
        all_assignments.extend(rows)

    plan_path = run_dir / "run_plan.json"
    plan = {
        "run_label": run_label,
        "run_dir": str(run_dir),
        "model": args.model,
        "ollama_url": args.ollama_url,
        "options": options,
        "per_category_fallback": args.per_category,
        "counts_by_category": counts_by_category,
        "resolved_category_counts": {category: len(rows) for category, rows in manifests.items()},
        "categories": args.categories,
        "total_artifacts_planned": len(all_assignments),
        "retry_on_validation": args.retry_on_validation,
        "max_retries": args.max_retries,
        "hard_retries": args.hard_retries,
        "drop_failed_hard_violations": args.drop_failed_hard_violations,
    }
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Mill Creek pilot archive build")
    print("------------------------------")
    print(f"Run directory: {run_dir}")
    print(f"Model:         {args.model}")
    print(f"Categories:    {', '.join(args.categories)}")
    print(f"Fallback count: {args.per_category}")
    if counts_by_category:
        print(f"Count vector:  {counts_by_category}")
    print("Resolved counts:")
    for category, rows in manifests.items():
        print(f"  {category:15s} {len(rows)}")
    print(f"Total planned: {len(all_assignments)}")
    print(f"Retry validation warnings: {args.retry_on_validation}")
    print(f"Hard canon retries: {args.hard_retries}")
    print(f"Drop failed hard violations: {args.drop_failed_hard_violations}")
    print()

    if args.dry_run:
        print("Dry run manifest preview:")
        for category, rows in manifests.items():
            print(f"\n{category}: {len(rows)}")
            for r in rows[:min(3, len(rows))]:
                print(f"  {r['artifact_id']} | {r['year']} | {r['source_type']} | {r['title_seed']}")
        print(f"\nWrote manifests and gazetteer under:\n  {run_dir}")
        return

    metrics_path = metrics_dir / "artifact_metrics.csv"
    all_artifacts_path = artifact_dir / "all_artifacts.jsonl"

    metrics_rows: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()

    try:
        for i, assignment in enumerate(all_assignments, start=1):
            category = assignment["category"]
            category_out = artifact_dir / f"{category}_artifacts.jsonl"

            print(f"[{i}/{len(all_assignments)}] {assignment['artifact_id']} | {category} | {assignment['title_seed']}")

            record, metrics = generate_one(assignment, args, options)
            metrics_rows.append(metrics)

            append_csv(metrics_path, metrics)

            if record is not None:
                append_jsonl(category_out, record)
                append_jsonl(all_artifacts_path, record)

            status = metrics["status"]
            print(
                f"  {status} | words={metrics['body_word_count']} | "
                f"wall={metrics['wall_clock_s']}s | eval_tps={metrics['eval_tokens_per_s']} | "
                f"warnings={metrics['warning_count']}"
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\nInterrupted. Completed artifacts and metrics have been saved.")
    finally:
        total_wall = time.perf_counter() - total_t0
        category_summary, overall = summarize_metrics(metrics_rows, total_wall)

        write_csv(metrics_dir / "category_summary.csv", category_summary)
        (metrics_dir / "timing_summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\nTiming summary")
        print("--------------")
        print(f"Generated: {overall['total_generated']}")
        print(f"Wall time: {overall['total_wall_minutes']} minutes")
        print(f"Status counts: {overall['status_counts']}")
        print("\nCategory averages and projections:")
        for row_obj in category_summary:
            print(
                f"  {row_obj['category']:15s} "
                f"n={row_obj['generated']:3d} "
                f"avg_wall={row_obj['avg_wall_s']:6.2f}s "
                f"avg_tps={row_obj['avg_eval_tokens_per_s']:6.2f} "
                f"avg_words={row_obj['avg_words']:6.1f} "
                f"target={row_obj['final_archive_target']:5d} "
                f"projected={row_obj['projected_hours_for_target']:6.2f} h"
            )
        print(f"\nProjected full archive time from this timing run: {overall['projected_full_archive_hours']} hours")

        print("\nFiles:")
        print(f"  Manifests:        {manifest_dir}")
        print(f"  Artifacts:        {artifact_dir}")
        print(f"  Artifact metrics: {metrics_path}")
        print(f"  Category summary: {metrics_dir / 'category_summary.csv'}")
        print(f"  Timing summary:   {metrics_dir / 'timing_summary.json'}")
        print(f"  Gazetteer:        {gazetteer_dir}")


if __name__ == "__main__":
    main()
