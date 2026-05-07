#!/usr/bin/env python3
"""
make_mill_creek_civic_data.py

Create the quantitative civic-data backbone for the fictional Mill Creek,
Cobberland digital humanities archive.

This script does NOT call an LLM. It generates deterministic-but-noisy CSV/JSONL
records that can be used with the textual archive for humanities ML labs.

Default output directory:
    data/civic_data/

Run:
    cd /home/darin/PycharmProjects/HumantiesBook/MillCreek
    python make_mill_creek_civic_data.py

Useful options:
    python make_mill_creek_civic_data.py --out data/civic_data_v1
    python make_mill_creek_civic_data.py --seed 1776
    python make_mill_creek_civic_data.py --start-year 1920 --end-year 2026

Files produced:
    mill_creek_population.csv
    mill_creek_school_records.csv
    mill_creek_city_budget.csv
    mill_creek_flood_records.csv
    mill_creek_business_counts.csv
    mill_creek_land_use.csv
    mill_creek_church_civic_records.csv
    mill_creek_public_offices.csv
    mill_creek_election_results.csv
    mill_creek_institution_timeline.csv
    mill_creek_events_timeline.csv
    mill_creek_place_gazetteer.csv
    mill_creek_data_dictionary.csv
    mill_creek_civic_data_summary.json

Design principles:
- Quantitative records are plausible, not historically real.
- Trends align with the Mill Creek world bible:
    river/flood risk, downtown change, school identity, public memory,
    church/civic service, agriculture, redevelopment, digitization.
- Major shocks are built in:
    1934 Depression relief
    1948 Memorial Field / postwar expansion
    1957 flood
    1968 bypass planning
    1988 Historical Society
    1993 flood
    1996 Maizey Olotón student award
    2001 Olotón Foods
    2009 Depot District redevelopment plan
    2011 South Flats buyouts
    2022 Riverfront Trail
    2025 digitization project
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def iround(value: float) -> int:
    return int(round(value))


def smooth_noise(rng: random.Random, scale: float = 1.0) -> float:
    # Small zero-centered noise.
    return rng.gauss(0.0, scale)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Preserve first row order, then include any later keys.
    fieldnames: list[str] = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def year_index(year: int, start_year: int, end_year: int) -> float:
    if end_year == start_year:
        return 0.0
    return (year - start_year) / (end_year - start_year)


# ---------------------------------------------------------------------
# World anchors
# ---------------------------------------------------------------------

MAJOR_EVENTS: list[dict[str, Any]] = [
    {
        "year": 1922,
        "event_id": "EVT_1922_READING_ROOM",
        "event": "Women’s Aid Society reading room fund drive",
        "themes": "reading culture; women’s civic labor; Main Street",
        "quantitative_effect": "small boost to civic participation and library/archive activity",
    },
    {
        "year": 1927,
        "event_id": "EVT_1927_HARVEST_HOME",
        "event": "Harvest Home program becomes established civic ritual",
        "themes": "festival; agriculture; belonging",
        "quantitative_effect": "festival volunteers and church/community events rise",
    },
    {
        "year": 1934,
        "event_id": "EVT_1934_DEPRESSION_RELIEF",
        "event": "Harvest Relief Supper and Depression-era service efforts",
        "themes": "economic hardship; church service; civic aid",
        "quantitative_effect": "city budget strain; church relief events rise; business growth falls",
    },
    {
        "year": 1948,
        "event_id": "EVT_1948_MEMORIAL_FIELD",
        "event": "Memorial Field dedicated before opening game",
        "themes": "postwar memory; school identity; civic ceremony",
        "quantitative_effect": "school sports participation and civic ceremony indicators rise",
    },
    {
        "year": 1957,
        "event_id": "EVT_1957_FLOOD",
        "event": "Major Prairie River flood",
        "themes": "flood; South Flats; public health; infrastructure",
        "quantitative_effect": "river crest spike; homes affected; flood-control spending rises",
    },
    {
        "year": 1968,
        "event_id": "EVT_1968_BYPASS",
        "event": "Highway bypass route debate",
        "themes": "downtown; planning; merchants; infrastructure",
        "quantitative_effect": "downtown business count begins longer decline; planning spending rises",
    },
    {
        "year": 1975,
        "event_id": "EVT_1975_RAIL_MISHAP",
        "event": "Rail mishap near Depot District",
        "themes": "rail; Depot District; infrastructure risk",
        "quantitative_effect": "small infrastructure and safety spending bump",
    },
    {
        "year": 1988,
        "event_id": "EVT_1988_HISTORICAL_SOCIETY",
        "event": "Mill Creek Historical Society opens first exhibit in restored depot room",
        "themes": "public memory; preservation; Depot District",
        "quantitative_effect": "archive budget and public-memory events begin",
    },
    {
        "year": 1993,
        "event_id": "EVT_1993_FLOOD",
        "event": "Flood volunteers build sandbag line near South Flats",
        "themes": "flood; volunteers; South Flats",
        "quantitative_effect": "river crest and sandbags spike; homes affected; volunteer counts rise",
    },
    {
        "year": 1996,
        "event_id": "EVT_1996_MAIZEY_AWARD",
        "event": "Maizey Olotón wins state entrepreneurship award",
        "themes": "school; entrepreneurship; agriculture",
        "quantitative_effect": "student business participation and local entrepreneurship narrative rise",
    },
    {
        "year": 2001,
        "event_id": "EVT_2001_OLOTON_FOODS",
        "event": "Olotón Foods opens local production space",
        "themes": "agriculture; business; employment; Maizey Olotón",
        "quantitative_effect": "farm-related business and employment indicators rise",
    },
    {
        "year": 2009,
        "event_id": "EVT_2009_DEPOT_REDEVELOPMENT",
        "event": "Depot District redevelopment plan released for public review",
        "themes": "redevelopment; preservation; public memory",
        "quantitative_effect": "redevelopment spending and public hearings rise",
    },
    {
        "year": 2011,
        "event_id": "EVT_2011_SOUTH_FLATS_BUYOUT",
        "event": "South Flats buyout notices after flooding",
        "themes": "floodplain; loss; home; public policy",
        "quantitative_effect": "South Flats households fall; buyout acres rise",
    },
    {
        "year": 2022,
        "event_id": "EVT_2022_RIVERFRONT_TRAIL",
        "event": "Riverfront Trail opens along former flood buyout land",
        "themes": "riverfront; memory; recreation; South Flats",
        "quantitative_effect": "parks spending and trail visits rise",
    },
    {
        "year": 2025,
        "event_id": "EVT_2025_DIGITIZATION",
        "event": "CobberTech students help Historical Society scan newspaper clippings",
        "themes": "digitization; public memory; digital humanities",
        "quantitative_effect": "archive/digitization spending and scanned items rise",
    },
]


INSTITUTIONS: list[dict[str, Any]] = [
    {"institution": "Mill Creek Chronicle", "type": "newspaper", "active_from": 1904, "active_to": 1987, "notes": "Original town newspaper."},
    {"institution": "Mill Creek Herald", "type": "newspaper", "active_from": 1988, "active_to": "", "notes": "Later successor newspaper."},
    {"institution": "Mill Creek City Council", "type": "government", "active_from": 1874, "active_to": "", "notes": "Primary municipal governing body."},
    {"institution": "Mill Creek Public Schools", "type": "school", "active_from": 1902, "active_to": "", "notes": "Town school system."},
    {"institution": "Mill Creek High School", "type": "school", "active_from": 1928, "active_to": "", "notes": "High school identity, sports, yearbooks."},
    {"institution": "St. Ansgar Lutheran Church", "type": "religious", "active_from": 1886, "active_to": "", "notes": "Major religious/civic institution."},
    {"institution": "Sacred Heart Catholic Mission", "type": "religious", "active_from": 1898, "active_to": "", "notes": "Major religious/civic institution."},
    {"institution": "Mill Creek Women’s Aid Society", "type": "civic", "active_from": 1908, "active_to": 1965, "notes": "Women’s civic and relief organization."},
    {"institution": "Mill Creek Commercial Club", "type": "business/civic", "active_from": 1910, "active_to": 1980, "notes": "Merchant and business association."},
    {"institution": "Prairie River Clinic", "type": "health", "active_from": 1954, "active_to": "", "notes": "Clinic important to flood/public health records."},
    {"institution": "Cobberland County Planning Commission", "type": "planning", "active_from": 1960, "active_to": "", "notes": "County planning and bypass/redevelopment documents."},
    {"institution": "Mill Creek Historical Society", "type": "archive/history", "active_from": 1988, "active_to": "", "notes": "Public memory institution in restored depot room."},
    {"institution": "CobberTech Extension Center", "type": "education/technology", "active_from": 1998, "active_to": "", "notes": "Later archive and digital humanities partner."},
    {"institution": "Cobberland Future Business Leaders Association", "type": "student/business", "active_from": 1980, "active_to": "", "notes": "Student entrepreneurship awards."},
    {"institution": "Olotón Foods", "type": "business/agriculture", "active_from": 2001, "active_to": "", "notes": "Maizey Olotón’s food/agriculture business."},
]


PLACES: list[dict[str, Any]] = [
    {"place_id": "PLC_001", "name": "Old Mill Bend", "district": "river", "x": 0.0, "y": 0.0, "active_from": 1874, "active_to": "", "themes": "origin story; river; mill; memory"},
    {"place_id": "PLC_002", "name": "Prairie River bridge", "district": "river", "x": 1.5, "y": -0.4, "active_from": 1893, "active_to": "", "themes": "flood; infrastructure; movement"},
    {"place_id": "PLC_003", "name": "Downtown/Main Street", "district": "downtown", "x": 0.5, "y": 0.7, "active_from": 1879, "active_to": "", "themes": "commerce; civic ritual; decline; nostalgia"},
    {"place_id": "PLC_004", "name": "Depot District", "district": "depot", "x": 1.2, "y": 1.0, "active_from": 1879, "active_to": "", "themes": "rail; labor; preservation; redevelopment"},
    {"place_id": "PLC_005", "name": "South Flats", "district": "river/residential", "x": 1.0, "y": -1.1, "active_from": 1900, "active_to": "", "themes": "home; flood risk; buyout; loss"},
    {"place_id": "PLC_006", "name": "North Orchard", "district": "residential", "x": 0.1, "y": 2.0, "active_from": 1948, "active_to": "", "themes": "postwar growth; schools; newcomers"},
    {"place_id": "PLC_007", "name": "West Rows farms", "district": "agricultural", "x": -2.0, "y": 0.3, "active_from": 1874, "active_to": "", "themes": "agriculture; seed; Maizey Olotón"},
    {"place_id": "PLC_008", "name": "grain elevator", "district": "agricultural/depot", "x": 1.0, "y": 1.2, "active_from": 1882, "active_to": "", "themes": "grain; rail; farm economy"},
    {"place_id": "PLC_009", "name": "Old Grange Hall", "district": "downtown", "x": 0.3, "y": 0.9, "active_from": 1908, "active_to": "", "themes": "relief suppers; meetings; public ritual"},
    {"place_id": "PLC_010", "name": "Main Street reading room", "district": "downtown", "x": 0.4, "y": 0.8, "active_from": 1915, "active_to": 1945, "themes": "reading culture; women’s volunteer labor"},
    {"place_id": "PLC_011", "name": "St. Ansgar Lutheran Church", "district": "downtown", "x": 0.0, "y": 1.1, "active_from": 1886, "active_to": "", "themes": "religion; service; civic identity"},
    {"place_id": "PLC_012", "name": "Sacred Heart Catholic Mission", "district": "downtown", "x": 0.8, "y": 0.6, "active_from": 1898, "active_to": "", "themes": "religion; service; community"},
    {"place_id": "PLC_013", "name": "Mill Creek Public Schools", "district": "school", "x": -0.2, "y": 1.6, "active_from": 1902, "active_to": "", "themes": "youth; civic identity; memory"},
    {"place_id": "PLC_014", "name": "Mill Creek High School", "district": "school", "x": -0.4, "y": 1.8, "active_from": 1928, "active_to": "", "themes": "youth; sports; Maizey Olotón"},
    {"place_id": "PLC_015", "name": "Memorial Field", "district": "school", "x": -0.7, "y": 1.5, "active_from": 1948, "active_to": "", "themes": "war memory; sports; public ceremony"},
    {"place_id": "PLC_016", "name": "City Hall", "district": "civic", "x": 0.6, "y": 0.9, "active_from": 1910, "active_to": "", "themes": "government; minutes; public notices"},
    {"place_id": "PLC_017", "name": "Prairie River Clinic", "district": "civic/health", "x": 0.9, "y": 0.4, "active_from": 1954, "active_to": "", "themes": "health; flood; public notices"},
    {"place_id": "PLC_018", "name": "Mill Creek Historical Society", "district": "depot", "x": 1.1, "y": 1.05, "active_from": 1988, "active_to": "", "themes": "public memory; archive; preservation"},
    {"place_id": "PLC_019", "name": "restored depot room", "district": "depot", "x": 1.15, "y": 1.08, "active_from": 1988, "active_to": "", "themes": "exhibit; preservation; rail memory"},
    {"place_id": "PLC_020", "name": "Olotón Foods", "district": "agricultural/business", "x": -1.6, "y": 0.0, "active_from": 2001, "active_to": "", "themes": "Maizey Olotón; entrepreneurship; agriculture"},
    {"place_id": "PLC_021", "name": "CobberTech Extension Center", "district": "education/civic", "x": 0.2, "y": 2.3, "active_from": 1998, "active_to": "", "themes": "digitization; students; digital humanities"},
    {"place_id": "PLC_022", "name": "Riverfront Trail", "district": "river", "x": 0.8, "y": -0.8, "active_from": 2022, "active_to": "", "themes": "buyout land; public memory; recreation"},
]


OFFICES: list[dict[str, Any]] = [
    # Mayors
    {"person": "Mayor Nels Hovland", "office": "mayor", "start_year": 1918, "end_year": 1939, "notes": "early civic improvements; Depression relief"},
    {"person": "Mayor Ingrid Lunde", "office": "mayor", "start_year": 1946, "end_year": 1962, "notes": "postwar expansion; Memorial Field; 1957 flood"},
    {"person": "Mayor Edwin Rask", "office": "mayor", "start_year": 1963, "end_year": 1989, "notes": "bypass planning; downtown transition; Historical Society opening"},
    {"person": "Mayor Naomi Reyes", "office": "mayor", "start_year": 1990, "end_year": 2026, "notes": "redevelopment; South Flats buyout; Riverfront Trail"},
    # Clerks
    {"person": "Clerk Elsie Bratten", "office": "city clerk", "start_year": 1915, "end_year": 1940, "notes": "early council records"},
    {"person": "Clerk Ruth Ellingson", "office": "city clerk", "start_year": 1941, "end_year": 1978, "notes": "school/civic records; postwar minutes"},
    {"person": "Clerk Janine Roberts", "office": "city clerk", "start_year": 1979, "end_year": 2010, "notes": "redevelopment records; early digitization"},
    {"person": "Clerk Asha Patel", "office": "city clerk", "start_year": 2011, "end_year": 2026, "notes": "South Flats buyout; Riverfront Trail; archive pilot"},
    # Other recurring public roles
    {"person": "Professor Carrel Englekorn", "office": "historian", "start_year": 1975, "end_year": 2026, "notes": "local historian; public memory; archive interpretation"},
    {"person": "Maizey Olotón", "office": "business founder", "start_year": 2001, "end_year": 2026, "notes": "Olotón Foods; local agriculture and entrepreneurship"},
    {"person": "Nora Reyes", "office": "student archive assistant", "start_year": 2022, "end_year": 2026, "notes": "CobberTech digitization and metadata work"},
    {"person": "Dr. Helen Markham", "office": "clinic physician", "start_year": 1954, "end_year": 1995, "notes": "public health and flood notices"},
    {"person": "Coach Harold Bratten", "office": "coach", "start_year": 1940, "end_year": 1995, "notes": "Memorial Field and school athletics"},
]


# ---------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------

def generate_population(start_year: int, end_year: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for year in range(start_year, end_year + 1):
        # Long arc:
        # 1920s modest town; Depression stagnation; postwar growth; bypass/downtown transition;
        # slight decline/stagnation; redevelopment recovery; contemporary modest growth.
        if year <= 1930:
            base_pop = lerp(1750, 2050, (year - 1920) / 10)
        elif year <= 1940:
            base_pop = lerp(2050, 1950, (year - 1930) / 10)
        elif year <= 1960:
            base_pop = lerp(1950, 2850, (year - 1940) / 20)
        elif year <= 1980:
            base_pop = lerp(2850, 3050, (year - 1960) / 20)
        elif year <= 2000:
            base_pop = lerp(3050, 2920, (year - 1980) / 20)
        elif year <= 2015:
            base_pop = lerp(2920, 3150, (year - 2000) / 15)
        else:
            base_pop = lerp(3150, 3420, (year - 2015) / max(1, end_year - 2015))

        # Shocks.
        if year in {1934, 1935}:
            base_pop -= 35
        if year in {1957, 1958}:
            base_pop -= 20
        if year in {1993, 1994}:
            base_pop -= 25
        if 2011 <= year <= 2014:
            base_pop -= 45  # South Flats buyout and displacement.
        if year >= 2022:
            base_pop += 20  # mild riverfront-redevelopment bump.

        population = max(1200, iround(base_pop + smooth_noise(rng, 18)))
        households = iround(population / clamp(3.15 - 0.006 * (year - 1920), 2.25, 3.15))
        median_age = round(clamp(31 + 0.075 * (year - 1920) + smooth_noise(rng, 0.45), 29, 43), 1)

        children_under_18 = iround(population * clamp(0.31 - 0.0012 * (year - 1920) + smooth_noise(rng, 0.006), 0.20, 0.34))
        adults_over_65 = iround(population * clamp(0.08 + 0.0010 * (year - 1920) + smooth_noise(rng, 0.004), 0.07, 0.19))

        farm_households = iround(households * clamp(0.34 - 0.0023 * (year - 1920), 0.07, 0.34))
        downtown_households = iround(households * clamp(0.22 - 0.0007 * max(year - 1968, 0), 0.12, 0.24))
        # South Flats remains a meaningful neighborhood for much of the town's history,
        # but the 2011 flood/buyout period causes a visible decline in occupied homes.
        # We model this as a declining share after 2011 plus an absolute buyout shock,
        # so the raw household count visibly drops even while the whole town grows.
        south_flats_share = clamp(0.15 - 0.0065 * max(year - 2011, 0), 0.018, 0.16)
        south_flats_households = iround(households * south_flats_share)

        if year == 2011:
            south_flats_households -= 18
        elif year == 2012:
            south_flats_households -= 32
        elif year == 2013:
            south_flats_households -= 42
        elif year == 2014:
            south_flats_households -= 48
        elif year >= 2015:
            south_flats_households -= 52

        south_flats_households = max(18, south_flats_households)
        north_orchard_households = iround(households * (0.02 if year < 1948 else clamp(0.05 + 0.0015 * (year - 1948), 0.05, 0.16)))

        rows.append({
            "year": year,
            "population": population,
            "households": households,
            "median_age": median_age,
            "children_under_18": children_under_18,
            "adults_over_65": adults_over_65,
            "farm_households": farm_households,
            "downtown_households": downtown_households,
            "south_flats_households": south_flats_households,
            "north_orchard_households": north_orchard_households,
        })
    return rows


def generate_school_records(pop_rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for p in pop_rows:
        year = p["year"]
        population = p["population"]
        children = p["children_under_18"]

        total_enrollment = iround(children * clamp(0.68 + smooth_noise(rng, 0.015), 0.60, 0.75))
        high_school_enrollment = 0 if year < 1928 else iround(total_enrollment * clamp(0.24 + 0.0007 * (year - 1928), 0.22, 0.35))
        graduating_class_size = 0 if year < 1928 else iround(high_school_enrollment / 4 * clamp(0.82 + smooth_noise(rng, 0.035), 0.70, 0.95))
        graduation_rate = 0 if year < 1928 else round(clamp(0.62 + 0.0029 * (year - 1928) + smooth_noise(rng, 0.018), 0.55, 0.94), 3)
        attendance_rate = round(clamp(0.89 + 0.0004 * (year - 1920) + smooth_noise(rng, 0.007), 0.84, 0.97), 3)

        student_teacher_ratio = round(clamp(22 - 0.065 * (year - 1920) + smooth_noise(rng, 0.6), 12, 24), 1)
        sports_participation = 0 if year < 1928 else iround(high_school_enrollment * clamp(0.18 + (0.08 if year >= 1948 else 0.0) + smooth_noise(rng, 0.025), 0.15, 0.40))
        music_participation = iround(total_enrollment * clamp(0.11 + 0.0007 * (year - 1920) + smooth_noise(rng, 0.012), 0.08, 0.23))
        club_participation = iround(total_enrollment * clamp(0.08 + 0.0014 * (year - 1920) + smooth_noise(rng, 0.018), 0.05, 0.34))
        business_club_participation = 0 if year < 1980 else iround(high_school_enrollment * clamp(0.04 + (0.04 if year >= 1996 else 0) + smooth_noise(rng, 0.012), 0.02, 0.15))

        if year in {1996, 1997}:
            business_club_participation += 8

        rows.append({
            "year": year,
            "total_enrollment": total_enrollment,
            "high_school_enrollment": high_school_enrollment,
            "graduating_class_size": graduating_class_size,
            "graduation_rate": graduation_rate,
            "attendance_rate": attendance_rate,
            "student_teacher_ratio": student_teacher_ratio,
            "sports_participation": sports_participation,
            "music_participation": music_participation,
            "club_participation": club_participation,
            "business_club_participation": business_club_participation,
        })
    return rows


def generate_flood_records(start_year: int, end_year: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for year in range(start_year, end_year + 1):
        seasonal = 12.0 + 1.2 * math.sin((year - 1920) / 4.7)
        crest = seasonal + smooth_noise(rng, 0.9)
        days_above = 0
        homes_affected = 0
        sandbags = 0
        bridge_closures = 0
        buyout_acres = 0.0

        if year == 1957:
            crest = 20.8 + smooth_noise(rng, 0.2)
            days_above = 12
            homes_affected = 86
            sandbags = 18000
            bridge_closures = 3
        elif year == 1993:
            crest = 21.6 + smooth_noise(rng, 0.2)
            days_above = 16
            homes_affected = 112
            sandbags = 26000
            bridge_closures = 4
        elif year == 2011:
            crest = 20.2 + smooth_noise(rng, 0.25)
            days_above = 10
            homes_affected = 74
            sandbags = 21000
            bridge_closures = 2
            buyout_acres = 14.5
        elif crest >= 17.0:
            days_above = iround((crest - 16.0) * 2 + smooth_noise(rng, 1.0))
            homes_affected = max(0, iround((crest - 16.5) * 8 + smooth_noise(rng, 4)))
            sandbags = max(0, iround((crest - 16.0) * 900 + smooth_noise(rng, 250)))
            bridge_closures = 1 if crest > 18.0 else 0

        if 2012 <= year <= 2020:
            buyout_acres = round(max(0.0, 14.5 + 0.9 * (year - 2011) + smooth_noise(rng, 0.25)), 1)
        elif year >= 2021:
            buyout_acres = round(max(0.0, 22.0 + smooth_noise(rng, 0.4)), 1)

        rows.append({
            "year": year,
            "spring_crest_feet": round(crest, 2),
            "flood_stage_exceeded": bool(crest >= 17.0),
            "days_above_flood_stage": max(0, iround(days_above)),
            "sandbags_filled": max(0, iround(sandbags)),
            "homes_affected": max(0, iround(homes_affected)),
            "bridge_closures": max(0, iround(bridge_closures)),
            "south_flats_buyout_acres_cumulative": buyout_acres,
        })
    return rows


def generate_city_budget(pop_rows: list[dict[str, Any]], flood_rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    flood_by_year = {r["year"]: r for r in flood_rows}
    rows = []
    for p in pop_rows:
        year = p["year"]
        population = p["population"]
        t = year - 1920

        total_budget = population * (72 + 6.8 * t) + 25000 * math.sin(t / 7.5) + smooth_noise(rng, 7000)
        if year in {1934, 1935}:
            total_budget *= 0.88
        if year in {1957, 1993, 2011}:
            total_budget *= 1.18
        if year >= 2009:
            total_budget *= 1.06
        if year >= 2022:
            total_budget *= 1.04

        flood = flood_by_year[year]
        flood_control_budget = max(2000, total_budget * (0.035 + (0.11 if flood["flood_stage_exceeded"] else 0.0)))
        if year >= 2011:
            flood_control_budget += 60000
        street_budget = total_budget * clamp(0.22 + (0.04 if year in {1957, 1968, 1993, 2009} else 0) + smooth_noise(rng, 0.01), 0.16, 0.34)
        school_appropriation = total_budget * clamp(0.18 + 0.0005 * min(t, 60) + smooth_noise(rng, 0.01), 0.16, 0.28)
        parks_budget = total_budget * clamp(0.035 + (0.045 if year >= 2022 else 0.0) + smooth_noise(rng, 0.006), 0.02, 0.12)
        public_health_budget = total_budget * clamp(0.03 + (0.035 if flood["flood_stage_exceeded"] else 0.0) + (0.02 if year >= 1954 else 0.0), 0.015, 0.12)

        library_archive_budget = total_budget * 0.012
        if 1922 <= year <= 1945:
            library_archive_budget += 1200
        if year >= 1988:
            library_archive_budget += total_budget * 0.018
        if year >= 2025:
            library_archive_budget += total_budget * 0.012

        redevelopment_budget = 0.0
        if 1968 <= year <= 1975:
            redevelopment_budget += total_budget * 0.035
        if 2009 <= year <= 2015:
            redevelopment_budget += total_budget * 0.075
        if year >= 2022:
            redevelopment_budget += total_budget * 0.03

        rows.append({
            "year": year,
            "total_city_budget_dollars": iround(total_budget),
            "street_budget_dollars": iround(street_budget),
            "school_appropriation_dollars": iround(school_appropriation),
            "flood_control_budget_dollars": iround(flood_control_budget),
            "parks_budget_dollars": iround(parks_budget),
            "library_archive_budget_dollars": iround(library_archive_budget),
            "public_health_budget_dollars": iround(public_health_budget),
            "redevelopment_budget_dollars": iround(redevelopment_budget),
            "debt_service_dollars": iround(total_budget * clamp(0.045 + smooth_noise(rng, 0.008), 0.02, 0.09)),
        })
    return rows


def generate_business_counts(start_year: int, end_year: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for year in range(start_year, end_year + 1):
        if year <= 1968:
            downtown = lerp(42, 56, (year - 1920) / (1968 - 1920))
        elif year <= 1990:
            downtown = lerp(56, 38, (year - 1968) / (1990 - 1968))
        elif year <= 2010:
            downtown = lerp(38, 34, (year - 1990) / 20)
        else:
            downtown = lerp(34, 43, (year - 2010) / max(1, end_year - 2010))

        if year in {1934, 1935}:
            downtown -= 4
        if year in {1957, 1993, 2011}:
            downtown -= 2
        if year >= 2022:
            downtown += 3

        depot_business = 14 if year < 1968 else lerp(14, 5, min(1, (year - 1968) / 25))
        if year >= 2009:
            depot_business += lerp(0, 8, min(1, (year - 2009) / 15))

        farm_related = lerp(26, 16, (year - 1920) / max(1, end_year - 1920))
        if year >= 2001:
            farm_related += 5
        if year >= 2022:
            farm_related += 1

        vacant = clamp(2 + 0.10 * max(year - 1968, 0) - 0.18 * max(year - 2010, 0), 1, 14)
        if year in {1934, 1935, 1993, 2011}:
            vacant += 3
        if year >= 2022:
            vacant -= 2

        new_licenses = clamp(3 + 0.02 * (year - 1920) + (2 if year >= 2001 else 0) + (2 if year >= 2022 else 0) + smooth_noise(rng, 1.2), 0, 16)
        building_permits = clamp(8 + 0.07 * (year - 1920) + (6 if year >= 2009 else 0) + smooth_noise(rng, 2.0), 2, 30)

        rows.append({
            "year": year,
            "downtown_business_count": max(0, iround(downtown + smooth_noise(rng, 1.3))),
            "depot_district_business_count": max(0, iround(depot_business + smooth_noise(rng, 0.9))),
            "farm_related_business_count": max(0, iround(farm_related + smooth_noise(rng, 1.0))),
            "vacant_storefronts": max(0, iround(vacant + smooth_noise(rng, 1.0))),
            "new_business_licenses": max(0, iround(new_licenses)),
            "building_permits": max(0, iround(building_permits)),
            "oloton_foods_estimated_employees": 0 if year < 2001 else max(4, iround(6 + 1.9 * (year - 2001) + smooth_noise(rng, 3.0))),
        })
    return rows


def generate_land_use(pop_rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for p in pop_rows:
        year = p["year"]

        residential = 410 + 2.2 * (year - 1920)
        commercial = 55 + 0.25 * (year - 1920)
        industrial = 22 + 0.30 * max(year - 1940, 0)
        agricultural = 2100 - 4.0 * (year - 1920)
        civic = 35 + 0.33 * (year - 1920)
        parks = 22 + 0.08 * (year - 1920)

        if year >= 1948:
            residential += 35
            civic += 12
        if year >= 1968:
            commercial += 18
            agricultural -= 60
        if year >= 2009:
            commercial += 12
            industrial -= 4
            civic += 9
        if year >= 2011:
            residential -= 35
            parks += 28
        if year >= 2022:
            parks += 34
            civic += 8

        rows.append({
            "year": year,
            "residential_acres": round(max(0, residential + smooth_noise(rng, 3.0)), 1),
            "commercial_acres": round(max(0, commercial + smooth_noise(rng, 1.8)), 1),
            "industrial_acres": round(max(0, industrial + smooth_noise(rng, 1.0)), 1),
            "agricultural_acres": round(max(0, agricultural + smooth_noise(rng, 8.0)), 1),
            "civic_institutional_acres": round(max(0, civic + smooth_noise(rng, 1.0)), 1),
            "parks_trails_open_space_acres": round(max(0, parks + smooth_noise(rng, 1.2)), 1),
            "south_flats_buyout_acres": round(max(0, (0 if year < 2011 else 14.5 + 0.75 * (year - 2011))), 1),
        })
    return rows


def generate_church_civic_records(start_year: int, end_year: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for year in range(start_year, end_year + 1):
        pop_factor = 1 + 0.002 * (year - 1920)
        st_ansgar = iround(clamp(260 * pop_factor - 0.35 * max(year - 1975, 0) + smooth_noise(rng, 8), 160, 420))
        sacred_heart = iround(clamp(115 * pop_factor + 0.20 * max(year - 1960, 0) + smooth_noise(rng, 5), 80, 260))

        service_events = iround(clamp(6 + 0.04 * (year - 1920) + smooth_noise(rng, 1.0), 3, 18))
        if year in {1934, 1957, 1993, 2011}:
            service_events += 6
        if year >= 2022:
            service_events += 2

        relief_collections = iround(clamp(2 + (4 if year in {1934, 1957, 1993, 2011} else 0) + smooth_noise(rng, 0.8), 0, 12))
        choir_participants = iround(clamp(28 + 0.05 * (year - 1920) + smooth_noise(rng, 3), 15, 60))
        festival_volunteers = iround(clamp(24 + 0.22 * (year - 1927) + (10 if year >= 2022 else 0) + smooth_noise(rng, 5), 10, 90))

        rows.append({
            "year": year,
            "st_ansgar_membership_estimate": st_ansgar,
            "sacred_heart_membership_estimate": sacred_heart,
            "combined_service_events": service_events,
            "choir_participants_estimate": choir_participants,
            "relief_collections": relief_collections,
            "festival_volunteers_estimate": festival_volunteers,
        })
    return rows


def generate_election_results(start_year: int, end_year: int, rng: random.Random) -> list[dict[str, Any]]:
    # Municipal election every 4 years. These are intentionally simple.
    rows = []
    for year in range(start_year, end_year + 1):
        if year % 4 != 2:
            continue

        mayor = "Mayor Nels Hovland"
        challenger = "Councilman Martin Kvale"
        if year >= 1946:
            mayor = "Mayor Ingrid Lunde"
            challenger = "Councilman Peter Harlan"
        if year >= 1966:
            mayor = "Mayor Edwin Rask"
            challenger = "Councilwoman Lena Voss"
        if year >= 1990:
            mayor = "Mayor Naomi Reyes"
            challenger = "Councilman David Harlan"

        turnout = round(clamp(0.48 + 0.08 * math.sin(year / 6) + smooth_noise(rng, 0.04), 0.32, 0.76), 3)
        incumbent_share = round(clamp(0.56 + smooth_noise(rng, 0.06), 0.44, 0.72), 3)

        rows.append({
            "election_year": year,
            "office": "mayor",
            "incumbent_candidate": mayor,
            "challenger_candidate": challenger,
            "registered_voters_estimate": iround(1150 + 12 * (year - start_year) + smooth_noise(rng, 50)),
            "turnout_rate": turnout,
            "incumbent_vote_share": incumbent_share,
            "challenger_vote_share": round(1.0 - incumbent_share, 3),
            "notes": "Simplified fictional municipal election record for humanities ML exercises.",
        })
    return rows


def generate_data_dictionary() -> list[dict[str, Any]]:
    return [
        {"file": "mill_creek_population.csv", "field": "population", "description": "Estimated town population by year."},
        {"file": "mill_creek_population.csv", "field": "south_flats_households", "description": "Estimated number of households in South Flats; drops after 2011 buyouts."},
        {"file": "mill_creek_school_records.csv", "field": "graduation_rate", "description": "Estimated high-school graduation rate, 0 to 1."},
        {"file": "mill_creek_school_records.csv", "field": "business_club_participation", "description": "Estimated student participation in business/entrepreneurship clubs."},
        {"file": "mill_creek_city_budget.csv", "field": "flood_control_budget_dollars", "description": "Estimated municipal flood-control spending."},
        {"file": "mill_creek_flood_records.csv", "field": "spring_crest_feet", "description": "Estimated spring river crest in feet."},
        {"file": "mill_creek_flood_records.csv", "field": "south_flats_buyout_acres_cumulative", "description": "Cumulative buyout acreage after South Flats floodplain policy."},
        {"file": "mill_creek_business_counts.csv", "field": "vacant_storefronts", "description": "Estimated number of vacant downtown storefronts."},
        {"file": "mill_creek_church_civic_records.csv", "field": "combined_service_events", "description": "Estimated number of church/civic service events in a year."},
        {"file": "mill_creek_public_offices.csv", "field": "start_year/end_year", "description": "Active office window for a recurring public figure."},
        {"file": "mill_creek_institution_timeline.csv", "field": "active_from/active_to", "description": "Active date range for important Mill Creek institutions."},
        {"file": "mill_creek_place_gazetteer.csv", "field": "x/y", "description": "Simple local coordinate system for mapping labs; not real GPS coordinates."},
    ]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Mill Creek quantitative civic-data CSV files.")
    parser.add_argument("--start-year", type=int, default=1920)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=1776)
    parser.add_argument("--out", type=Path, default=Path("data/civic_data"))
    parser.add_argument("--also-jsonl", action="store_true", help="Also write JSONL copies of the main yearly tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    pop = generate_population(args.start_year, args.end_year, rng)
    school = generate_school_records(pop, rng)
    floods = generate_flood_records(args.start_year, args.end_year, rng)
    budget = generate_city_budget(pop, floods, rng)
    business = generate_business_counts(args.start_year, args.end_year, rng)
    land = generate_land_use(pop, rng)
    church = generate_church_civic_records(args.start_year, args.end_year, rng)
    elections = generate_election_results(args.start_year, args.end_year, rng)
    dictionary = generate_data_dictionary()

    files = {
        "mill_creek_population.csv": pop,
        "mill_creek_school_records.csv": school,
        "mill_creek_city_budget.csv": budget,
        "mill_creek_flood_records.csv": floods,
        "mill_creek_business_counts.csv": business,
        "mill_creek_land_use.csv": land,
        "mill_creek_church_civic_records.csv": church,
        "mill_creek_public_offices.csv": OFFICES,
        "mill_creek_election_results.csv": elections,
        "mill_creek_institution_timeline.csv": INSTITUTIONS,
        "mill_creek_events_timeline.csv": MAJOR_EVENTS,
        "mill_creek_place_gazetteer.csv": PLACES,
        "mill_creek_data_dictionary.csv": dictionary,
    }

    for filename, rows in files.items():
        write_csv(out / filename, rows)

    if args.also_jsonl:
        for filename, rows in files.items():
            write_jsonl(out / filename.replace(".csv", ".jsonl"), rows)

    summary = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "seed": args.seed,
        "output_directory": str(out),
        "files_written": list(files.keys()),
        "row_counts": {filename: len(rows) for filename, rows in files.items()},
        "design_note": (
            "These are fictional quantitative records for the Mill Creek, Cobberland "
            "digital humanities archive. They are designed to be internally plausible "
            "and useful for ML/distant-reading exercises, not historically real."
        ),
        "major_event_years": [event["year"] for event in MAJOR_EVENTS],
    }

    (out / "mill_creek_civic_data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Mill Creek civic-data generator")
    print("-------------------------------")
    print(f"Output directory: {out}")
    print(f"Year range:       {args.start_year}-{args.end_year}")
    print(f"Seed:             {args.seed}")
    print("\nFiles written:")
    for filename in files:
        print(f"  {filename:45s} rows={len(files[filename])}")
    print("  mill_creek_civic_data_summary.json")
    if args.also_jsonl:
        print("\nJSONL copies were also written.")


if __name__ == "__main__":
    main()
