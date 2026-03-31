from models import Mode

TASKS = {
    # ─── Task 1: Easy — Hospital Triage ───────────────────────────────────────
    # Small scenario, 3 patients, 2 doctors, 1 hospital
    # Agent just needs to assign doctors to critical patients first
    "task_1": {
        "mode": Mode.hospital,
        "time_limit": 15,
        "description": "Triage 3 patients with limited doctors. Prioritize critical cases.",
        "difficulty": "easy",
        "patients": [
            {"id": "p1", "severity": "critical"},
            {"id": "p2", "severity": "moderate"},
            {"id": "p3", "severity": "mild"},
        ],
        "hospitals": [
            {"id": "h1", "icu_capacity": 3, "current_patients": 0},
        ],
        "ambulances": [
            {"id": "a1", "status": "available"},
            {"id": "a2", "status": "available"},
        ],
        "doctors": [
            {"id": "d1", "specialty": "emergency", "status": "available"},
            {"id": "d2", "specialty": "general", "status": "available"},
        ],
        "rescue_teams": [],
    },

    # ─── Task 2: Medium — Multi-Hospital Coordination ─────────────────────────
    # 6 patients across 2 hospitals with limited ICU beds
    # Agent must distribute patients efficiently without overflowing ICU
    "task_2": {
        "mode": Mode.hospital,
        "time_limit": 20,
        "description": "Distribute 6 patients across 2 hospitals. Avoid ICU overflow.",
        "difficulty": "medium",
        "patients": [
            {"id": "p1", "severity": "critical"},
            {"id": "p2", "severity": "critical"},
            {"id": "p3", "severity": "moderate"},
            {"id": "p4", "severity": "moderate"},
            {"id": "p5", "severity": "mild"},
            {"id": "p6", "severity": "mild"},
        ],
        "hospitals": [
            {"id": "h1", "icu_capacity": 3, "current_patients": 0},
            {"id": "h2", "icu_capacity": 4, "current_patients": 0},
        ],
    
        "ambulances": [
            {"id": "a1", "status": "available"},
            {"id": "a2", "status": "available"},
            {"id": "a3", "status": "available"},
        ],
        "doctors": [
            {"id": "d1", "specialty": "emergency", "status": "available"},
            {"id": "d2", "specialty": "emergency", "status": "available"},
            {"id": "d3", "specialty": "general", "status": "available"},
        ],
        "rescue_teams": [],
    },

    # ─── Task 3: Hard — National Disaster Response ────────────────────────────
    # Mass casualty event: earthquake with 10 patients scattered across locations
    # Agent must coordinate rescue teams + ambulances + hospitals simultaneously
    # ICU capacity is deliberately tight to force smart allocation
    "task_3": {
        "mode": Mode.disaster,
        "time_limit": 30,
        "description": "Earthquake with 10 casualties. Rescue, triage and allocate under tight capacity.",
        "difficulty": "hard",
        "patients": [
            {"id": "p1",  "severity": "critical", "rescued": False},
            {"id": "p2",  "severity": "critical", "rescued": False},
            {"id": "p3",  "severity": "critical", "rescued": False},
            {"id": "p4",  "severity": "moderate", "rescued": False},
            {"id": "p5",  "severity": "moderate", "rescued": False},
            {"id": "p6",  "severity": "moderate", "rescued": False},
            {"id": "p7",  "severity": "mild",     "rescued": False},
            {"id": "p8",  "severity": "mild",     "rescued": False},
            {"id": "p9",  "severity": "mild",     "rescued": False},
            {"id": "p10", "severity": "mild",     "rescued": False},
        ],
        "hospitals": [
            {"id": "h1", "icu_capacity": 4, "current_patients": 0},
            {"id": "h2", "icu_capacity": 4, "current_patients": 0},
            {"id": "h3", "icu_capacity": 3, "current_patients": 0},
        ],
        "ambulances": [
            {"id": "a1", "status": "available"},
            {"id": "a2", "status": "available"},
            {"id": "a3", "status": "available"},
            {"id": "a4", "status": "available"},
        ],
        "doctors": [
            {"id": "d1", "specialty": "emergency",  "status": "available"},
            {"id": "d2", "specialty": "emergency",  "status": "available"},
            {"id": "d3", "specialty": "surgery",    "status": "available"},
            {"id": "d4", "specialty": "general",    "status": "available"},
        ],
        "rescue_teams": [
            {"id": "r1", "status": "available"},
            {"id": "r2", "status": "available"},
            {"id": "r3", "status": "available"},
            {"id": "r4", "status": "available"},
        ],
    },
    # ─── Task 4: Flood Disaster ───────────────────────────────────────────────
    # Rising floodwaters cut off multiple zones. Agent must prioritize rescue
    # of critical patients before zones become inaccessible. Time pressure is
    # higher than earthquake — rescue teams have limited window.
    "task_4": {
        "mode": Mode.disaster,
        "time_limit": 30,
        "description": "Flood disaster with rising water. Rescue critical patients before zones are cut off.",
        "difficulty": "hard",
        "patients": [
            {"id": "p1",  "severity": "critical", "rescued": False},
            {"id": "p2",  "severity": "critical", "rescued": False},
            {"id": "p3",  "severity": "critical", "rescued": False},
            {"id": "p4",  "severity": "critical", "rescued": False},
            {"id": "p5",  "severity": "moderate", "rescued": False},
            {"id": "p6",  "severity": "moderate", "rescued": False},
            {"id": "p7",  "severity": "moderate", "rescued": False},
            {"id": "p8",  "severity": "mild",     "rescued": False},
            {"id": "p9",  "severity": "mild",     "rescued": False},
            {"id": "p10", "severity": "mild",     "rescued": False},
            {"id": "p11", "severity": "mild",     "rescued": False},
            {"id": "p12", "severity": "mild",     "rescued": False},
        ],
        "hospitals": [
            {"id": "h1", "icu_capacity": 5, "current_patients": 0},
            {"id": "h2", "icu_capacity": 5, "current_patients": 0},
            {"id": "h3", "icu_capacity": 4, "current_patients": 0},
        ],
        "ambulances": [
            {"id": "a1", "status": "available"},
            {"id": "a2", "status": "available"},
            {"id": "a3", "status": "available"},
            {"id": "a4", "status": "available"},
            {"id": "a5", "status": "available"},
        ],
        "doctors": [
            {"id": "d1", "specialty": "emergency",  "status": "available"},
            {"id": "d2", "specialty": "emergency",  "status": "available"},
            {"id": "d3", "specialty": "emergency",  "status": "available"},
            {"id": "d4", "specialty": "surgery",    "status": "available"},
            {"id": "d5", "specialty": "general",    "status": "available"},
            {"id": "d6", "specialty": "general",    "status": "available"},
        ],
        "rescue_teams": [
            {"id": "r1", "status": "available"},
            {"id": "r2", "status": "available"},
            {"id": "r3", "status": "available"},
            {"id": "r4", "status": "available"},
            {"id": "r5", "status": "available"},
        ],
    },

    # ─── Task 5: Multi-City Coordination ──────────────────────────────────────
    # Simultaneous emergencies in 3 cities. Agent must coordinate resources
    # across cities with limited inter-city ambulance transfers.
    # Hardest task — tests true multi-objective optimization.
    "task_5": {
        "mode": Mode.disaster,
        "time_limit": 65,
        "description": "Simultaneous emergencies in 3 cities. Coordinate across limited resources.",
        "difficulty": "expert",
        "patients": [
            {"id": "p1",  "severity": "critical", "rescued": False},
            {"id": "p2",  "severity": "critical", "rescued": False},
            {"id": "p3",  "severity": "critical", "rescued": False},
            {"id": "p4",  "severity": "critical", "rescued": False},
            {"id": "p5",  "severity": "critical", "rescued": False},
            {"id": "p6",  "severity": "moderate", "rescued": False},
            {"id": "p7",  "severity": "moderate", "rescued": False},
            {"id": "p8",  "severity": "moderate", "rescued": False},
            {"id": "p9",  "severity": "moderate", "rescued": False},
            {"id": "p10", "severity": "moderate", "rescued": False},
            {"id": "p11", "severity": "mild",     "rescued": False},
            {"id": "p12", "severity": "mild",     "rescued": False},
            {"id": "p13", "severity": "mild",     "rescued": False},
            {"id": "p14", "severity": "mild",     "rescued": False},
            {"id": "p15", "severity": "mild",     "rescued": False},
        ],
        "hospitals": [
            {"id": "h1", "icu_capacity": 5, "current_patients": 0},
            {"id": "h2", "icu_capacity": 5, "current_patients": 0},
            {"id": "h3", "icu_capacity": 5, "current_patients": 0},
            {"id": "h4", "icu_capacity": 3, "current_patients": 0},
        ],
        "ambulances": [
            {"id": "a1", "status": "available"},
            {"id": "a2", "status": "available"},
            {"id": "a3", "status": "available"},
            {"id": "a4", "status": "available"},
            {"id": "a5", "status": "available"},
            {"id": "a6", "status": "available"},
        ],
        "doctors": [
            {"id": "d1", "specialty": "emergency",  "status": "available"},
            {"id": "d2", "specialty": "emergency",  "status": "available"},
            {"id": "d3", "specialty": "emergency",  "status": "available"},
            {"id": "d4", "specialty": "emergency",  "status": "available"},
            {"id": "d5", "specialty": "emergency",  "status": "available"},
            {"id": "d6", "specialty": "surgery",    "status": "available"},
            {"id": "d7", "specialty": "surgery",    "status": "available"},
            {"id": "d8", "specialty": "general",    "status": "available"},
            {"id": "d9", "specialty": "general",    "status": "available"},
            {"id": "d10","specialty": "general",    "status": "available"},
        ],
        "rescue_teams": [
            {"id": "r1", "status": "available"},
            {"id": "r2", "status": "available"},
            {"id": "r3", "status": "available"},
            {"id": "r4", "status": "available"},
            {"id": "r5", "status": "available"},
            {"id": "r6", "status": "available"},
        ],
    },
}