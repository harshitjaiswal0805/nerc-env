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
        "time_limit": 25,
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
        ],
    },
}