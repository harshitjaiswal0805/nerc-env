---
title: NERC-Env
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# NERC-Env: National Emergency Response & Care Environment

An OpenEnv-compliant RL environment for training AI agents on real-world emergency response and hospital triage tasks.

## Environment Description

NERC-Env simulates two real-world emergency scenarios:
- **Hospital Mode**: Emergency room triage, doctor allocation, ICU assignment
- **Disaster Mode**: Mass casualty events, rescue coordination, multi-hospital distribution

## Action Space

| Action | Parameters |
|--------|-----------|
| `assign_doctor` | patient_id, doctor_id |
| `dispatch_ambulance` | patient_id, ambulance_id, hospital_id |
| `transfer_patient` | patient_id, hospital_id |
| `dispatch_rescue` | rescue_team_id, location |
| `prioritize_patient` | patient_id |
| `wait` | — |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| mode | string | hospital or disaster |
| patients | list | severity, status, assignments |
| hospitals | list | capacity, current patients |
| ambulances | list | status, assignments |
| doctors | list | specialty, status |
| rescue_teams | list | status, location |
| time_left | int | steps remaining |

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| task_1 | Easy | Triage 3 patients with limited doctors |
| task_2 | Medium | Distribute 6 patients across 2 hospitals |
| task_3 | Hard | Earthquake with 10 casualties |

## Baseline Scores

| Task | Score |
|------|-------|
| task_1 | 0.9933 |
| task_2 | 0.9900 |
| task_3 | 0.9860 |
| task_4 | 0.9913 |
| task_5 | 1.0000 |
| **Average** | **0.9921** |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks |
| `/grader` | POST | Get episode score |
| `/baseline` | POST | Run baseline agent |

## Testing
```bash
pip install pytest
pytest test_env.py -v
```

All 27 tests pass covering reset(), step(), state(), grader, reward signals, and task validation.

## Setup
```bash
docker build -t nerc-env .
docker run -p 7860:7860 nerc-env
```

## Usage
```python
import requests

BASE = "https://harshitjaiswal08-nerc-env.hf.space"

# Reset environment
obs = requests.post(f"{BASE}/reset", json={"task_id": "task_1"}).json()

# Take a step
action = {"action_type": "assign_doctor", "patient_id": "p1", "doctor_id": "d1"}
result = requests.post(f"{BASE}/step", json={"action": action}).json()

# Get score
score = requests.post(f"{BASE}/grader").json()
```
