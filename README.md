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

A complete OpenEnv-compliant reinforcement learning environment for training AI agents on real-world emergency response and hospital triage tasks.

---

## Overview

NERC-Env simulates two real-world emergency scenarios that humans actually face:

- **Hospital Mode** — Emergency room triage, doctor allocation, ICU assignment, inter-hospital transfers
- **Disaster Mode** — Mass casualty events (earthquakes, floods), rescue coordination, multi-hospital distribution

Agents interact through a standard `step()` / `reset()` / `state()` HTTP API and are evaluated on how many lives they save, how efficiently they allocate resources, and how quickly they respond to critical patients.

---

## Why This Environment Matters

Emergency response coordination is one of the hardest real-world optimization problems:
- Resources are scarce (limited doctors, ambulances, ICU beds)
- Decisions are time-critical (critical patients die without care)
- Actions have cascading effects (sending an ambulance to the wrong hospital causes overflow)
- Multiple objectives must be balanced simultaneously (save the most lives while using resources efficiently)

This makes it an ideal benchmark for evaluating AI agent decision-making under pressure.

---

## Tasks

| Task | Mode | Difficulty | Patients | Time Limit | Description |
|------|------|-----------|---------|-----------|-------------|
| task_1 | Hospital | Easy | 3 | 15 steps | Triage 3 patients with limited doctors. Prioritize critical cases. |
| task_2 | Hospital | Medium | 6 | 20 steps | Distribute 6 patients across 2 hospitals. Avoid ICU overflow. |
| task_3 | Disaster | Hard | 10 | 30 steps | Earthquake with 10 casualties. Rescue, triage and allocate under tight capacity. |
| task_4 | Disaster | Hard | 12 | 30 steps | Flood disaster with rising water. Rescue critical patients before zones are cut off. |
| task_5 | Disaster | Expert | 15 | 55 steps | Simultaneous emergencies in 3 cities. Coordinate across limited resources. |

---

## Baseline Scores

Scores from the built-in rule-based baseline agent:

| Task | Score | Alive | Critical Saved |
|------|-------|-------|----------------|
| task_1 | 0.9933 | 3/3 | 1/1 |
| task_2 | 0.9900 | 6/6 | 2/2 |
| task_3 | 0.9910 | 10/10 | 3/3 |
| task_4 | 0.9913 | 12/12 | 4/4 |
| task_5 | 1.0000 | 15/15 | 5/5 |
| **Average** | **0.9921** | | |

---

## Action Space

Agents send one action per step as a JSON object:

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `assign_doctor` | patient_id, doctor_id | Assign a doctor to a patient |
| `dispatch_ambulance` | patient_id, ambulance_id, hospital_id | Transport patient to hospital |
| `transfer_patient` | patient_id, hospital_id | Move patient between hospitals |
| `dispatch_rescue` | rescue_team_id, location | Deploy rescue team to disaster zone |
| `prioritize_patient` | patient_id | Reset patient neglect counter |
| `wait` | — | Do nothing this step |

Example action:
```json
{
    "action_type": "assign_doctor",
    "patient_id": "p1",
    "doctor_id": "d1"
}
```

---

## Observation Space

Each step returns a full observation of the environment state:

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `hospital` or `disaster` |
| `patients` | list | id, severity, assigned_doctor, assigned_hospital, rescued, alive, steps_without_care |
| `hospitals` | list | id, icu_capacity, current_patients |
| `ambulances` | list | id, status, assigned_patient |
| `doctors` | list | id, specialty, status, assigned_patient |
| `rescue_teams` | list | id, status, assigned_location |
| `time_left` | int | Steps remaining in episode |
| `last_reward` | float | Reward from last action |
| `done` | bool | Whether episode has ended |

---

## Reward Function

Rewards are shaped to provide signal at every step — not just at episode end:

| Event | Reward |
|-------|--------|
| Assign doctor to critical patient | +6.0 |
| Assign doctor to moderate patient | +3.0 |
| Assign doctor to mild patient | +1.0 |
| Immediate response to critical patient | +2.0 bonus |
| Transport critical patient to hospital | +5.0 |
| Transport moderate patient to hospital | +3.0 |
| Dispatch rescue team | +4.0 per critical + 1.0 per other |
| ICU overflow attempt | -3.0 |
| Wait with critical patients unattended | -3.0 |
| Patient neglected (critical, unallocated) | -2.0 per step |
| Patient neglected (moderate, unallocated) | -1.0 per step |
| Patient death | -8.0 |

---

## Patient Severity & Death Thresholds

| Severity | Death Threshold | Priority |
|----------|----------------|---------|
| Critical | 5 steps without care | Highest |
| Moderate | 16 steps without care | Medium |
| Mild | 20 steps without care | Lowest |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Environment info and available endpoints |
| `/reset` | POST | Start new episode for a given task |
| `/step` | POST | Take one action in the environment |
| `/state` | GET | Get full current environment state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | POST | Get score for current episode |
| `/baseline` | POST | Run rule-based agent on all tasks |

---

## Setup & Installation

### Local Development
```bash
# Clone and setup
git clone https://huggingface.co/spaces/harshitjaiswal08/nerc-env
cd nerc-env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run server
uvicorn server:app --host 0.0.0.0 --port 7860 --reload

# Run tests
pytest test_env.py -v
```

### Docker
```bash
docker build -t nerc-env .
docker run -p 7860:7860 nerc-env
```

---

## Usage Example
```python
import requests

BASE = "https://harshitjaiswal08-nerc-env.hf.space"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task_id": "task_1"}).json()
print(f"Patients: {len(obs['patients'])} | Time left: {obs['time_left']}")

# Take actions
action = {
    "action_type": "assign_doctor",
    "patient_id": "p1",
    "doctor_id": "d1"
}
result = requests.post(f"{BASE}/step", json={"action": action}).json()
print(f"Reward: {result['reward']} | Done: {result['done']}")

# Get final score
score = requests.post(f"{BASE}/grader").json()
print(f"Score: {score['score']} | {score['summary']}")

# Run baseline agent on all tasks
baseline = requests.post(f"{BASE}/baseline").json()
print(f"Baseline scores: {baseline['baseline_scores']}")
print(f"Average: {baseline['average']}")
```

---

## Testing
```bash
pip install pytest
pytest test_env.py -v
```

27 tests covering:
- `reset()` — all 5 tasks, clean state, invalid task handling
- `step()` — doctor assignment, ambulance dispatch, time decrement, done detection
- Reward signals — severity ordering, ICU overflow penalties, invalid actions
- `state()` — dict format, consistency with observation
- Grader — score range, breakdown fields, perfect score detection
- Tasks — all 5 exist, required fields, difficulty progression, disaster rescue teams

---

## Project Structure
```
nerc-env/
├── env.py          # Core environment — step(), reset(), state()
├── models.py       # Pydantic typed models — Action, Observation, State
├── tasks.py        # 5 task definitions with scenarios
├── reward.py       # Shaped reward function
├── grader.py       # Episode scoring 0.0–1.0
├── server.py       # FastAPI server — all endpoints
├── inference.py    # Baseline inference script
├── test_env.py     # pytest test suite — 27 tests
├── openenv.yaml    # OpenEnv spec metadata
├── Dockerfile      # Container definition
└── requirements.txt
```

---

## Environment Design Principles

1. **Real-world fidelity** — Models actual emergency response workflows used by hospitals and disaster management agencies
2. **Shaped rewards** — Signal at every step, not just episode end
3. **Difficulty progression** — Easy hospital triage → Expert multi-city coordination
4. **Deterministic grading** — Same episode always produces same score
5. **Partial observability** — Agent sees full state but must act under time pressure

---

## Live Demo

Environment is live at:
`https://harshitjaiswal08-nerc-env.hf.space`

Test it directly:
```bash
curl https://harshitjaiswal08-nerc-env.hf.space/
curl -X POST https://harshitjaiswal08-nerc-env.hf.space/baseline
```
