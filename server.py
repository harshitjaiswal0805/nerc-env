from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from env import NERCEnv
from models import Action
from grader import grade_episode
from tasks import TASKS
import os

app = FastAPI(
    title="NERC-Env",
    description="National Emergency Response & Care Environment — OpenEnv compliant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── One environment instance per server ──────────────────────────────────────
env = NERCEnv()

# ─── Request bodies ───────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"

class StepRequest(BaseModel):
    action: Action

# ─── Core OpenEnv endpoints ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "NERC-Env",
        "version": "1.0.0",
        "description": "National Emergency Response & Care Environment",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"]
    }

@app.post("/reset")
def reset(request: ResetRequest = None):
    task_id = request.task_id if request else "task_1"
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    obs = env.reset(task_id)
    return obs.model_dump()

@app.post("/step")
def step(request: StepRequest):
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    if env.state.done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset")
    try:
        obs, reward, done, info = env.step(request.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state_snapshot()

# ─── Required extra endpoints ─────────────────────────────────────────────────

@app.get("/tasks")
def get_tasks():
    """Returns list of tasks and their action schema."""
    return {
        "tasks": [
            {
                "task_id": tid,
                "description": t["description"],
                "difficulty": t["difficulty"],
                "mode": t["mode"],
                "time_limit": t["time_limit"],
            }
            for tid, t in TASKS.items()
        ],
        "action_schema": {
            "action_type": "one of: assign_doctor | dispatch_ambulance | transfer_patient | dispatch_rescue | prioritize_patient | wait",
            "patient_id": "string (optional)",
            "doctor_id": "string (optional)",
            "ambulance_id": "string (optional)",
            "hospital_id": "string (optional)",
            "rescue_team_id": "string (optional)",
            "location": "string (optional)",
        }
    }

@app.post("/grader")
def grader():
    """Returns grader score after an episode is completed."""
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    result = grade_episode(env.state)
    return result

@app.post("/baseline")
def baseline():
    """
    Runs a simple rule-based agent across all 3 tasks.
    Returns reproducible baseline scores.
    """
    scores = {}
    for task_id in TASKS:
        obs = env.reset(task_id)
        done = False
        while not done:
            action = _rule_based_agent(obs)
            obs, _, done, _ = env.step(action)
        result = grade_episode(env.state)
        scores[task_id] = result["score"]

    return {
        "baseline_scores": scores,
        "average": round(sum(scores.values()) / len(scores), 4)
    }

# ─── Simple rule-based agent for baseline ─────────────────────────────────────
def _rule_based_agent(obs):
    from models import Action

    patients   = obs.patients
    doctors    = obs.doctors
    ambulances = obs.ambulances
    hospitals  = obs.hospitals
    rescue     = obs.rescue_teams

    available_doctors    = [d for d in doctors    if d.status == "available"]
    available_ambulances = [a for a in ambulances if a.status == "available"]
    available_rescue     = [r for r in rescue      if r.status == "available"]
    alive_patients       = [p for p in patients    if p.alive]

    # hospitals with space
    hospitals_with_space = [
        h for h in hospitals
        if h.current_patients < h.icu_capacity
    ]

    # unallocated alive patients
    unallocated = [
        p for p in alive_patients
        if p.assigned_hospital is None
    ]

    # priority 1: assign doctor to critical patient
    for p in alive_patients:
        if p.severity == "critical" and p.assigned_doctor is None:
            if available_doctors:
                return Action(
                    action_type="assign_doctor",
                    patient_id=p.id,
                    doctor_id=available_doctors[0].id
                )

    # priority 2: dispatch rescue teams
    if available_rescue:
        unrescued = [p for p in alive_patients if not p.rescued]
        if unrescued:
            return Action(
                action_type="dispatch_rescue",
                rescue_team_id=available_rescue[0].id,
                location="disaster_zone"
            )

    # priority 3: dispatch ambulance ONLY to hospital with space
    if unallocated and available_ambulances and hospitals_with_space:
        priority_order = {"critical": 0, "moderate": 1, "mild": 2}
        unallocated_sorted = sorted(
            unallocated,
            key=lambda p: priority_order.get(p.severity, 3)
        )
        # pick hospital with most remaining space
        best_hospital = max(
            hospitals_with_space,
            key=lambda h: h.icu_capacity - h.current_patients
        )
        return Action(
            action_type="dispatch_ambulance",
            patient_id=unallocated_sorted[0].id,
            ambulance_id=available_ambulances[0].id,
            hospital_id=best_hospital.id
        )

    # priority 4: assign doctor to moderate patient
    for p in alive_patients:
        if p.severity == "moderate" and p.assigned_doctor is None:
            if available_doctors:
                return Action(
                    action_type="assign_doctor",
                    patient_id=p.id,
                    doctor_id=available_doctors[0].id
                )

    # priority 5: transfer from full hospital to one with space
    if hospitals_with_space:
        full_hospital_ids = [
            h.id for h in hospitals
            if h.current_patients >= h.icu_capacity
        ]
        for p in alive_patients:
            if p.assigned_hospital in full_hospital_ids:
                return Action(
                    action_type="transfer_patient",
                    patient_id=p.id,
                    hospital_id=hospitals_with_space[0].id
                )

    return Action(action_type="wait")
