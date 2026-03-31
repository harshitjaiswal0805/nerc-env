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

    # Priority 1: assign doctors to critical patients first
    for patient in obs.patients:
        if patient.alive and patient.severity == "critical" and patient.assigned_doctor is None:
            for doctor in obs.doctors:
                if doctor.status == "available":
                    return Action(
                        action_type="assign_doctor",
                        patient_id=patient.id,
                        doctor_id=doctor.id
                    )

    # Priority 2: dispatch rescue teams (disaster mode)
    for team in obs.rescue_teams:
        if team.status == "available":
            return Action(
                action_type="dispatch_rescue",
                rescue_team_id=team.id,
                location="disaster_zone"
            )

    # Priority 3: dispatch ambulances to unallocated patients
    for patient in obs.patients:
        if patient.alive and patient.assigned_hospital is None:
            for amb in obs.ambulances:
                if amb.status == "available":
                    for hospital in obs.hospitals:
                        if hospital.has_capacity:
                            return Action(
                                action_type="dispatch_ambulance",
                                patient_id=patient.id,
                                ambulance_id=amb.id,
                                hospital_id=hospital.id
                            )

    # Priority 4: assign doctors to moderate patients
    for patient in obs.patients:
        if patient.alive and patient.severity == "moderate" and patient.assigned_doctor is None:
            for doctor in obs.doctors:
                if doctor.status == "available":
                    return Action(
                        action_type="assign_doctor",
                        patient_id=patient.id,
                        doctor_id=doctor.id
                    )

    return Action(action_type="wait")