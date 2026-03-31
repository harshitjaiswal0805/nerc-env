from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

# ─── Enums ────────────────────────────────────────────────────────────────────

class Mode(str, Enum):
    hospital = "hospital"
    disaster = "disaster"

class PatientSeverity(str, Enum):
    critical = "critical"
    moderate = "moderate"
    mild = "mild"

class AmbulanceStatus(str, Enum):
    available = "available"
    dispatched = "dispatched"

class DoctorStatus(str, Enum):
    available = "available"
    busy = "busy"

class RescueTeamStatus(str, Enum):
    available = "available"
    deployed = "deployed"

# ─── Entity Models ────────────────────────────────────────────────────────────

class Patient(BaseModel):
    id: str
    severity: PatientSeverity
    assigned_doctor: Optional[str] = None
    assigned_hospital: Optional[str] = None
    rescued: bool = False
    alive: bool = True
    steps_without_care: int = 0

class Hospital(BaseModel):
    id: str
    icu_capacity: int
    current_patients: int = 0

    @property
    def has_capacity(self) -> bool:
        return self.current_patients < self.icu_capacity

class Ambulance(BaseModel):
    id: str
    status: AmbulanceStatus = AmbulanceStatus.available
    assigned_patient: Optional[str] = None

class Doctor(BaseModel):
    id: str
    specialty: str
    status: DoctorStatus = DoctorStatus.available
    assigned_patient: Optional[str] = None

class RescueTeam(BaseModel):
    id: str
    status: RescueTeamStatus = RescueTeamStatus.available
    assigned_location: Optional[str] = None

# ─── Action Model (what the agent sends) ──────────────────────────────────────

class Action(BaseModel):
    action_type: Literal[
        "assign_doctor",
        "dispatch_ambulance",
        "transfer_patient",
        "dispatch_rescue",
        "prioritize_patient",
        "wait"
    ]
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    ambulance_id: Optional[str] = None
    hospital_id: Optional[str] = None
    rescue_team_id: Optional[str] = None
    location: Optional[str] = None

# ─── Observation Model (what the agent sees) ──────────────────────────────────

class Observation(BaseModel):
    mode: Mode
    patients: List[Patient]
    hospitals: List[Hospital]
    ambulances: List[Ambulance]
    doctors: List[Doctor]
    rescue_teams: List[RescueTeam]
    time_left: int
    last_reward: float = 0.0
    done: bool = False

# ─── Reward Model ─────────────────────────────────────────────────────────────

class RewardInfo(BaseModel):
    reward: float
    reason: str
    cumulative_reward: float

# ─── State Model (full internal state) ───────────────────────────────────────

class EnvState(BaseModel):
    mode: Mode
    patients: List[Patient] = Field(default_factory=list)
    hospitals: List[Hospital] = Field(default_factory=list)
    ambulances: List[Ambulance] = Field(default_factory=list)
    doctors: List[Doctor] = Field(default_factory=list)
    rescue_teams: List[RescueTeam] = Field(default_factory=list)
    time_left: int = 20
    cumulative_reward: float = 0.0
    done: bool = False
    current_task: Optional[str] = None