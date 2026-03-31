from models import (
    EnvState, Mode, Patient, Hospital, Ambulance,
    Doctor, RescueTeam, Action, Observation, RewardInfo,
    PatientSeverity, AmbulanceStatus, DoctorStatus, RescueTeamStatus
)
from reward import compute_reward
from tasks import TASKS
from typing import Tuple
import copy

class NERCEnv:
    def __init__(self):
        self.state: EnvState = None

    # ─── reset() ──────────────────────────────────────────────────────────────
    def reset(self, task_id: str = "task_1") -> Observation:
        task = TASKS.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")

        self.state = EnvState(
            mode=task["mode"],
            patients=[Patient(**p) for p in task["patients"]],
            hospitals=[Hospital(**h) for h in task["hospitals"]],
            ambulances=[Ambulance(**a) for a in task["ambulances"]],
            doctors=[Doctor(**d) for d in task["doctors"]],
            rescue_teams=[RescueTeam(**r) for r in task["rescue_teams"]],
            time_left=task["time_limit"],
            current_task=task_id
        )
        return self._get_observation()

    # ─── step() ───────────────────────────────────────────────────────────────
    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")
        if self.state.done:
            raise RuntimeError("Episode is done. Call reset()")

        reward, reason = compute_reward(self.state, action)
        self._apply_action(action)
        self._tick()

        self.state.cumulative_reward += reward
        self.state.done = self.state.time_left <= 0 or self._all_patients_resolved()

        obs = self._get_observation()
        obs.last_reward = reward
        obs.done = self.state.done

        info = {
            "reason": reason,
            "cumulative_reward": self.state.cumulative_reward,
            "time_left": self.state.time_left
        }
        return obs, reward, self.state.done, info

    # ─── state() ──────────────────────────────────────────────────────────────
    def state_snapshot(self) -> dict:
        if self.state is None:
            raise RuntimeError("Call reset() first")
        return self.state.model_dump()

    # ─── Internal helpers ─────────────────────────────────────────────────────
    def _apply_action(self, action: Action):
        s = self.state

        if action.action_type == "assign_doctor":
            patient = self._get_patient(action.patient_id)
            doctor = self._get_doctor(action.doctor_id)
            if patient and doctor and doctor.status == DoctorStatus.available:
                doctor.status = DoctorStatus.busy
                doctor.assigned_patient = action.patient_id
                patient.assigned_doctor = action.doctor_id
                patient.steps_without_care = 0

        elif action.action_type == "dispatch_ambulance":
            patient = self._get_patient(action.patient_id)
            ambulance = self._get_ambulance(action.ambulance_id)
            hospital = self._get_hospital(action.hospital_id)
            if patient and ambulance and hospital:
                if ambulance.status == AmbulanceStatus.available and hospital.current_patients < hospital.icu_capacity:
                    ambulance.status = AmbulanceStatus.dispatched
                    ambulance.assigned_patient = action.patient_id
                    patient.assigned_hospital = action.hospital_id
                    hospital.current_patients += 1

        elif action.action_type == "transfer_patient":
            patient = self._get_patient(action.patient_id)
            new_hospital = self._get_hospital(action.hospital_id)
            if patient and new_hospital and new_hospital.current_patients < new_hospital.icu_capacity:
                old_hospital = self._get_hospital(patient.assigned_hospital)
                if old_hospital:
                    old_hospital.current_patients -= 1
                patient.assigned_hospital = action.hospital_id
                new_hospital.current_patients += 1

        elif action.action_type == "dispatch_rescue":
            team = self._get_rescue_team(action.rescue_team_id)
            if team and team.status == RescueTeamStatus.available:
                team.status = RescueTeamStatus.deployed
                team.assigned_location = action.location
                # rescue all unrescued patients
                for p in s.patients:
                    if not p.rescued:
                        p.rescued = True

        elif action.action_type == "prioritize_patient":
            patient = self._get_patient(action.patient_id)
            if patient:
                patient.steps_without_care = 0

    def _tick(self):
        """Advance time and update patient states."""
        self.state.time_left -= 1
        for patient in self.state.patients:
            if patient.alive:
                if patient.assigned_doctor is None:
                    patient.steps_without_care += 1
                # Critical patients die after 5 steps without care
                if patient.severity == PatientSeverity.critical and patient.steps_without_care >= 5:
                    patient.alive = False
                # Moderate patients die after 8 steps without care
                elif patient.severity == PatientSeverity.moderate and patient.steps_without_care >= 8:
                    patient.alive = False

        # Free up doctors whose patients are now stable (assigned to hospital)
        for doctor in self.state.doctors:
            if doctor.status == DoctorStatus.busy:
                patient = self._get_patient(doctor.assigned_patient)
                if patient and patient.assigned_hospital:
                    doctor.status = DoctorStatus.available
                    doctor.assigned_patient = None

        # Free up ambulances after dispatch
        for amb in self.state.ambulances:
            if amb.status == AmbulanceStatus.dispatched:
                amb.status = AmbulanceStatus.available
                amb.assigned_patient = None

    def _all_patients_resolved(self) -> bool:
        return all(
            (p.assigned_hospital is not None) or (not p.alive)
            for p in self.state.patients
        )

    def _get_observation(self) -> Observation:
        return Observation(
            mode=self.state.mode,
            patients=self.state.patients,
            hospitals=self.state.hospitals,
            ambulances=self.state.ambulances,
            doctors=self.state.doctors,
            rescue_teams=self.state.rescue_teams,
            time_left=self.state.time_left,
            done=self.state.done
        )

    def _get_patient(self, pid): 
        return next((p for p in self.state.patients if p.id == pid), None)
    def _get_doctor(self, did): 
        return next((d for d in self.state.doctors if d.id == did), None)
    def _get_ambulance(self, aid): 
        return next((a for a in self.state.ambulances if a.id == aid), None)
    def _get_hospital(self, hid): 
        return next((h for h in self.state.hospitals if h.id == hid), None)
    def _get_rescue_team(self, rid): 
        return next((r for r in self.state.rescue_teams if r.id == rid), None)