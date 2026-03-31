from models import EnvState, Action, PatientSeverity
from typing import Tuple

def compute_reward(state: EnvState, action: Action) -> Tuple[float, str]:
    """
    Returns (reward, reason) for a given action in the current state.
    Rewards are shaped to give signal at every step, not just episode end.
    """
    reward = 0.0
    reasons = []

    # ─── Action: assign_doctor ─────────────────────────────────────────────────
    if action.action_type == "assign_doctor":
        patient = _get_patient(state, action.patient_id)
        doctor = _get_doctor(state, action.doctor_id)

        if patient is None or doctor is None:
            return -2.0, "Invalid assign_doctor — patient or doctor not found"

        if not patient.alive:
            return -1.0, "Tried to assign doctor to dead patient"

        if doctor.status == "busy":
            return -1.0, "Tried to assign already busy doctor"

        if patient.assigned_doctor is not None:
            return -0.5, "Patient already has a doctor"

        # Reward based on severity — critical patients matter more
        if patient.severity == PatientSeverity.critical:
            reward += 6.0
            reasons.append("assigned doctor to critical patient (+6)")
        elif patient.severity == PatientSeverity.moderate:
            reward += 3.0
            reasons.append("assigned doctor to moderate patient (+3)")
        else:
            reward += 1.0
            reasons.append("assigned doctor to mild patient (+1)")

        # Bonus for acting fast on critical patients
        if patient.severity == PatientSeverity.critical and patient.steps_without_care == 0:
            reward += 2.0
            reasons.append("immediate response to critical patient (+2)")

    # ─── Action: dispatch_ambulance ────────────────────────────────────────────
    elif action.action_type == "dispatch_ambulance":
        patient = _get_patient(state, action.patient_id)
        ambulance = _get_ambulance(state, action.ambulance_id)
        hospital = _get_hospital(state, action.hospital_id)

        if patient is None or ambulance is None or hospital is None:
            return -2.0, "Invalid dispatch_ambulance — entity not found"

        if not patient.alive:
            return -1.0, "Dispatching ambulance for dead patient"

        if ambulance.status == "dispatched":
            return -1.0, "Ambulance already dispatched"

        if not hospital.has_capacity:
            reward -= 3.0
            reasons.append("ICU overflow attempt (-3)")
        else:
            if patient.severity == PatientSeverity.critical:
                reward += 5.0
                reasons.append("transported critical patient (+5)")
            elif patient.severity == PatientSeverity.moderate:
                reward += 3.0
                reasons.append("transported moderate patient (+3)")
            else:
                reward += 1.0
                reasons.append("transported mild patient (+1)")

            # Bonus for efficient routing: critical to hospital with most space
            if patient.severity == PatientSeverity.critical:
                space = hospital.icu_capacity - hospital.current_patients
                if space >= 2:
                    reward += 1.0
                    reasons.append("good hospital choice (+1)")

    # ─── Action: transfer_patient ──────────────────────────────────────────────
    elif action.action_type == "transfer_patient":
        patient = _get_patient(state, action.patient_id)
        new_hospital = _get_hospital(state, action.hospital_id)

        if patient is None or new_hospital is None:
            return -2.0, "Invalid transfer — entity not found"

        if not new_hospital.has_capacity:
            return -3.0, "Transfer to full hospital — ICU overflow"

        if patient.assigned_hospital == action.hospital_id:
            return -0.5, "Useless transfer to same hospital"

        reward += 2.0
        reasons.append("valid patient transfer (+2)")

    # ─── Action: dispatch_rescue ───────────────────────────────────────────────
    elif action.action_type == "dispatch_rescue":
        team = _get_rescue_team(state, action.rescue_team_id)

        if team is None:
            return -2.0, "Invalid rescue team"

        if team.status == "deployed":
            return -1.0, "Rescue team already deployed"

        # Count unrescued patients
        unrescued = [p for p in state.patients if not p.rescued and p.alive]
        critical_unrescued = [p for p in unrescued if p.severity == PatientSeverity.critical]

        if len(unrescued) == 0:
            return -0.5, "No patients to rescue"

        reward += 4.0 * len(critical_unrescued) + 1.0 * (len(unrescued) - len(critical_unrescued))
        reasons.append(f"rescue dispatch covers {len(unrescued)} patients (+{reward:.1f})")

    # ─── Action: prioritize_patient ────────────────────────────────────────────
    elif action.action_type == "prioritize_patient":
        patient = _get_patient(state, action.patient_id)

        if patient is None:
            return -2.0, "Patient not found for prioritization"

        if not patient.alive:
            return -1.0, "Cannot prioritize dead patient"

        if patient.severity == PatientSeverity.critical and patient.steps_without_care > 1:
            reward += 2.0
            reasons.append("prioritized at-risk critical patient (+2)")
        else:
            reward += 0.5
            reasons.append("prioritized patient (+0.5)")

    # ─── Action: wait ──────────────────────────────────────────────────────────
    elif action.action_type == "wait":
        # Penalize waiting if there are unattended critical patients
        critical_unattended = [
            p for p in state.patients
            if p.alive and p.severity == PatientSeverity.critical
            and p.assigned_doctor is None
        ]
        if critical_unattended:
            reward -= 3.0
            reasons.append(f"waited with {len(critical_unattended)} critical patients unattended (-3)")
        else:
            reward += 0.0
            reasons.append("wait with no critical patients pending (0)")

    # ─── Passive penalties applied every step ─────────────────────────────────
    # ─── Passive penalties applied every step ─────────────────────────────────
    for patient in state.patients:
        if patient.alive:
            # only penalize if patient is NOT yet in a hospital
            if patient.assigned_hospital is None:
                if patient.severity == PatientSeverity.critical and patient.steps_without_care >= 2:
                    reward -= 2.0
                    reasons.append(f"patient {patient.id} critical and neglected (-2)")
                if patient.severity == PatientSeverity.moderate and patient.steps_without_care >= 4:
                    reward -= 1.0
                    reasons.append(f"patient {patient.id} moderate and neglected (-1)")

        if not patient.alive:
            # only penalize death once — when steps_without_care just hit the limit
            if patient.severity == PatientSeverity.critical and patient.steps_without_care == 3:
                reward -= 8.0
                reasons.append(f"patient {patient.id} died — critical neglect (-8)")
            elif patient.severity == PatientSeverity.moderate and patient.steps_without_care == 6:
                reward -= 8.0
                reasons.append(f"patient {patient.id} died — moderate neglect (-8)")
# ─── Lookup helpers ────────────────────────────────────────────────────────────

def _get_patient(state, pid):
    return next((p for p in state.patients if p.id == pid), None)

def _get_doctor(state, did):
    return next((d for d in state.doctors if d.id == did), None)

def _get_ambulance(state, aid):
    return next((a for a in state.ambulances if a.id == aid), None)

def _get_hospital(state, hid):
    return next((h for h in state.hospitals if h.id == hid), None)

def _get_rescue_team(state, rid):
    return next((r for r in state.rescue_teams if r.id == rid), None)