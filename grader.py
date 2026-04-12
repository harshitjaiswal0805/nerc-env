from models import EnvState, PatientSeverity
from typing import Dict

def grade_episode(state: EnvState) -> Dict:
    """
    Scores a completed episode from 0.0 to 1.0.
    Called after episode ends (done=True).
    Returns score + breakdown for transparency.
    """

    patients = state.patients
    total = len(patients)

    if total == 0:
        return {"score": 0.0, "breakdown": {}, "summary": "No patients found"}

    # ─── Count outcomes ────────────────────────────────────────────────────────
    alive = [p for p in patients if p.alive]
    dead  = [p for p in patients if not p.alive]

    critical_total    = [p for p in patients if p.severity == PatientSeverity.critical]
    critical_saved    = [p for p in alive    if p.severity == PatientSeverity.critical]

    moderate_total    = [p for p in patients if p.severity == PatientSeverity.moderate]
    moderate_saved    = [p for p in alive    if p.severity == PatientSeverity.moderate]

    mild_total        = [p for p in patients if p.severity == PatientSeverity.mild]
    mild_saved        = [p for p in alive    if p.severity == PatientSeverity.mild]

    allocated         = [p for p in patients if p.assigned_hospital is not None]
    rescued           = [p for p in patients if p.rescued]

    # ─── Component scores (each 0.0–1.0) ──────────────────────────────────────

    # 1. Survival rate — weighted by severity
    #    Critical patients count 3x, moderate 2x, mild 1x
    weighted_saved  = (len(critical_saved) * 3 +
                       len(moderate_saved) * 2 +
                       len(mild_saved)     * 1)
    weighted_total  = (len(critical_total) * 3 +
                       len(moderate_total) * 2 +
                       len(mild_total)     * 1)

    survival_score  = weighted_saved / weighted_total if weighted_total > 0 else 0.0

    # 2. Allocation efficiency — how many patients got to a hospital
    allocation_score = len(allocated) / total

    # 3. Critical response — did critical patients get a doctor assigned?
    critical_treated = [
        p for p in critical_total
        if p.assigned_doctor is not None or p.assigned_hospital is not None
    ]
    critical_score = (
        len(critical_treated) / len(critical_total)
        if critical_total else 1.0
    )

    # 4. Rescue rate — only matters in disaster mode
    if state.mode == "disaster" and total > 0:
        rescue_score = len(rescued) / total
    else:
        rescue_score = 1.0  # not applicable in hospital mode

    # 5. Time efficiency — reward finishing early
    task_time_limits = {"task_1": 15, "task_2": 20, "task_3": 25}
    time_limit = task_time_limits.get(state.current_task, 20)
    time_used = time_limit - state.time_left
    time_score = max(0.0, 1.0 - (time_used / time_limit) * 0.5)

    # ─── Weighted final score ──────────────────────────────────────────────────
    # Survival is the most important signal
    final_score = (
        survival_score   * 0.40 +
        critical_score   * 0.25 +
        allocation_score * 0.20 +
        rescue_score     * 0.10 +
        time_score       * 0.05
    )

    # Hard penalty: if any critical patient died, cap score at 0.6
    if len(critical_saved) < len(critical_total):
        final_score = min(final_score, 0.6)

    # Hard penalty: if more than half of all patients died, cap at 0.3
    if len(dead) > total / 2:
        final_score = min(final_score, 0.3)

    final_score = round(max(0.001, min(0.999, final_score)), 4)

    breakdown = {
        "survival_score":   round(survival_score,   4),
        "critical_score":   round(critical_score,   4),
        "allocation_score": round(allocation_score, 4),
        "rescue_score":     round(rescue_score,     4),
        "time_score":       round(time_score,       4),
        "patients_alive":   len(alive),
        "patients_dead":    len(dead),
        "critical_saved":   f"{len(critical_saved)}/{len(critical_total)}",
        "moderate_saved":   f"{len(moderate_saved)}/{len(moderate_total)}",
        "mild_saved":       f"{len(mild_saved)}/{len(mild_total)}",
    }

    summary = (
        f"Score: {final_score} | "
        f"Alive: {len(alive)}/{total} | "
        f"Critical saved: {len(critical_saved)}/{len(critical_total)} | "
        f"Cumulative reward: {round(state.cumulative_reward, 2)}"
    )

    return {
        "score": final_score,
        "breakdown": breakdown,
        "summary": summary
    }