import os
import json
import requests

BASE_URL = os.getenv("NERC_ENV_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def get_action_from_llm(obs: dict) -> dict:
    """
    Use OpenAI API if key is available, otherwise fall back to rule-based agent.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            prompt = f"""You are an emergency response coordinator.
Current state: {json.dumps(obs, indent=2)}
Respond with a single JSON action object only.
Fields: action_type, patient_id, doctor_id, ambulance_id, hospital_id, rescue_team_id, location
action_type must be one of: assign_doctor, dispatch_ambulance, transfer_patient, dispatch_rescue, prioritize_patient, wait
Prioritize critical patients. Avoid ICU overflow."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"  LLM error: {e} — falling back to rule-based agent")

    return get_rule_based_action(obs)


def get_rule_based_action(obs: dict) -> dict:
    """
    Improved rule-based agent — strictly checks capacity before dispatching.
    """
    patients   = obs.get("patients", [])
    doctors    = obs.get("doctors", [])
    ambulances = obs.get("ambulances", [])
    hospitals  = obs.get("hospitals", [])
    rescue     = obs.get("rescue_teams", [])

    available_doctors    = [d for d in doctors    if d["status"] == "available"]
    available_ambulances = [a for a in ambulances if a["status"] == "available"]
    available_rescue     = [r for r in rescue      if r["status"] == "available"]
    alive_patients       = [p for p in patients    if p["alive"]]

    # strictly only hospitals with remaining space
    hospitals_with_space = [
        h for h in hospitals
        if h["current_patients"] < h["icu_capacity"]
    ]

    # patients not yet allocated to any hospital
    unallocated = [
        p for p in alive_patients
        if p["assigned_hospital"] is None
    ]

    # priority 1: assign doctor to critical patient
    for p in alive_patients:
        if p["severity"] == "critical" and p["assigned_doctor"] is None:
            if available_doctors:
                return {
                    "action_type": "assign_doctor",
                    "patient_id": p["id"],
                    "doctor_id": available_doctors[0]["id"]
                }

    # priority 2: dispatch rescue teams
    if available_rescue:
        unrescued = [p for p in alive_patients if not p["rescued"]]
        if unrescued:
            return {
                "action_type": "dispatch_rescue",
                "rescue_team_id": available_rescue[0]["id"],
                "location": "disaster_zone"
            }

    # priority 3: dispatch ambulance ONLY if hospital has space AND patient not yet allocated
    if unallocated and available_ambulances and hospitals_with_space:
        # sort by severity
        priority_order = {"critical": 0, "moderate": 1, "mild": 2}
        unallocated_sorted = sorted(
            unallocated,
            key=lambda p: priority_order.get(p["severity"], 3)
        )
        # pick hospital with most space
        best_hospital = max(
            hospitals_with_space,
            key=lambda h: h["icu_capacity"] - h["current_patients"]
        )
        return {
            "action_type": "dispatch_ambulance",
            "patient_id": unallocated_sorted[0]["id"],
            "ambulance_id": available_ambulances[0]["id"],
            "hospital_id": best_hospital["id"]
        }

    # priority 4: assign doctor to moderate patient
    for p in alive_patients:
        if p["severity"] == "moderate" and p["assigned_doctor"] is None:
            if available_doctors:
                return {
                    "action_type": "assign_doctor",
                    "patient_id": p["id"],
                    "doctor_id": available_doctors[0]["id"]
                }

    # priority 5: assign doctor to mild patient
    for p in alive_patients:
        if p["severity"] == "mild" and p["assigned_doctor"] is None:
            if available_doctors:
                return {
                    "action_type": "assign_doctor",
                    "patient_id": p["id"],
                    "doctor_id": available_doctors[0]["id"]
                }

    return {"action_type": "wait"}


def run_episode(task_id: str) -> dict:
    """Run one full episode."""

    res = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    obs = res.json()
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < 50:
        action = get_action_from_llm(obs)

        try:
            step_res = requests.post(
                f"{BASE_URL}/step",
                json={"action": action}
            ).json()

            # handle both response formats safely
            if "observation" in step_res:
                obs         = step_res["observation"]
                reward      = step_res.get("reward", 0)
                done        = step_res.get("done", False)
            elif "detail" in step_res:
                print(f"  Server error: {step_res['detail']}")
                break
            else:
                print(f"  Unexpected response: {step_res}")
                break

            total_reward += reward
            steps += 1
            print(f"  Step {steps}: action={action.get('action_type')} reward={reward}")

        except Exception as e:
            print(f"  Step error: {e}")
            break

    # get final grade — works even if episode not fully done
    try:
        grade_res = requests.post(f"{BASE_URL}/grader").json()
        grade = grade_res.get("score", 0.0)
        summary = grade_res.get("summary", "no summary")
    except Exception as e:
        print(f"  Grader error: {e}")
        grade = 0.0
        summary = "grader failed"

    return {
        "task_id":      task_id,
        "steps":        steps,
        "total_reward": round(total_reward, 2),
        "grade":        grade,
        "summary":      summary
    }


def run_baseline():
    """Run baseline across all 3 tasks."""
    print("=" * 60)
    print("NERC-Env Baseline Inference Script")
    print(f"Environment: {BASE_URL}")
    print("=" * 60)

    tasks = ["task_1", "task_2", "task_3"]
    results = []

    for task_id in tasks:
        print(f"\nRunning {task_id}...")
        result = run_episode(task_id)
        results.append(result)
        print(f"  Grade:   {result['grade']}")
        print(f"  Summary: {result['summary']}")

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    for r in results:
        print(f"{r['task_id']}: {r['grade']} ({r['steps']} steps, reward={r['total_reward']})")

    avg = sum(r["grade"] for r in results) / len(results)
    print(f"\nAverage score: {round(avg, 4)}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_baseline()