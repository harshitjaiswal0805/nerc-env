import os
import json
import requests
from openai import OpenAI

# ─── Required environment variables per hackathon spec ────────────────────────
BASE_URL      = os.getenv("NERC_ENV_URL",  "http://localhost:7860")
API_BASE_URL  = os.getenv("API_BASE_URL",  "https://api.openai.com/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",    "gpt-4o-mini")
HF_TOKEN      = os.getenv("HF_TOKEN",      "")

# ─── OpenAI client using required variables ───────────────────────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

SYSTEM_PROMPT = """You are an emergency response coordinator AI.
You will receive the current state of an emergency environment and must decide the best action.

Always respond with a single JSON object with this exact structure:
{
    "action_type": "assign_doctor|dispatch_ambulance|transfer_patient|dispatch_rescue|prioritize_patient|wait",
    "patient_id": "p1",
    "doctor_id": "d1",
    "ambulance_id": "a1",
    "hospital_id": "h1",
    "rescue_team_id": "r1",
    "location": "zone_a"
}

Only include fields relevant to your chosen action_type.
Prioritize critical patients. Avoid ICU overflow. Save as many lives as possible."""


def get_action_from_llm(obs: dict) -> dict:
    """
    Use OpenAI-compatible API if HF_TOKEN is available,
    otherwise fall back to rule-based agent.
    """
    if HF_TOKEN:
        try:
            prompt = f"""Current environment state:
{json.dumps(obs, indent=2)}

What action should be taken? Respond with a single JSON action object only."""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
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
    """Deterministic rule-based agent — used when no API key is present."""
    patients   = obs.get("patients",   [])
    doctors    = obs.get("doctors",    [])
    ambulances = obs.get("ambulances", [])
    hospitals  = obs.get("hospitals",  [])
    rescue     = obs.get("rescue_teams", [])

    available_doctors    = [d for d in doctors    if d["status"] == "available"]
    available_ambulances = [a for a in ambulances if a["status"] == "available"]
    available_rescue     = [r for r in rescue      if r["status"] == "available"]
    alive                = [p for p in patients    if p["alive"]]
    hospitals_with_space = [h for h in hospitals   if h["current_patients"] < h["icu_capacity"]]
    unallocated          = [p for p in alive        if p["assigned_hospital"] is None]

    # priority 1: assign doctor to critical patient
    for p in alive:
        if p["severity"] == "critical" and p["assigned_doctor"] is None and available_doctors:
            return {
                "action_type": "assign_doctor",
                "patient_id":  p["id"],
                "doctor_id":   available_doctors[0]["id"]
            }

    # priority 2: dispatch rescue teams
    if available_rescue:
        unrescued = [p for p in alive if not p["rescued"]]
        if unrescued:
            return {
                "action_type":    "dispatch_rescue",
                "rescue_team_id": available_rescue[0]["id"],
                "location":       "disaster_zone"
            }

    # priority 3: dispatch ambulance only if hospital has space
    if unallocated and available_ambulances and hospitals_with_space:
        priority     = {"critical": 0, "moderate": 1, "mild": 2}
        best_patient = sorted(unallocated, key=lambda p: priority.get(p["severity"], 3))[0]
        best_hosp    = max(hospitals_with_space, key=lambda h: h["icu_capacity"] - h["current_patients"])
        return {
            "action_type":  "dispatch_ambulance",
            "patient_id":   best_patient["id"],
            "ambulance_id": available_ambulances[0]["id"],
            "hospital_id":  best_hosp["id"]
        }

    # priority 4: assign doctor to moderate patient
    for p in alive:
        if p["severity"] == "moderate" and p["assigned_doctor"] is None and available_doctors:
            return {
                "action_type": "assign_doctor",
                "patient_id":  p["id"],
                "doctor_id":   available_doctors[0]["id"]
            }

    # priority 5: assign doctor to mild patient
    for p in alive:
        if p["severity"] == "mild" and p["assigned_doctor"] is None and available_doctors:
            return {
                "action_type": "assign_doctor",
                "patient_id":  p["id"],
                "doctor_id":   available_doctors[0]["id"]
            }

    return {"action_type": "wait"}


def run_episode(task_id: str) -> dict:
    """Run one full episode."""
    res  = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    obs  = res.json()
    done = False
    steps       = 0
    total_reward = 0.0

    while not done and steps < 50:
        action = get_action_from_llm(obs)
        try:
            step_res = requests.post(
                f"{BASE_URL}/step",
                json={"action": action}
            ).json()

            if "observation" in step_res:
                obs          = step_res["observation"]
                reward       = step_res.get("reward", 0)
                done         = step_res.get("done", False)
                total_reward += reward
                steps        += 1
                print(f"  Step {steps}: action={action.get('action_type')} reward={reward}")
            elif "detail" in step_res:
                print(f"  Server error: {step_res['detail']}")
                break
        except Exception as e:
            print(f"  Step error: {e}")
            break

    try:
        grade_res = requests.post(f"{BASE_URL}/grader").json()
        grade   = grade_res.get("score",   0.0)
        summary = grade_res.get("summary", "no summary")
    except Exception as e:
        grade   = 0.0
        summary = f"grader error: {e}"

    return {
        "task_id":      task_id,
        "steps":        steps,
        "total_reward": round(total_reward, 2),
        "grade":        grade,
        "summary":      summary
    }


def run_baseline():
    """Run baseline across all tasks and print results."""
    print("=" * 60)
    print("NERC-Env Baseline Inference Script")
    print(f"Environment : {BASE_URL}")
    print(f"API base    : {API_BASE_URL}")
    print(f"Model       : {MODEL_NAME}")
    print("=" * 60)

    tasks   = ["task_1", "task_2", "task_3", "task_4", "task_5"]
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
